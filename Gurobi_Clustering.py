import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 노드별 수요/공급 데이터
fixed_net_demand = torch.tensor([
    [-9140,   627,  8446],
    [-7567, -1162, -2792],
    [ 1284, -1567,  6216],
    [-9231,   555,  -732],
    [ 6023,  7159, -1429],
    [-5574,  1483, -1471],
    [ 6850,  5787, -7973],
    [ 1964,  1016, -2371],
    [-4610, -6848,  2185],
    [-4689,  2206,  7262],
    [31786, -8415, -1994],
    [ 4423, -7388,  6448],
    [-4949,  4541, -7432],
    [-8315,   253, 11480],
    [-3735, -1774,  1534],
    [ 1363,  4502, -8984],
    [-1678, -1208, -4724],
    [ 5795,   233, -3669]
], dtype=torch.float32)

def solve_divisible_balanced_clustering_gurobi(demand_data, num_clusters=4):
    """
    Gurobi를 사용하여 노드 분할 가능한 균형 클러스터링 문제 해결
    각 노드는 여러 클러스터에 정수 비율로 분할 가능
    
    Parameters:
    - demand_data: 노드별 수요/공급 데이터 (n_nodes x n_items)
    - num_clusters: 클러스터 개수
    
    Returns:
    - allocation: 각 노드의 클러스터별 할당량
    - cluster_sums: 각 클러스터의 수요/공급 합계 (모두 0이어야 함)
    """
    
    # 데이터 준비
    demand_np = demand_data.numpy()
    n_nodes, n_items = demand_np.shape
    
    print(f"노드 수: {n_nodes}, 품목 수: {n_items}, 클러스터 수: {num_clusters}")
    print(f"전체 수요/공급 합계: {np.sum(demand_np, axis=0)}")
    
    # Gurobi 모델 생성
    model = gp.Model("divisible_balanced_clustering")
    model.setParam('OutputFlag', 1)  # 로그 출력 활성화
    model.setParam('TimeLimit', 300)  # 5분 시간 제한
    model.setParam('MIPGap', 1e-6)    # 높은 정확도 요구
    
    # 결정 변수: x[i,k] = 노드 i가 클러스터 k에 할당되는 양 (정수)
    # 음수 수요/공급도 있으므로 하한을 설정하지 않음
    max_abs_demand = int(np.max(np.abs(demand_np))) + 1
    x = model.addVars(n_nodes, num_clusters, 
                     vtype=GRB.INTEGER, 
                     lb=-max_abs_demand, 
                     ub=max_abs_demand,
                     name="x")
    
    # 제약조건 1: 각 노드의 할당량 합은 원래 값과 같아야 함
    for i in range(n_nodes):
        for j in range(n_items):
            model.addConstr(
                gp.quicksum(x[i, k] for k in range(num_clusters)) == demand_np[i, j],
                name=f"node_conservation_{i}_{j}"
            )
    
    # 핵심 제약조건 2: 각 클러스터의 각 품목별 수요/공급 합이 정확히 0이어야 함
    for k in range(num_clusters):
        for j in range(n_items):
            cluster_sum = gp.quicksum(x[i, k] * (demand_np[i, j] / abs(demand_np[i, j]) if demand_np[i, j] != 0 else 0) 
                                    for i in range(n_nodes) if demand_np[i, j] != 0)
            # 각 품목j에 대해 클러스터 k의 합이 0
            total_contribution = gp.LinExpr()
            for i in range(n_nodes):
                if demand_np[i, j] != 0:
                    # x[i,k]는 노드 i의 품목 j에 대한 클러스터 k로의 할당량
                    # 하지만 각 노드는 모든 품목에 대해 동일한 비율로 분할되어야 함
                    pass
    
    # 더 간단한 접근: 각 노드에 대해 분할 비율을 결정하는 변수 도입
    # y[i,k] = 노드 i가 클러스터 k에 할당되는 비율 (정수, 원래 값의 분수)
    
    # 새로운 접근법: 각 노드의 각 품목을 독립적으로 분할
    model = gp.Model("divisible_balanced_clustering_v2")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 300)
    model.setParam('MIPGap', 1e-6)
    
    # 결정 변수: split[i,j,k] = 노드 i의 품목 j가 클러스터 k에 할당되는 양 (정수)
    split = {}
    for i in range(n_nodes):
        for j in range(n_items):
            for k in range(num_clusters):
                # 원래 값이 양수면 0 이상, 음수면 0 이하
                if demand_np[i, j] > 0:
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(demand_np[i, j]),
                                                name=f"split_{i}_{j}_{k}")
                elif demand_np[i, j] < 0:
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=int(demand_np[i, j]), ub=0,
                                                name=f"split_{i}_{j}_{k}")
                else:  # demand_np[i, j] == 0
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0,
                                                name=f"split_{i}_{j}_{k}")
    
    # 제약조건 1: 각 노드의 각 품목에 대한 분할의 합은 원래 값과 같아야 함
    for i in range(n_nodes):
        for j in range(n_items):
            model.addConstr(
                gp.quicksum(split[i, j, k] for k in range(num_clusters)) == demand_np[i, j],
                name=f"split_conservation_{i}_{j}"
            )
    
    # 핵심 제약조건 2: 각 클러스터의 각 품목별 합이 정확히 0이어야 함
    for k in range(num_clusters):
        for j in range(n_items):
            cluster_sum = gp.quicksum(split[i, j, k] for i in range(n_nodes))
            model.addConstr(cluster_sum == 0, name=f"perfect_balance_{k}_{j}")
    
    # 제약조건 3: 각 클러스터는 최소한 하나의 노드로부터 기여를 받아야 함
    # 절댓값 제약을 위한 보조 변수들
    abs_split = {}
    for i in range(n_nodes):
        for j in range(n_items):
            for k in range(num_clusters):
                # 절댓값을 나타내는 보조 변수
                abs_split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, 
                                                name=f"abs_split_{i}_{j}_{k}")
                # 절댓값 제약: abs_split >= split, abs_split >= -split
                model.addConstr(abs_split[i, j, k] >= split[i, j, k])
                model.addConstr(abs_split[i, j, k] >= -split[i, j, k])
    
    for k in range(num_clusters):
        # 클러스터 k에 할당된 총 절댓값이 1 이상이어야 함
        total_contribution = gp.quicksum(
            abs_split[i, j, k] 
            for i in range(n_nodes) 
            for j in range(n_items)
        )
        model.addConstr(total_contribution >= 1, name=f"cluster_nonempty_{k}")
    
    # 목적함수: 분할의 복잡성을 최소화 (사용된 분할 변수의 절댓값 합 최소화)
    total_splits = gp.quicksum(
        abs_split[i, j, k] 
        for i in range(n_nodes) 
        for j in range(n_items) 
        for k in range(num_clusters)
    )
    
    model.setObjective(total_splits, GRB.MINIMIZE)
    
    # 모델 최적화
    print("\n분할 가능한 균형 클러스터링 최적화 시작...")
    model.optimize()
    
    # 결과 처리
    if model.status == GRB.OPTIMAL:
        print(f"\n최적해 발견 | 목적함수 값: {model.objVal:.6f}")
        
        # 결과 추출
        allocation = {}
        cluster_contributions = [np.zeros(n_items) for _ in range(num_clusters)]
        
        for i in range(n_nodes):
            allocation[i] = {}
            for j in range(n_items):
                allocation[i][j] = {}
                for k in range(num_clusters):
                    val = split[i, j, k].x
                    if abs(val) > 1e-6:  # 0이 아닌 값만 저장
                        allocation[i][j][k] = int(round(val))
                        cluster_contributions[k][j] += allocation[i][j][k]
        
        # 결과 출력
        print("\n클러스터별 결과:")
        for k in range(num_clusters):
            print(f"\n클러스터 {k+1}:")
            print(f"  합계: [{cluster_contributions[k][0]:6.1f}, {cluster_contributions[k][1]:6.1f}, {cluster_contributions[k][2]:6.1f}]")
            
            # 이 클러스터에 기여하는 노드들 표시
            contributing_nodes = []
            for i in range(n_nodes):
                node_contribution = [0, 0, 0]
                has_contribution = False
                for j in range(n_items):
                    if j in allocation[i] and k in allocation[i][j]:
                        node_contribution[j] = allocation[i][j][k]
                        has_contribution = True
                
                if has_contribution:
                    contributing_nodes.append((i, node_contribution))
            
            print(f"  기여 노드 수: {len(contributing_nodes)}")
            for node_id, contrib in contributing_nodes:
                print(f"    노드 {node_id:2d}: [{contrib[0]:4}, {contrib[1]:4}, {contrib[2]:4}]")
        
        return allocation, cluster_contributions, model.objVal
        
    elif model.status == GRB.INFEASIBLE:
        print("\n문제가 실행 불가능합니다.")
        model.computeIIS()
        model.write("infeasible_divisible.ilp")
        return None, None, None
        
    elif model.status == GRB.TIME_LIMIT:
        print("\n시간 제한에 도달했습니다.")
        return None, None, None
        
    else:
        print(f"\n최적화 실패 | 상태: {model.status}")
        return None, None, None

def print_detailed_allocation_results(allocation, cluster_contributions, demand_data):
    """할당 결과를 자세히 출력하는 함수"""
    print("\n" + "="*80)
    print("상세 할당 결과")
    print("="*80)
    
    demand_np = demand_data.numpy()
    n_nodes, n_items = demand_np.shape
    
    # 각 노드의 분할 현황
    print("\n각 노드의 분할 현황:")
    for i in range(n_nodes):
        print(f"\n노드 {i:2d} - 원본: [{demand_np[i, 0]:4}, {demand_np[i, 1]:4}, {demand_np[i, 2]:4}]")
        
        if i in allocation:
            for j in range(n_items):
                if j in allocation[i] and allocation[i][j]:
                    splits = allocation[i][j]
                    total_split = sum(splits.values())
                    print(f"  품목 {j+1} ({demand_np[i, j]:4}): ", end="")
                    
                    split_strs = []
                    for k, amount in splits.items():
                        split_strs.append(f"클러스터{k+1}={amount}")
                    print(" + ".join(split_strs) + f" = {total_split}")
    
    # 클러스터별 검증
    print(f"\n클러스터별 균형 검증:")
    for k in range(len(cluster_contributions)):
        contrib = cluster_contributions[k]
        is_balanced = all(abs(contrib[j]) < 1e-6 for j in range(n_items))
        print(f"클러스터 {k+1}: [{contrib[0]:8.3f}, {contrib[1]:8.3f}, {contrib[2]:8.3f}] - {'✓' if is_balanced else '✗'}")

# 메인 실행
if __name__ == "__main__":
    print("분할 가능한 균형 클러스터링 최적화 시작")
    print(f"입력 데이터 shape: {fixed_net_demand.shape}")
    
    # Gurobi로 최적화 실행
    allocation, cluster_contributions, obj_value = solve_divisible_balanced_clustering_gurobi(fixed_net_demand)
    
    if allocation is not None and cluster_contributions is not None:
        # 상세 결과 출력
        print_detailed_allocation_results(allocation, cluster_contributions, fixed_net_demand)
        
        # 전체 균형 검증
        total_balance = np.sum(cluster_contributions, axis=0)
        print(f"\n전체 균형 검증: [{total_balance[0]:8.3f}, {total_balance[1]:8.3f}, {total_balance[2]:8.3f}]")
        print(f"목적함수 값 (총 분할량): {obj_value:.6f}")
        
        # 분할 통계
        total_nodes_split = 0
        for i in range(len(fixed_net_demand)):
            if i in allocation:
                node_split = False
                for j in range(3):
                    if j in allocation[i] and len(allocation[i][j]) > 1:
                        node_split = True
                        break
                if node_split:
                    total_nodes_split += 1
        
        print(f"분할된 노드 수: {total_nodes_split}")
        
    else:
        print("분할 가능한 균형 클러스터링에 실패했습니다.")