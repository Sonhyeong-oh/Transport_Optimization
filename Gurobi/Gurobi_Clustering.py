import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 파라미터 조정
num_clusters = 4 # 클러스터 개수
max_solutions = 10 # 구할 솔루션의 최대 개수
solution_ratio = 0.1 # 다수 해를 구할 때 최적 솔루션 목적함수 값과의 차이 허용치 (0.1 = 10% 이내로 차이나는 솔루션만 저장)

'''
            *** 거리 메트릭 설명 ***

TPC (Total Pairwise Cost) : 전체 노드 쌍 거리 총합
MFC (Mean Facility Cost) : 클러스터 중심(시설)까지의 평균 거리
APC (Average Pairwise Cost) : 모든 노드쌍 거리의 평균
MPC (Maximum Pairwise Cost) : 가장 먼 두 노드 사이 거리 (지름)
중심점 거리 : 중심점(centroid)에서 모든 노드까지의 평균 거리
지름 (Diameter) : 네트워크에서 가장 먼 두 노드 간 거리
반지름 (Radius) : 중심점에서 가장 먼 노드까지의 거리
실루엣 점수 : 클러스터 품질: 음수 → 군집 경계가 불분명
중심점 노드 : 가장 중앙에 있는 노드 (centroid node)
'''


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

# 노드 간 거리 행렬
dist_matrix = torch.tensor([
    #춘천 원주  강릉  동해  태백 속초  삼척 홍천  횡성  영월 평창  정선  철원 화천  양구 인제  고성  양양
    [0,   84,  160, 200, 222, 108, 209, 38,  61,  150, 119, 148, 78,  30,  44,  82,  116, 114], # 춘천
    [84,  0,   127, 166, 141, 181, 175, 60,  20,  72,  70,  123, 156, 112, 122, 112, 202, 158], # 원주
    [160, 127, 0,   47,  99,  65,  59,  134, 107, 117, 90,  68,  243, 187, 128, 110, 96,  53 ], # 강릉
    [200, 166, 47,  0,   54,  104, 14,  173, 146, 114, 129, 76,  282, 227, 168, 150, 136, 89 ], # 동해
    [222, 141, 99,  54,  0,   164, 47,  189, 159, 64,  84,  52,  287, 249, 216, 198, 185, 142], # 태백
    [108, 181, 65,  104, 164, 0,   118, 105, 138, 171, 144, 138, 221, 118, 67,  50,  25,  17 ], # 속초
    [209, 175, 59,  14,  47,  118, 0,   184, 159, 108, 107, 78,  293, 237, 178, 160, 147, 102], # 삼척
    [38,  60,  134, 173, 189, 105, 184, 0,   34,  124, 87,  116, 123, 65,  64,  54,  109, 90 ], # 홍천
    [61,  20,  107, 146, 159, 138, 159, 34,  0,   95,  55,  85,  145, 87,  97,  83,  178, 119], # 횡성
    [150, 72,  117, 114, 64,  171, 108, 124, 95,  0,   29,  52,  224, 187, 197, 186, 199, 151], # 영월
    [119, 70,  90,  129, 84,  144, 107, 87,  55,  29,  0,   31,  211, 143, 150, 128, 171, 124], # 평창
    [148, 123, 68,  76,  52,  138, 78,  116, 85,  52,  31,  0,   262, 220, 197, 142, 166, 118], # 정선
    [78,  156, 243, 282, 287, 221, 293, 123, 145, 224, 211, 262, 0,   61,  119, 149, 192, 211], # 철원
    [30,  112, 187, 227, 249, 118, 237, 65,  87,  187, 143, 220, 61,  0,   44,  73,  155, 143], # 화천
    [44,  122, 128, 168, 216, 67,  178, 64,  97,  197, 150, 197, 119, 44,  0,   31,  75,  74 ], # 양구
    [82,  112, 110, 150, 198, 50,  160, 54,  83,  186, 128, 142, 149, 73,  31,  0,   57,  55 ], # 인제
    [116, 202, 96,  136, 185, 25,  147, 109, 178, 199, 171, 166, 192, 155, 75,  57,  0,   43 ], # 고성
    [114, 158, 53,  89,  142, 17,  102, 90,  119, 151, 124, 118, 211, 143, 74,  55,  43,  0  ]  # 양양
], dtype=torch.float32)

def solve_divisible_balanced_clustering_gurobi(demand_data, num_clusters=4):
    """
    Gurobi를 사용하여 노드 분할 가능한 균형 클러스터링 문제 해결
    각 노드는 여러 클러스터에 정수 비율로 분할 가능
    
    Parameters:
    - demand_data: 노드별 수요/공급 데이터 (n_nodes x n_items)
    - num_clusters: 클러스터 개수
    
    Returns:
    - all_solutions: 모든 솔루션들의 리스트 [(allocation, cluster_contributions, obj_value), ...]
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

    model.setParam(GRB.Param.PoolSearchMode, 2)  # 2: 모든 해를 적극적으로 탐색
    model.setParam(GRB.Param.PoolSolutions, max_solutions)  # 최대 10개의 해를 저장
    model.setParam(GRB.Param.PoolGap, solution_ratio) # 최적해 대비 10% 차이 나는 솔루션만 저장
    
    model.optimize()
    
    print(f"저장된 솔루션 개수: {model.SolCount}")

    # 각 솔루션 확인 및 저장
    all_solutions = []
    for sol_idx in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, sol_idx)
        print(f"\n솔루션 {sol_idx+1}:")
        print(f"목적함수 값: {model.PoolObjVal}")
        
        # 이 솔루션의 allocation 추출
        allocation = {}
        cluster_contributions = [np.zeros(n_items) for _ in range(num_clusters)]
        
        for i in range(n_nodes):
            allocation[i] = {}
            for j in range(n_items):
                allocation[i][j] = {}
                for k in range(num_clusters):
                    val = split[i, j, k].Xn  # Xn으로 n번째 솔루션 값 가져오기
                    if abs(val) > 1e-6:
                        allocation[i][j][k] = int(round(val))
                        cluster_contributions[k][j] += allocation[i][j][k]
        
        # 이 솔루션 저장
        all_solutions.append((allocation, cluster_contributions, model.PoolObjVal))

    # 결과 처리
    if model.status == GRB.OPTIMAL:
        print(f"\n최적해 발견!")
        return all_solutions
        
    elif model.status == GRB.INFEASIBLE:
        print("\n문제가 실행 불가능합니다.")
        model.computeIIS()
        model.write("infeasible_divisible.ilp")
        return []
        
    elif model.status == GRB.TIME_LIMIT:
        print("\n시간 제한에 도달했습니다.")
        return []
        
    else:
        print(f"\n최적화 실패 | 상태: {model.status}")
        return []

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

def calculate_all_distance_metrics(allocation, dist_matrix, num_clusters=4):
    """
    모든 거리 메트릭을 계산하는 함수
    """
    results = {}
    
    for k in range(num_clusters):
        # 클러스터에 속한 노드들 찾기
        cluster_nodes = []
        for i in range(len(dist_matrix)):
            if i in allocation:
                has_contribution = False
                for j in range(3):
                    if j in allocation[i] and k in allocation[i][j]:
                        if abs(allocation[i][j][k]) > 0:
                            has_contribution = True
                            break
                if has_contribution:
                    cluster_nodes.append(i)
        
        if len(cluster_nodes) < 2:
            results[k] = {
                'nodes': cluster_nodes,
                'TPC': 0, 'MFC': 0, 'APC': 0, 'MPC': 0,
                'centroid_distance': 0, 'diameter': 0,
                'radius': 0, 'silhouette': 0
            }
            continue
        
        # 거리 행렬 추출
        cluster_dist_matrix = []
        for i in cluster_nodes:
            row = []
            for j in cluster_nodes:
                row.append(dist_matrix[i][j].item())
            cluster_dist_matrix.append(row)
        cluster_dist_matrix = np.array(cluster_dist_matrix)
        
        # 1. TPC (Total Pairwise Cost) - 모든 쌍의 거리 합
        tpc = 0
        pairwise_distances = []
        for i in range(len(cluster_nodes)):
            for j in range(i+1, len(cluster_nodes)):
                dist = cluster_dist_matrix[i][j]
                tpc += dist
                pairwise_distances.append(dist)
        
        # 2. MFC (Mean Facility Cost) - 평균 거리
        mfc = np.mean(pairwise_distances) if pairwise_distances else 0
        
        # 3. APC (Average Pairwise Cost) - TPC와 동일하지만 정규화
        apc = tpc / len(pairwise_distances) if pairwise_distances else 0
        
        # 4. MPC (Maximum Pairwise Cost) - 최대 거리 (지름)
        mpc = np.max(pairwise_distances) if pairwise_distances else 0
        
        # 5. Centroid Distance - 중심점으로부터의 평균 거리
        centroid_idx = 0
        min_avg_dist = float('inf')
        for i in range(len(cluster_nodes)):
            avg_dist = np.mean([cluster_dist_matrix[i][j] for j in range(len(cluster_nodes)) if i != j])
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                centroid_idx = i
        
        centroid_distance = min_avg_dist
        
        # 6. Diameter - 클러스터 내 최대 거리
        diameter = mpc
        
        # 7. Radius - 중심점으로부터 최대 거리
        radius = np.max([cluster_dist_matrix[centroid_idx][j] for j in range(len(cluster_nodes))])
        
        # 8. Silhouette Score (간단화된 버전)
        intra_cluster_dist = np.mean(pairwise_distances)
        # 다른 클러스터들과의 최소 거리 (간단화)
        min_inter_cluster_dist = float('inf')
        for other_k in range(num_clusters):
            if other_k == k:
                continue
            other_nodes = []
            for i in range(len(dist_matrix)):
                if i in allocation:
                    has_contribution = False
                    for j in range(3):
                        if j in allocation[i] and other_k in allocation[i][j]:
                            if abs(allocation[i][j][other_k]) > 0:
                                has_contribution = True
                                break
                    if has_contribution:
                        other_nodes.append(i)
            
            if other_nodes:
                inter_dists = []
                for node1 in cluster_nodes:
                    for node2 in other_nodes:
                        inter_dists.append(dist_matrix[node1][node2].item())
                if inter_dists:
                    avg_inter = np.mean(inter_dists)
                    min_inter_cluster_dist = min(min_inter_cluster_dist, avg_inter)
        
        if min_inter_cluster_dist == float('inf'):
            silhouette = 0
        else:
            silhouette = (min_inter_cluster_dist - intra_cluster_dist) / max(min_inter_cluster_dist, intra_cluster_dist)
        
        results[k] = {
            'nodes': cluster_nodes,
            'node_count': len(cluster_nodes),
            'TPC': tpc,
            'MFC': mfc,
            'APC': apc,
            'MPC': mpc,
            'centroid_distance': centroid_distance,
            'diameter': diameter,
            'radius': radius,
            'silhouette': silhouette,
            'centroid_node': cluster_nodes[centroid_idx]
        }
    
    return results

def print_all_distance_metrics(allocation, dist_matrix, num_clusters=4):
    """
    모든 거리 메트릭을 출력
    """
    print("\n" + "="*80)
    print("전체 거리 메트릭 분석")
    print("="*80)
    
    results = calculate_all_distance_metrics(allocation, dist_matrix, num_clusters)
    
    for k, metrics in results.items():
        print(f"\n[클러스터 {k + 1}]")
        print(f"  소속 노드: {metrics['nodes']}")
        print(f"  노드 수: {metrics['node_count']}")
        
        if metrics['node_count'] >= 2:
            print(f"  TPC (Total Pairwise Cost): {metrics['TPC']:.2f}")
            print(f"  MFC (Mean Facility Cost): {metrics['MFC']:.2f}")
            print(f"  APC (Average Pairwise Cost): {metrics['APC']:.2f}")
            print(f"  MPC (Maximum Pairwise Cost): {metrics['MPC']:.2f}")
            print(f"  중심점 거리 (Centroid Distance): {metrics['centroid_distance']:.2f}")
            print(f"  지름 (Diameter): {metrics['diameter']:.2f}")
            print(f"  반지름 (Radius): {metrics['radius']:.2f}")
            print(f"  실루엣 점수 (Silhouette): {metrics['silhouette']:.3f}")
            print(f"  중심점 노드: {metrics['centroid_node']}")
        else:
            print("  단일 노드 또는 빈 클러스터 - 거리 메트릭 계산 불가")
    
    # 전체 요약
    print(f"\n[전체 요약]")
    total_tpc = sum(m['TPC'] for m in results.values())
    avg_mfc = np.mean([m['MFC'] for m in results.values() if m['node_count'] >= 2])
    max_diameter = max([m['diameter'] for m in results.values() if m['node_count'] >= 2], default=0)
    avg_silhouette = np.mean([m['silhouette'] for m in results.values() if m['node_count'] >= 2])
    
    print(f"  전체 TPC: {total_tpc:.2f}")
    print(f"  평균 MFC: {avg_mfc:.2f}")
    print(f"  최대 지름: {max_diameter:.2f}")
    print(f"  평균 실루엣: {avg_silhouette:.3f}")

# 메인 실행
if __name__ == "__main__":
    print("분할 가능한 균형 클러스터링 최적화 시작")
    print(f"입력 데이터 shape: {fixed_net_demand.shape}")
    
    # Gurobi로 최적화 실행
    all_solutions = solve_divisible_balanced_clustering_gurobi(fixed_net_demand, num_clusters=num_clusters)
    
    if all_solutions:
        # 균형이 맞는 솔루션들만 필터링
        balanced_solutions = []
        
        for sol_idx, (allocation, cluster_contributions, obj_value) in enumerate(all_solutions):
            # 균형 검증
            total_balance = np.sum(cluster_contributions, axis=0)
            is_balanced = all(abs(total_balance[j]) < 1e-6 for j in range(3))
            
            if is_balanced:
                # 거리 메트릭 계산
                distance_results = calculate_all_distance_metrics(allocation, dist_matrix, num_clusters)
                total_tpc = sum(m['TPC'] for m in distance_results.values())
                
                balanced_solutions.append({
                    'solution_idx': sol_idx,
                    'allocation': allocation,
                    'cluster_contributions': cluster_contributions,
                    'obj_value': obj_value,
                    'total_tpc': total_tpc,
                    'distance_results': distance_results
                })
                
                print(f"솔루션 {sol_idx + 1}: 균형 ✓, 총 TPC = {total_tpc:.2f}")
        
        if balanced_solutions:
            # 최소 거리 솔루션 찾기
            best_solution = min(balanced_solutions, key=lambda x: x['total_tpc'])
            
            print(f"\n{'='*100}")
            print(f"최적 솔루션: {best_solution['solution_idx'] + 1}번 (총 TPC: {best_solution['total_tpc']:.2f})")
            print(f"{'='*100}")
            
            # 최적 솔루션 상세 분석
            allocation = best_solution['allocation']
            cluster_contributions = best_solution['cluster_contributions']
            obj_value = best_solution['obj_value']
            
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
            
            # 클러스터별 소속 노드와 분할된 수요/공급량 출력
            print(f"\n{'='*80}")
            print("클러스터별 소속 노드 및 분할된 수요/공급량")
            print(f"{'='*80}")
            
            for k in range(num_clusters):
                print(f"\n[클러스터 {k + 1}]")
                cluster_nodes_detailed = {}
                
                # 각 노드의 이 클러스터에 대한 기여 수집
                for i in range(len(fixed_net_demand)):
                    if i in allocation:
                        node_contribution = [0, 0, 0]
                        has_contribution = False
                        for j in range(3):
                            if j in allocation[i] and k in allocation[i][j]:
                                node_contribution[j] = allocation[i][j][k]
                                if abs(node_contribution[j]) > 0:
                                    has_contribution = True
                        
                        if has_contribution:
                            cluster_nodes_detailed[i] = node_contribution
                
                if cluster_nodes_detailed:
                    print(f"  소속 노드 수: {len(cluster_nodes_detailed)}")
                    print(f"  노드별 분할된 수요/공급량:")
                    
                    cluster_total = [0, 0, 0]
                    for node_id, contrib in sorted(cluster_nodes_detailed.items()):
                        print(f"    노드 {node_id:2d}: [{contrib[0]:6}, {contrib[1]:6}, {contrib[2]:6}]")
                        for j in range(3):
                            cluster_total[j] += contrib[j]
                    
                    print(f"  클러스터 합계: [{cluster_total[0]:6}, {cluster_total[1]:6}, {cluster_total[2]:6}]")
                else:
                    print(f"  빈 클러스터")

            # 거리 메트릭 분석
            print_all_distance_metrics(allocation, dist_matrix, num_clusters=num_clusters)
            
        else:
            print("균형이 맞는 솔루션이 없습니다.")
            
    else:
        print("분할 가능한 균형 클러스터링에 실패했습니다.")
