import torch
import numpy as np
from pulp import *
import pandas as pd
import time

def solve_hybrid_cluster_balancing(fixed_net_demand, dist_matrix, num_clusters=4, 
                                 use_fast_stage1=False, stage1_timeout=300):
    """
    하이브리드 접근법: 1단계 균형 해결 + 2단계 거리 최적화
    **반드시 완벽한 균형(합=0) 달성**
    
    Parameters:
    - fixed_net_demand: 노드별 공급-수요 데이터
    - dist_matrix: 거리 행렬
    - num_clusters: 클러스터 수
    - use_fast_stage1: True면 빠른 휴리스틱, False면 정수계획법 사용
    - stage1_timeout: 1단계 최적화 시간 제한 (초)
    
    Returns:
    - 최적화 결과 (완벽한 균형 보장)
    """
    
    print("=== 하이브리드 클러스터링 시작 (완벽한 균형 필수) ===")
    total_start_time = time.time()
    
    if use_fast_stage1:
        print("1단계: 빠른 휴리스틱으로 완벽한 균형 해결")
        stage1_result = solve_stage1_heuristic(fixed_net_demand, num_clusters)
        
        if not stage1_result[0]:  # 휴리스틱 실패 시 정수계획법으로 전환
            print("휴리스틱 실패 - 정수계획법으로 전환")
            stage1_result = solve_stage1_integer_programming(fixed_net_demand, num_clusters, stage1_timeout)
    else:
        print("1단계: 정수계획법으로 완벽한 균형 해결")
        stage1_result = solve_stage1_integer_programming(fixed_net_demand, num_clusters, stage1_timeout)
    
    if not stage1_result[0]:
        print("❌ 완벽한 균형 달성 실패")
        return False, None, None, 0
    
    # 균형 검증
    success, results, cluster_assignments = stage1_result
    print("✓ 1단계 균형 검증:")
    total_imbalance = 0
    for c, info in cluster_assignments.items():
        cluster_imbalance = np.sum(np.abs(info['balance']))
        total_imbalance += cluster_imbalance
        balance_status = "✓" if cluster_imbalance < 1e-8 else "✗"
        print(f"  클러스터 {c}: {info['balance']} {balance_status}")
    
    if total_imbalance >= 1e-8:
        print(f"❌ 균형 미달성 (총 불균형: {total_imbalance:.10f})")
        return False, None, None, 0
    
    print("✅ 완벽한 균형 달성 확인!")
    
    print("2단계: 거리 최적화")
    final_result = solve_stage2_distance_optimization(
        stage1_result, fixed_net_demand, dist_matrix, num_clusters
    )
    
    total_elapsed = time.time() - total_start_time
    print(f"총 실행 시간: {total_elapsed:.2f}초")
    
    return final_result + (total_elapsed,)


def solve_stage1_integer_programming(fixed_net_demand, num_clusters, timeout=300):
    """
    1단계: 기존 정수계획법으로 균형 해결 (시간 제한 있음)
    """
    start_time = time.time()
    
    # 데이터 준비
    demand_np = fixed_net_demand.numpy()
    n_nodes, n_commodities = demand_np.shape
    
    print(f"  노드 수: {n_nodes}, 품목 수: {n_commodities}, 클러스터 수: {num_clusters}")
    
    # 문제 생성
    prob = LpProblem("Stage1_Balance", LpMinimize)
    
    # 결정변수
    x = {}
    for i in range(n_nodes):
        for k in range(n_commodities):
            for c in range(num_clusters):
                if demand_np[i, k] >= 0:  # 공급
                    x[(i, k, c, 'supply')] = LpVariable(f"x_supply_{i}_{k}_{c}", 
                                                       lowBound=0, upBound=demand_np[i, k], 
                                                       cat='Integer')
                else:  # 수요
                    x[(i, k, c, 'demand')] = LpVariable(f"x_demand_{i}_{k}_{c}", 
                                                       lowBound=0, upBound=-demand_np[i, k], 
                                                       cat='Integer')
    
    # 클러스터 사용 여부
    cluster_used = {}
    for c in range(num_clusters):
        cluster_used[c] = LpVariable(f"cluster_used_{c}", cat='Binary')
    
    # 목적함수: 모든 클러스터 사용 강제
    prob += -lpSum([cluster_used[c] for c in range(num_clusters)])
    
    # 제약조건 추가
    add_balance_constraints_stage1(prob, x, cluster_used, demand_np, n_nodes, n_commodities, num_clusters)
    
    # 시간 제한으로 해결
    print(f"  최적화 시작 (최대 {timeout}초)...")
    solver = PULP_CBC_CMD(msg=1, timeLimit=timeout)
    prob.solve(solver)
    
    elapsed = time.time() - start_time
    print(f"  1단계 완료 시간: {elapsed:.2f}초")
    
    if prob.status == LpStatusOptimal:
        print("  1단계 성공: 최적해 발견")
        return extract_stage1_results(x, demand_np, n_nodes, n_commodities, num_clusters)
    elif prob.status == LpStatusNotSolved:
        print("  1단계 시간 초과: 현재까지의 해 사용")
        return extract_stage1_results(x, demand_np, n_nodes, n_commodities, num_clusters)
    else:
        print(f"  1단계 실패: {LpStatus[prob.status]}")
        return False, None, None


def solve_stage1_heuristic(fixed_net_demand, num_clusters):
    """
    1단계: 빠른 휴리스틱으로 **완벽한** 균형 해결 (노드 분할 허용)
    """
    start_time = time.time()
    
    demand_np = fixed_net_demand.numpy()
    n_nodes, n_commodities = demand_np.shape
    
    print(f"  빠른 휴리스틱 사용 (완벽한 균형 보장)...")
    
    # 완벽한 균형을 위한 휴리스틱 접근법
    # 1. 각 노드를 여러 클러스터에 분할할 수 있음
    # 2. 목표: 각 클러스터의 수요-공급 합이 정확히 0
    
    # 노드별 할당 결과 저장
    node_allocations = {}  # {node: {cluster: allocation_vector}}
    cluster_balances = {c: np.zeros(n_commodities) for c in range(num_clusters)}
    
    # 초기화: 모든 노드를 모든 클러스터에 0으로 할당
    for i in range(n_nodes):
        node_allocations[i] = {c: np.zeros(n_commodities) for c in range(num_clusters)}
    
    # 그리디 알고리즘: 각 노드를 순차적으로 처리
    for node_idx in range(n_nodes):
        node_demand = demand_np[node_idx].copy()
        
        # 이 노드의 수요/공급을 클러스터들에 분배
        remaining_demand = node_demand.copy()
        
        # 각 품목별로 처리
        for commodity in range(n_commodities):
            if abs(remaining_demand[commodity]) < 1e-10:
                continue
                
            # 이 품목에 대해 가장 불균형한 클러스터들 찾기
            cluster_needs = []
            for c in range(num_clusters):
                current_balance = cluster_balances[c][commodity]
                # 수요가 있으면 공급이 필요한 클러스터 우선
                # 공급이 있으면 수요가 필요한 클러스터 우선
                if remaining_demand[commodity] > 0:  # 공급
                    need_score = -current_balance  # 음수(수요 과다)일수록 우선
                else:  # 수요
                    need_score = current_balance   # 양수(공급 과다)일수록 우선
                cluster_needs.append((c, need_score))
            
            # 필요도 순으로 정렬 (높은 순)
            cluster_needs.sort(key=lambda x: x[1], reverse=True)
            
            # 남은 수요/공급을 클러스터들에 분배
            remaining_amount = remaining_demand[commodity]
            
            for c, need_score in cluster_needs:
                if abs(remaining_amount) < 1e-10:
                    break
                
                # 이 클러스터에 할당할 양 결정
                if abs(remaining_amount) <= abs(cluster_balances[c][commodity]):
                    # 완전히 할당 가능
                    allocation = remaining_amount
                    remaining_amount = 0
                else:
                    # 부분 할당
                    if cluster_balances[c][commodity] * remaining_amount < 0:
                        # 반대 부호 (상쇄 가능)
                        allocation = -cluster_balances[c][commodity]
                        remaining_amount -= allocation
                    else:
                        # 같은 부호이거나 0 (균등 분배)
                        num_remaining_clusters = len([x for x in cluster_needs if abs(cluster_balances[x[0]][commodity]) < 1e-10])
                        if num_remaining_clusters > 0:
                            allocation = remaining_amount / num_remaining_clusters
                            remaining_amount -= allocation
                        else:
                            allocation = 0
                
                # 할당 실행
                if abs(allocation) > 1e-10:
                    node_allocations[node_idx][c][commodity] = allocation
                    cluster_balances[c][commodity] += allocation
    
    # 미세 조정: 완벽한 균형을 위한 후처리
    max_adjustment_iterations = 50
    for iteration in range(max_adjustment_iterations):
        max_imbalance = 0
        worst_cluster = -1
        worst_commodity = -1
        
        # 가장 불균형한 클러스터와 품목 찾기
        for c in range(num_clusters):
            for k in range(n_commodities):
                imbalance = abs(cluster_balances[c][k])
                if imbalance > max_imbalance:
                    max_imbalance = imbalance
                    worst_cluster = c
                    worst_commodity = k
        
        if max_imbalance < 1e-10:  # 충분히 균형잡힘
            break
        
        # 불균형 해결: 다른 클러스터에서 조정
        needed_amount = -cluster_balances[worst_cluster][worst_commodity]
        
        # 보상해줄 수 있는 클러스터 찾기
        for source_cluster in range(num_clusters):
            if source_cluster == worst_cluster:
                continue
            
            # 이 클러스터에서 조정 가능한 노드 찾기
            for node_idx in range(n_nodes):
                current_allocation = node_allocations[node_idx][source_cluster][worst_commodity]
                
                if abs(current_allocation) > 1e-10 and current_allocation * needed_amount > 0:
                    # 조정 가능한 양 계산
                    adjustment = min(abs(needed_amount), abs(current_allocation))
                    if needed_amount > 0:
                        transfer_amount = adjustment
                    else:
                        transfer_amount = -adjustment
                    
                    # 조정 실행
                    node_allocations[node_idx][source_cluster][worst_commodity] -= transfer_amount
                    node_allocations[node_idx][worst_cluster][worst_commodity] += transfer_amount
                    cluster_balances[source_cluster][worst_commodity] -= transfer_amount
                    cluster_balances[worst_cluster][worst_commodity] += transfer_amount
                    
                    needed_amount -= transfer_amount
                    
                    if abs(needed_amount) < 1e-10:
                        break
            
            if abs(needed_amount) < 1e-10:
                break
    
    elapsed = time.time() - start_time
    print(f"  1단계 완료 시간: {elapsed:.2f}초")
    
    # 결과 변환
    results = []
    cluster_info = {c: {'nodes': [], 'balance': cluster_balances[c]} for c in range(num_clusters)}
    
    for node_idx in range(n_nodes):
        for c in range(num_clusters):
            allocation = node_allocations[node_idx][c]
            if np.any(np.abs(allocation) > 1e-10):  # 0이 아닌 할당만
                cluster_info[c]['nodes'].append((node_idx, allocation))
                results.append({
                    'node': node_idx,
                    'cluster': c,
                    'original_demand': demand_np[node_idx],
                    'allocated_demand': allocation
                })
    
    # 균형 확인
    total_imbalance = sum(np.sum(np.abs(cluster_balances[c])) for c in range(num_clusters))
    print(f"  총 불균형: {total_imbalance:.10f}")
    
    # 완벽한 균형 달성 확인
    perfect_balance = total_imbalance < 1e-8
    if perfect_balance:
        print("  ✓ 완벽한 균형 달성!")
    else:
        print("  ✗ 완벽한 균형 미달성 - 정수계획법 필요")
        return False, None, None
    
    return True, results, cluster_info


def add_balance_constraints_stage1(prob, x, cluster_used, demand_np, n_nodes, n_commodities, num_clusters):
    """1단계용 제약조건 추가"""
    
    # 제약조건 1: 각 노드의 공급/수요량이 정확히 분할되어야 함
    for i in range(n_nodes):
        for k in range(n_commodities):
            if demand_np[i, k] >= 0:  # 공급
                prob += lpSum([x[(i, k, c, 'supply')] for c in range(num_clusters)]) == demand_np[i, k]
            else:  # 수요
                prob += lpSum([x[(i, k, c, 'demand')] for c in range(num_clusters)]) == -demand_np[i, k]
    
    # 제약조건 2: 각 클러스터의 각 품목에 대한 공급-수요 균형
    for c in range(num_clusters):
        for k in range(n_commodities):
            supply_sum = lpSum([x.get((i, k, c, 'supply'), 0) for i in range(n_nodes)])
            demand_sum = lpSum([x.get((i, k, c, 'demand'), 0) for i in range(n_nodes)])
            prob += supply_sum == demand_sum
    
    # 제약조건 3: 클러스터 사용 여부 연결
    M = 1000
    for c in range(num_clusters):
        total_allocation = lpSum([x.get((i, k, c, 'supply'), 0) + x.get((i, k, c, 'demand'), 0) 
                                 for i in range(n_nodes) for k in range(n_commodities)])
        prob += total_allocation >= cluster_used[c]
        prob += total_allocation <= M * cluster_used[c]
    
    # 제약조건 4: 모든 클러스터 사용 강제
    prob += lpSum([cluster_used[c] for c in range(num_clusters)]) == num_clusters


def extract_stage1_results(x, demand_np, n_nodes, n_commodities, num_clusters):
    """1단계 결과 추출"""
    
    results = []
    cluster_assignments = {c: {'nodes': [], 'balance': np.zeros(n_commodities)} 
                          for c in range(num_clusters)}
    
    for i in range(n_nodes):
        node_allocation = {c: np.zeros(n_commodities) for c in range(num_clusters)}
        
        for k in range(n_commodities):
            for c in range(num_clusters):
                if demand_np[i, k] >= 0:  # 공급
                    if (i, k, c, 'supply') in x:
                        val = x[(i, k, c, 'supply')].varValue or 0
                        if val > 0:
                            node_allocation[c][k] += val
                            cluster_assignments[c]['balance'][k] += val
                else:  # 수요
                    if (i, k, c, 'demand') in x:
                        val = x[(i, k, c, 'demand')].varValue or 0
                        if val > 0:
                            node_allocation[c][k] -= val
                            cluster_assignments[c]['balance'][k] -= val
        
        for c in range(num_clusters):
            if np.any(node_allocation[c] != 0):
                cluster_assignments[c]['nodes'].append((i, node_allocation[c]))
                results.append({
                    'node': i,
                    'cluster': c,
                    'original_demand': demand_np[i],
                    'allocated_demand': node_allocation[c]
                })
    
    return True, results, cluster_assignments


def solve_stage2_distance_optimization(stage1_result, fixed_net_demand, dist_matrix, num_clusters):
    """
    2단계: 1단계 결과를 바탕으로 거리 최적화
    """
    start_time = time.time()
    
    success, results, cluster_assignments = stage1_result
    if not success:
        return False, None, None
    
    demand_np = fixed_net_demand.numpy()
    dist_np = dist_matrix.numpy()
    
    # 각 클러스터에 속한 노드들 추출
    cluster_nodes = {c: [] for c in range(num_clusters)}
    for result in results:
        node = result['node']
        cluster = result['cluster']
        if node not in cluster_nodes[cluster]:
            cluster_nodes[cluster].append(node)
    
    # 각 클러스터의 중심 노드 선택 (거리 최소화)
    optimized_assignments = {}
    total_distance = 0
    
    for c in range(num_clusters):
        nodes = cluster_nodes[c]
        if len(nodes) <= 1:
            center = nodes[0] if nodes else None
            cluster_distance = 0
        else:
            # 클러스터 내 모든 노드 간 평균 거리가 최소인 노드를 중심으로 선택
            best_center = None
            min_avg_distance = float('inf')
            
            for potential_center in nodes:
                avg_distance = np.mean([dist_np[potential_center][j] for j in nodes if j != potential_center])
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_center = potential_center
            
            center = best_center
            cluster_distance = sum([dist_np[center][j] for j in nodes if j != center])
        
        total_distance += cluster_distance
        optimized_assignments[c] = {
            'center': center,
            'nodes': nodes,
            'distance': cluster_distance
        }
    
    # 기존 클러스터 할당에 거리 정보 추가
    for c in range(num_clusters):
        if c in cluster_assignments and c in optimized_assignments:
            cluster_assignments[c]['center'] = optimized_assignments[c]['center']
            cluster_assignments[c]['distance'] = optimized_assignments[c]['distance']
    
    elapsed = time.time() - start_time
    print(f"  2단계 완료 시간: {elapsed:.2f}초")
    print(f"  총 클러스터 내 거리: {total_distance:.1f}")
    
    return True, results, cluster_assignments


def print_hybrid_results(success, results, cluster_assignments, total_time, city_names=None):
    """하이브리드 결과 출력 (완벽한 균형 강조)"""
    if not success:
        print("해결할 수 없는 문제입니다.")
        return
    
    if city_names is None:
        city_names = [f"노드_{i}" for i in range(100)]  # 충분히 큰 리스트
    
    print(f"\n=== 하이브리드 클러스터링 결과 (총 {total_time:.2f}초) ===")
    
    total_distance = 0
    perfect_balance_achieved = True
    
    for c, info in cluster_assignments.items():
        cluster_imbalance = np.sum(np.abs(info['balance']))
        balance_perfect = cluster_imbalance < 1e-8
        
        if not balance_perfect:
            perfect_balance_achieved = False
        
        print(f"\n클러스터 {c}:")
        print(f"  ✅ 완벽한 균형: {'달성' if balance_perfect else '미달성'}")
        print(f"  균형 상태: {info['balance']} (오차: {cluster_imbalance:.2e})")
        print(f"  노드 수: {len(info['nodes'])}")
        
        if 'center' in info and info['center'] is not None:
            center_name = city_names[info['center']] if info['center'] < len(city_names) else f"노드_{info['center']}"
            print(f"  중심 노드: {center_name}")
        
        if 'distance' in info:
            print(f"  클러스터 내 거리: {info['distance']:.1f}")
            total_distance += info['distance']
        
        print("  포함 노드:")
        for node_idx, allocation in info['nodes']:
            node_name = city_names[node_idx] if node_idx < len(city_names) else f"노드_{node_idx}"
            # 할당량이 원래 수요와 다른 경우 (분할된 경우) 표시
            print(f"    {node_name}: {allocation}")
    
    print(f"\n{'='*50}")
    print(f"🎯 완벽한 균형 달성: {'✅ 성공' if perfect_balance_achieved else '❌ 실패'}")
    print(f"📍 총 클러스터 내 거리 합: {total_distance:.1f}")
    print(f"⏱️  총 실행 시간: {total_time:.2f}초")
    print(f"{'='*50}")


# 실행 예제
if __name__ == "__main__":
    # 데이터 설정
    fixed_net_demand = torch.tensor([
        [-5, -4,  1], [-5, -5,  5], [-1,  5,  3], [-5,  5, -5],
        [ 5, -5, -5], [ 1, -1,  2], [ 1, -2, -3], [-5, -3, -5],
        [-1, -2,  2], [ 3,  5, -1], [-2, -1, -5], [ 1, -3,  0],
        [ 2,  5,  3], [ 3, -5,  3], [-1,  5,  2], [ 4,  5,  1],
        [ 0,  5,  2], [ 5, -4,  0]
    ], dtype=torch.float32)
    
    dist_matrix = torch.tensor([
        [0,   84,  160, 200, 222, 108, 209, 38,  61,  150, 119, 148, 78,  30,  44,  82,  116, 114],
        [84,  0,   127, 166, 141, 181, 175, 60,  20,  72,  70,  123, 156, 112, 122, 112, 202, 158],
        [160, 127, 0,   47,  99,  65,  59,  134, 107, 117, 90,  68,  243, 187, 128, 110, 96,  53 ],
        [200, 166, 47,  0,   54,  104, 14,  173, 146, 114, 129, 76,  282, 227, 168, 150, 136, 89 ],
        [222, 141, 99,  54,  0,   164, 47,  189, 159, 64,  84,  52,  287, 249, 216, 198, 185, 142],
        [108, 181, 65,  104, 164, 0,   118, 105, 138, 171, 144, 138, 221, 118, 67,  50,  25,  17 ],
        [209, 175, 59,  14,  47,  118, 0,   184, 159, 108, 107, 78,  293, 237, 178, 160, 147, 102],
        [38,  60,  134, 173, 189, 105, 184, 0,   34,  124, 87,  116, 123, 65,  64,  54,  109, 90 ],
        [61,  20,  107, 146, 159, 138, 159, 34,  0,   95,  55,  85,  145, 87,  97,  83,  178, 119],
        [150, 72,  117, 114, 64,  171, 108, 124, 95,  0,   29,  52,  224, 187, 197, 186, 199, 151],
        [119, 70,  90,  129, 84,  144, 107, 87,  55,  29,  0,   31,  211, 143, 150, 128, 171, 124],
        [148, 123, 68,  76,  52,  138, 78,  116, 85,  52,  31,  0,   262, 220, 197, 142, 166, 118],
        [78,  156, 243, 282, 287, 221, 293, 123, 145, 224, 211, 262, 0,   61,  119, 149, 192, 211],
        [30,  112, 187, 227, 249, 118, 237, 65,  87,  187, 143, 220, 61,  0,   44,  73,  155, 143],
        [44,  122, 128, 168, 216, 67,  178, 64,  97,  197, 150, 197, 119, 44,  0,   31,  75,  74 ],
        [82,  112, 110, 150, 198, 50,  160, 54,  83,  186, 128, 142, 149, 73,  31,  0,   57,  55 ],
        [116, 202, 96,  136, 185, 25,  147, 109, 178, 199, 171, 166, 192, 155, 75,  57,  0,   43 ],
        [114, 158, 53,  89,  142, 17,  102, 90,  119, 151, 124, 118, 211, 143, 74,  55,  43,  0  ]
    ], dtype=torch.float32)
    
    city_names = ['춘천', '원주', '강릉', '동해', '태백', '속초', '삼척', '홍천', 
                 '횡성', '영월', '평창', '정선', '철원', '화천', '양구', '인제', '고성', '양양']
    
    print("=== 하이브리드 클러스터링 테스트 ===")
    
    # 방법 1: 정수계획법 + 거리최적화 (시간제한 60초)
    print("\n[방법 1] 정수계획법(60초) + 거리최적화")
    success1, results1, cluster_assignments1, total_time1 = solve_hybrid_cluster_balancing(
        fixed_net_demand, dist_matrix, num_clusters=4, 
        use_fast_stage1=False, stage1_timeout=60
    )
    if success1:
        print_hybrid_results(success1, results1, cluster_assignments1, total_time1, city_names)
    
    # 방법 2: 빠른 휴리스틱 + 거리최적화
    print("\n" + "="*60)
    print("[방법 2] 빠른 휴리스틱 + 거리최적화")
    success2, results2, cluster_assignments2, total_time2 = solve_hybrid_cluster_balancing(
        fixed_net_demand, dist_matrix, num_clusters=4, 
        use_fast_stage1=True
    )
    if success2:
        print_hybrid_results(success2, results2, cluster_assignments2, total_time2, city_names)
    
    print(f"\n=== 실행 시간 비교 ===")
    print(f"정수계획법 방법: {total_time1:.2f}초")
    print(f"휴리스틱 방법: {total_time2:.2f}초")