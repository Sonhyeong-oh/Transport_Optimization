import torch
import numpy as np
from pulp import *
import random
import time
from itertools import combinations

def solve_multiple_solutions_with_distance_selection(fixed_net_demand, dist_matrix, 
                                                   num_clusters=4, max_solutions=10, 
                                                   timeout_per_solution=60):
    """
    여러 해를 찾고 클러스터별 총 거리의 평균이 가장 작은 해를 선택
    
    Parameters:
    - fixed_net_demand: 노드별 공급-수요 데이터
    - dist_matrix: 거리 행렬
    - num_clusters: 클러스터 수
    - max_solutions: 탐색할 최대 해의 수
    - timeout_per_solution: 각 해당 찾기 시간 제한
    
    Returns:
    - 최적 거리를 가진 해와 모든 해의 정보
    """
    
    print("=== 다중해 탐색 및 거리 기반 최적해 선택 ===")
    start_time = time.time()
    
    demand_np = fixed_net_demand.numpy()
    dist_np = dist_matrix.numpy()
    n_nodes, n_commodities = demand_np.shape
    
    # 1단계: 여러 해 탐색
    print(f"1단계: 최대 {max_solutions}개 해 탐색...")
    solutions = find_diverse_solutions(demand_np, n_nodes, n_commodities, num_clusters, 
                                     max_solutions, timeout_per_solution)
    
    if not solutions:
        print("❌ 실행 가능한 해를 찾지 못했습니다.")
        return None, None
    
    print(f"✅ {len(solutions)}개의 서로 다른 해 발견!")
    
    # 2단계: 각 해의 거리 계산 (클러스터 별 총 거리 계산 후 num_clusters로 나누어 평균 계산)
    print("\n2단계: 각 해의 클러스터별 총 거리의 평균 계산...")
    solution_distances = []
    
    for idx, solution in enumerate(solutions):
        avg_distance = calculate_average_cluster_distance(solution, dist_np, n_nodes, num_clusters)
        solution_distances.append((idx, avg_distance, solution))
        print(f"  해 {idx + 1}: 클러스터 총 거리 평균 = {avg_distance:.2f}")
    
    # 3단계: 최적 해 선택
    solution_distances.sort(key=lambda x: x[1])  # 평균 거리 기준 정렬
    best_idx, best_distance, best_solution = solution_distances[0]
    
    print(f"\n3단계: 최적해 선택")
    print(f"✅ 해 {best_idx + 1}이 최적 (클러스터 총 거리 평균: {best_distance:.2f})")
    
    # 4단계: 결과 변환
    best_result = convert_solution_to_result_format(best_solution, demand_np, n_nodes, n_commodities, num_clusters)
    
    # 5단계: 거리 정보 추가
    best_result_with_distance = add_distance_info_to_result(best_result, dist_np, best_distance)
    
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    # 모든 해의 정보도 반환
    all_solutions_info = []
    for idx, distance, solution in solution_distances:
        result = convert_solution_to_result_format(solution, demand_np, n_nodes, n_commodities, num_clusters)
        result_with_distance = add_distance_info_to_result(result, dist_np, distance)
        all_solutions_info.append((idx + 1, distance, result_with_distance))
    
    return best_result_with_distance, all_solutions_info


def find_diverse_solutions(demand_np, n_nodes, n_commodities, num_clusters, max_solutions, timeout_per_solution):
    """다양한 해 탐색"""
    
    solutions = []
    
    for solution_idx in range(max_solutions):
        print(f"  해 {solution_idx + 1} 탐색중... ", end="")
        
        # 새로운 문제 생성
        prob = LpProblem(f"Solution_{solution_idx}", LpMinimize)
        
        # 결정변수 생성
        x = create_decision_variables(demand_np, n_nodes, n_commodities, num_clusters, solution_idx)
        cluster_used = {c: LpVariable(f"cluster_used_{c}_{solution_idx}", cat='Binary') 
                       for c in range(num_clusters)}
        
        # 목적함수 (다양성을 위한 랜덤 가중치)
        objective = -lpSum([cluster_used[c] for c in range(num_clusters)])
        if solution_idx > 0:
            # 이전 해들과 다른 해를 찾기 위한 랜덤 가중치
            random.seed(solution_idx * 42)  # 재현 가능한 랜덤
            for (i, k, c, type_), var in x.items():
                objective += random.uniform(0.0001, 0.001) * var
        
        prob += objective
        
        # 제약조건 추가
        add_balance_constraints(prob, x, cluster_used, demand_np, n_nodes, n_commodities, num_clusters)
        
        # 이전 해들과 다른 해를 강제하는 제약조건
        if solutions:
            add_diversity_constraints_improved(prob, x, solutions, n_nodes, n_commodities, num_clusters)
        
        # 해결 (시간 제한)
        solver = PULP_CBC_CMD(msg=0, timeLimit=timeout_per_solution)
        prob.solve(solver)
        
        if prob.status == LpStatusOptimal:
            solution = extract_solution_values(x, n_nodes, n_commodities, num_clusters)
            solutions.append(solution)
            print("✅")
        else:
            print("❌")
            break  # 더 이상 해를 찾을 수 없음
    
    return solutions


def create_decision_variables(demand_np, n_nodes, n_commodities, num_clusters, solution_idx):
    """결정변수 생성"""
    x = {}
    for i in range(n_nodes):
        for k in range(n_commodities):
            for c in range(num_clusters):
                if demand_np[i, k] >= 0:  # 공급
                    x[(i, k, c, 'supply')] = LpVariable(f"x_supply_{i}_{k}_{c}_{solution_idx}", 
                                                       lowBound=0, upBound=demand_np[i, k], 
                                                       cat='Integer')
                else:  # 수요
                    x[(i, k, c, 'demand')] = LpVariable(f"x_demand_{i}_{k}_{c}_{solution_idx}", 
                                                       lowBound=0, upBound=-demand_np[i, k], 
                                                       cat='Integer')
    return x


def add_balance_constraints(prob, x, cluster_used, demand_np, n_nodes, n_commodities, num_clusters):
    """균형 제약조건 추가"""
    
    # 제약조건 1: 각 노드의 공급/수요량이 정확히 분할되어야 함
    # 분할된 값들의 합이 원래 값과 동일해야 함
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


def add_diversity_constraints_improved(prob, x, previous_solutions, n_nodes, n_commodities, num_clusters):
    """개선된 다양성 제약조건"""
    
    for prev_solution in previous_solutions[-3:]:  # 최근 3개 해와만 비교 (성능상 이유)
        # 할당 패턴이 다르도록 강제
        differences = []
        
        # 주요 할당에 대해서만 차이 강제
        for (i, k, c, type_), prev_val in prev_solution.items():
            if prev_val > 0 and (i, k, c, type_) in x:
                # 이전 해에서 양수 할당된 것과 다르게 할당
                diff_var = LpVariable(f"diff_{i}_{k}_{c}_{type_}_{len(previous_solutions)}", cat='Binary')
                prob += x[(i, k, c, type_)] <= prev_val - 1 + 1000 * diff_var
                prob += x[(i, k, c, type_)] >= prev_val + 1 - 1000 * (1 - diff_var)
                differences.append(diff_var)
        
        if differences:
            # 최소 min(1, len(difference)) 개의 차이점이 있어야 다른 해로 인식
            prob += lpSum(differences) >= min(1, len(differences))


def extract_solution_values(x, n_nodes, n_commodities, num_clusters):
    """해 값 추출"""
    solution = {}
    for (i, k, c, type_), var in x.items():
        val = var.varValue or 0
        if val > 0:
            solution[(i, k, c, type_)] = val
    return solution


def calculate_average_cluster_distance(solution, dist_np, n_nodes, num_clusters):
    """
    각 클러스터 내 노드 간 총 거리를 구하고, 
    그 총 거리들의 평균을 계산 (클러스터 개수로 나눔)
    """
    
    # 각 클러스터에 속한 노드들 추출
    cluster_nodes = {c: set() for c in range(num_clusters)}
    
    for (i, k, c, type_), val in solution.items():
        if val > 0:
            cluster_nodes[c].add(i)
    
    # 각 클러스터의 총 거리를 저장
    cluster_total_distances = []
    
    for c in range(num_clusters):
        nodes = list(cluster_nodes[c])
        if len(nodes) <= 1:
            # 노드가 1개 이하인 클러스터는 거리를 0으로 처리
            cluster_total_distances.append(0)
            continue
        
        # 클러스터 내 모든 노드 쌍의 거리 합
        # ex) dist(A -> B) + dist(B -> C) + dist(C -> D) + ...
        cluster_distance = 0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                cluster_distance += dist_np[node1][node2]
        
        cluster_total_distances.append(cluster_distance)
    
    # 클러스터들의 총 거리를 합산하고 클러스터 개수로 나눔
    average_of_totals = sum(cluster_total_distances) / num_clusters
    
    return average_of_totals


def calculate_cluster_distances_detailed(solution, dist_np, n_nodes, num_clusters):
    """
    각 클러스터별 상세 거리 정보 계산
    """
    
    # 각 클러스터에 속한 노드들 추출
    cluster_nodes = {c: set() for c in range(num_clusters)}
    
    for (i, k, c, type_), val in solution.items():
        if val > 0:
            cluster_nodes[c].add(i)
    
    cluster_info = {}
    total_sum = 0
    
    for c in range(num_clusters):
        nodes = list(cluster_nodes[c])
        cluster_info[c] = {
            'nodes': nodes,
            'node_count': len(nodes),
            'total_distance': 0,
            'pair_count': 0,
            'average_distance': 0
        }
        
        if len(nodes) <= 1:
            continue
        
        # 클러스터 내 모든 노드 쌍의 거리 계산
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                distance = dist_np[node1][node2]
                cluster_info[c]['total_distance'] += distance
                cluster_info[c]['pair_count'] += 1
        
        # 클러스터 내 평균 거리
        if cluster_info[c]['pair_count'] > 0:
            cluster_info[c]['average_distance'] = (
                cluster_info[c]['total_distance'] / cluster_info[c]['pair_count']
            )
        
        total_sum += cluster_info[c]['total_distance']
    
    # 원하시는 방식의 평균 (각 클러스터 총 거리의 합 / 클러스터 개수)
    average_of_totals = total_sum / num_clusters
    
    return cluster_info, average_of_totals


def convert_solution_to_result_format(solution, demand_np, n_nodes, n_commodities, num_clusters):
    """해를 결과 형태로 변환"""
    
    results = []
    cluster_assignments = {c: {'nodes': [], 'balance': np.zeros(n_commodities)} 
                          for c in range(num_clusters)}
    
    # 노드별 할당 추출
    for i in range(n_nodes):
        node_allocation = {c: np.zeros(n_commodities) for c in range(num_clusters)}
        
        for k in range(n_commodities):
            for c in range(num_clusters):
                # 공급 할당
                supply_key = (i, k, c, 'supply')
                if supply_key in solution:
                    val = solution[supply_key]
                    node_allocation[c][k] += val
                    cluster_assignments[c]['balance'][k] += val
                
                # 수요 할당
                demand_key = (i, k, c, 'demand')
                if demand_key in solution:
                    val = solution[demand_key]
                    node_allocation[c][k] -= val
                    cluster_assignments[c]['balance'][k] -= val
        
        # 할당이 있는 클러스터만 기록
        for c in range(num_clusters):
            if np.any(np.abs(node_allocation[c]) > 1e-10):
                cluster_assignments[c]['nodes'].append((i, node_allocation[c]))
                results.append({
                    'node': i,
                    'cluster': c,
                    'original_demand': demand_np[i],
                    'allocated_demand': node_allocation[c]
                })
    
    return True, results, cluster_assignments


def add_distance_info_to_result(result_tuple, dist_np, avg_distance):
    """결과에 거리 정보 추가 """
    
    success, results, cluster_assignments = result_tuple
    
    # 상세 거리 정보 계산
    solution = {}
    for result in results:
        node = result['node']
        cluster = result['cluster']
        allocated = result['allocated_demand']
        for k in range(len(allocated)):
            if allocated[k] > 0:
                solution[(node, k, cluster, 'supply')] = allocated[k]
            elif allocated[k] < 0:
                solution[(node, k, cluster, 'demand')] = -allocated[k]
    
    cluster_info, _ = calculate_cluster_distances_detailed(solution, dist_np, len(dist_np), len(cluster_assignments))
    
    # 각 클러스터에 거리 정보 추가
    for c, info in cluster_assignments.items():
        if isinstance(c, int) and c in cluster_info:
            info['cluster_total_distance'] = cluster_info[c]['total_distance']
            info['cluster_avg_distance'] = cluster_info[c]['average_distance']
            info['pair_count'] = cluster_info[c]['pair_count']
    
    # 전체 평균 거리 추가
    cluster_assignments['overall_avg_distance'] = avg_distance
    
    return success, results, cluster_assignments


def print_distance_comparison_results(best_result, all_solutions_info, num_clusters, city_names=None):
    """거리 비교 결과 출력 """
    
    if city_names is None:
        city_names = [f"노드_{i}" for i in range(100)]
    
    print(f"\n=== 거리 기반 최적해 선택 결과 ===")
    
    # 모든 해의 거리 비교
    print(f"\n📊 모든 해의 클러스터 총 거리 평균 비교:")
    for solution_num, avg_distance, _ in all_solutions_info:
        marker = "👑" if solution_num == all_solutions_info[0][0] else "  "
        print(f"{marker} 해 {solution_num}: {avg_distance:.2f}")
    
    # 최적해 상세 정보
    success, results, cluster_assignments = best_result
    best_avg_distance = cluster_assignments['overall_avg_distance']
    
    print(f"\n🏆 최적해 상세 정보 (클러스터 총 거리 평균: {best_avg_distance:.2f})")
    
    # 각 클러스터의 총 거리 정보
    cluster_distances = []
    
    for c in range(num_clusters):  # 클러스터 개수만큼 출력
        if c in cluster_assignments:
            info = cluster_assignments[c]
            total_dist = info.get('cluster_total_distance', 0)
            cluster_distances.append(total_dist)
            
            print(f"\n클러스터 {c}:")
            print(f"  ✅ 균형: {info['balance']} (완벽한 균형)")
            print(f"  📍 클러스터 내 총 거리: {total_dist:.2f}")
            print(f"  📍 클러스터 내 평균 거리: {info.get('cluster_avg_distance', 0):.2f}")
            print(f"  🔢 노드 수: {len(info['nodes'])}")
            print(f"  🔢 노드 쌍 수: {info.get('pair_count', 0)}")
            
            print(f"  🏙️  포함 도시:")
            for node_idx, allocation in info['nodes']:
                city_name = city_names[node_idx] if node_idx < len(city_names) else f"노드_{node_idx}"
                print(f"      {city_name}: {allocation}")
    
    print(f"\n📊 전체 요약:")
    print(f"  각 클러스터 총 거리: {cluster_distances}")
    print(f"  총 거리의 합: {sum(cluster_distances):.2f}")
    print(f"  클러스터 개수로 나눈 평균: {best_avg_distance:.2f}")