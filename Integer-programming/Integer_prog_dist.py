import torch
import numpy as np
from pulp import *
import random
import time
from itertools import combinations

def solve_multiple_solutions_with_distance_selection(fixed_net_demand, dist_matrix, 
                                                   num_clusters=3, max_solutions=10, 
                                                   timeout_per_solution=60):
    """
    여러 해를 찾고 클러스터 내 평균 거리가 가장 작은 해를 선택
    
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
    
    # 2단계: 각 해의 거리 계산
    print("\n2단계: 각 해의 클러스터 내 평균 거리 계산...")
    solution_distances = []
    
    for idx, solution in enumerate(solutions):
        avg_distance = calculate_average_cluster_distance(solution, dist_np, n_nodes, num_clusters)
        solution_distances.append((idx, avg_distance, solution))
        print(f"  해 {idx + 1}: 평균 거리 = {avg_distance:.2f}")
    
    # 3단계: 최적 해 선택
    solution_distances.sort(key=lambda x: x[1])  # 평균 거리 기준 정렬
    best_idx, best_distance, best_solution = solution_distances[0]
    
    print(f"\n3단계: 최적해 선택")
    print(f"✅ 해 {best_idx + 1}이 최적 (평균 거리: {best_distance:.2f})")
    
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
            # 최소 몇 개의 차이점이 있어야 함
            prob += lpSum(differences) >= min(3, len(differences))


def extract_solution_values(x, n_nodes, n_commodities, num_clusters):
    """해 값 추출"""
    solution = {}
    for (i, k, c, type_), var in x.items():
        val = var.varValue or 0
        if val > 0:
            solution[(i, k, c, type_)] = val
    return solution


def calculate_average_cluster_distance(solution, dist_np, n_nodes, num_clusters):
    """클러스터 내 평균 거리 계산"""
    
    # 각 클러스터에 속한 노드들 추출
    cluster_nodes = {c: set() for c in range(num_clusters)}
    
    for (i, k, c, type_), val in solution.items():
        if val > 0:
            cluster_nodes[c].add(i)
    
    total_distance = 0
    total_pairs = 0
    
    for c in range(num_clusters):
        nodes = list(cluster_nodes[c])
        if len(nodes) <= 1:
            continue
        
        # 클러스터 내 모든 노드 쌍의 거리 합
        cluster_distance = 0
        cluster_pairs = 0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                cluster_distance += dist_np[node1][node2]
                cluster_pairs += 1
        
        if cluster_pairs > 0:
            total_distance += cluster_distance
            total_pairs += cluster_pairs
    
    return total_distance / total_pairs if total_pairs > 0 else 0


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
    """결과에 거리 정보 추가"""
    
    success, results, cluster_assignments = result_tuple
    
    # 각 클러스터에 거리 정보 추가
    for c, info in cluster_assignments.items():
        nodes = [node_idx for node_idx, _ in info['nodes']]
        
        if len(nodes) > 1:
            # 클러스터 내 거리 계산
            cluster_distance = 0
            pair_count = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    cluster_distance += dist_np[nodes[i]][nodes[j]]
                    pair_count += 1
            
            cluster_avg_distance = cluster_distance / pair_count if pair_count > 0 else 0
            info['cluster_avg_distance'] = cluster_avg_distance
            info['cluster_total_distance'] = cluster_distance
        else:
            info['cluster_avg_distance'] = 0
            info['cluster_total_distance'] = 0
    
    # 전체 평균 거리 추가
    cluster_assignments['overall_avg_distance'] = avg_distance
    
    return success, results, cluster_assignments


def print_distance_comparison_results(best_result, all_solutions_info, city_names=None):
    """거리 비교 결과 출력"""
    
    if city_names is None:
        city_names = [f"노드_{i}" for i in range(100)]
    
    print(f"\n=== 거리 기반 최적해 선택 결과 ===")
    
    # 모든 해의 거리 비교
    print(f"\n📊 모든 해의 평균 거리 비교:")
    for solution_num, avg_distance, _ in all_solutions_info:
        marker = "👑" if solution_num == all_solutions_info[0][0] else "  "
        print(f"{marker} 해 {solution_num}: {avg_distance:.2f}")
    
    # 최적해 상세 정보
    success, results, cluster_assignments = best_result
    best_avg_distance = cluster_assignments['overall_avg_distance']
    
    print(f"\n🏆 최적해 상세 정보 (평균 거리: {best_avg_distance:.2f})")
    
    for c, info in cluster_assignments.items():
        if c == 'overall_avg_distance':
            continue
            
        print(f"\n클러스터 {c}:")
        print(f"  ✅ 균형: {info['balance']} (완벽한 균형)")
        print(f"  📍 클러스터 내 평균 거리: {info.get('cluster_avg_distance', 0):.2f}")
        print(f"  🔢 노드 수: {len(info['nodes'])}")
        
        print(f"  🏙️  포함 도시:")
        for node_idx, allocation in info['nodes']:
            city_name = city_names[node_idx] if node_idx < len(city_names) else f"노드_{node_idx}"
            print(f"      {city_name}: {allocation}")


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
    
    print("🎯 거리 최적화 기반 해 선택 시작...")
    
    # 여러 해 탐색 및 최적 거리 해 선택
    best_result, all_solutions = solve_multiple_solutions_with_distance_selection(
        fixed_net_demand, dist_matrix, 
        num_clusters=3, 
        max_solutions=5,  # 5개 해 탐색
        timeout_per_solution=500  # 각 해당 60초 제한
    )
    
    if best_result:
        print_distance_comparison_results(best_result, all_solutions, city_names)
    else:
        print("❌ 해를 찾지 못했습니다.")
