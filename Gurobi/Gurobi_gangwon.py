import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 군집 내 거리 비슷하게
# 최대 적재량 설정 (한 노드의 공급량, 수요량 제한(수요량 제한은 더 생각해보기))
# 클러스터링 시각화


# 파라미터 조정
num_clusters = 4 # 클러스터 개수
max_solutions = 10 # 구할 솔루션의 최대 개수
solution_ratio = 0.1 # 다수 해를 구할 때 최적 솔루션 목적함수 값과의 차이 허용치 (0.1 = 10% 이내로 차이나는 솔루션만 저장)
limit = 65 # 적재 한도 (수요량, 공급량 제한 / 수요량 분할 범위 : -limit ~ 0 / 공급량 분할 범위 : 0 ~ limit)
            # 주의사항 : 모든 노드의 공급량이 num_clusters * limit 이하, 수요량이 -(num_clusters * limit) 이상이어야 함.

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
    [  51,  -37,  -20],
    [ -80,  -50,   74],
    [  88,   69,    3],
    [  -8,  -86,   61],
    [   2,   74,  -47],
    [ -26,   89,  -66],
    [ -86,   66,    5],
    [  16,  -50,   79],
    [   3,   89,  -51],
    [  21,  -80,  -57],
    [   2,  -28,   33],
    [   6,  -46,  -97],
    [  30,   87,   31],
    [ -55,   34,  -99],
    [ -29,   30,   45],
    [  -1,    7,   90],
    [ -13,   89,   63],
    [  79, -257,  -47]
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


def solve_divisible_balanced_clustering_gurobi(demand_data, dist_matrix, num_clusters=4, max_solutions=10, solution_ratio=solution_ratio):
    """
    Gurobi를 사용하여 정수 분할 가능한 균형 클러스터링 문제 해결
    각 노드는 여러 클러스터에 정수 비율로 분할 가능
    
    Parameters:
    - demand_data: (n_nodes x n_items) torch.Tensor
    - dist_matrix: (n_nodes x n_nodes) torch.Tensor
    - num_clusters: 클러스터 수
    - max_solutions: 최대 저장할 솔루션 수
    - solution_ratio: 목적함수 차이 허용 비율 (0.1 = 10%)
    - alpha, beta: 목적함수 내 분할복잡도/거리항 가중치
    
    Returns:
    - all_solutions: [(allocation, cluster_contributions, obj_value), ...]
    """
    demand_np = demand_data.numpy()
    dist_np = dist_matrix.numpy()
    n_nodes, n_items = demand_np.shape

    model = gp.Model("divisible_balanced_clustering_v3")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 300)
    model.setParam('MIPGap', solution_ratio)

    # split[i,j,k]: 노드 i의 품목 j가 클러스터 k에 기여하는 양
    split, abs_split = {}, {}
    for i in range(n_nodes):
        for j in range(n_items):
            for k in range(num_clusters):
                if demand_np[i, j] > 0:
                    # 공급: 0 ~ min(1000, 실제 공급량)
                    ub = min(limit, int(demand_np[i, j]))
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"split_{i}_{j}_{k}")
                elif demand_np[i, j] < 0:
                    # 수요: max(-1000, 실제 수요량) ~ 0
                    lb = max(-limit, int(demand_np[i, j]))
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=lb, ub=0, name=f"split_{i}_{j}_{k}")
                else:
                    split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f"split_{i}_{j}_{k}")

                abs_split[i, j, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"abs_split_{i}_{j}_{k}")
                model.addConstr(abs_split[i, j, k] >= split[i, j, k])
                model.addConstr(abs_split[i, j, k] >= -split[i, j, k])

    # 제약 1: split의 합은 원래 값과 같아야 함
    for i in range(n_nodes):
        for j in range(n_items):
            model.addConstr(gp.quicksum(split[i, j, k] for k in range(num_clusters)) == demand_np[i, j], name=f"split_sum_{i}_{j}")

    # 제약 2: 각 클러스터의 품목별 합은 0
    for k in range(num_clusters):
        for j in range(n_items):
            model.addConstr(gp.quicksum(split[i, j, k] for i in range(n_nodes)) == 0, name=f"cluster_balance_{k}_{j}")

    # 제약 3: 클러스터가 비어있지 않도록 (한 클러스터 당 노드가 최소 한 개 이상)
    for k in range(num_clusters):
        model.addConstr(gp.quicksum(abs_split[i, j, k] for i in range(n_nodes) for j in range(n_items)) >= 1, name=f"nonempty_{k}")

    # 거리 제약 조건 : 클러스터 기여량(한 클러스터 내 수요, 공급량이 할당량)이 높을 수록 중심에 배치되도록 설정
    # 노드별 클러스터 기여량
    contribution = {}
    for i in range(n_nodes):
        for k in range(num_clusters):
            contribution[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"contribution_{i}_{k}")
            model.addConstr(contribution[i, k] == gp.quicksum(abs_split[i, j, k] for j in range(n_items)), name=f"contrib_sum_{i}_{k}")

    # 쌍거리 기반 TPC 항 추가
    pairwise_terms = []
    for k in range(num_clusters):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # i와 j가 모두 클러스터 k에 기여한 경우만 거리 계산
                pairwise_terms.append(
                    dist_np[i][j] *
                    contribution[i, k] *
                    contribution[j, k]
                )
    distance_cost = gp.quicksum(pairwise_terms)


    # 분할 복잡도 항
    # total_splits = gp.quicksum(abs_split[i, j, k] for i in range(n_nodes) for j in range(n_items) for k in range(num_clusters))

    # 목적함수
    model.setObjective(distance_cost, GRB.MINIMIZE)

    # 최적화
    model.setParam(GRB.Param.PoolSearchMode, 2)
    model.setParam(GRB.Param.PoolSolutions, max_solutions)
    model.setParam(GRB.Param.PoolGap, solution_ratio)
    model.optimize()

    if model.status == gp.GRB.INFEASIBLE:
        print("모델 infeasible! IIS 계산 중...")
        model.computeIIS()
        model.write("model.ilp")         # 전체 모델 + IIS 정보 포함
        model.write("model_iis.ilp")     # 또는 IIS 추적용으로 따로 저장

    all_solutions = []
    for sol_idx in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, sol_idx)
        allocation = {}
        cluster_contributions = [np.zeros(n_items) for _ in range(num_clusters)]

        for i in range(n_nodes):
            allocation[i] = {}
            for j in range(n_items):
                allocation[i][j] = {}
                for k in range(num_clusters):
                    val = split[i, j, k].Xn
                    if abs(val) > 1e-6:
                        allocation[i][j][k] = int(round(val))
                        cluster_contributions[k][j] += allocation[i][j][k]

        all_solutions.append((allocation, cluster_contributions, model.PoolObjVal))

    # 각 솔루션 확인 및 저장
    all_solutions = []
    for sol_idx in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, sol_idx)
        print(f"\n솔루션 {sol_idx+1}:")

        allocation = {}
        cluster_contributions = [np.zeros(n_items) for _ in range(num_clusters)]

        for i in range(n_nodes):
            allocation[i] = {}
            for j in range(n_items):
                allocation[i][j] = {}
                for k in range(num_clusters):
                    val = split[i, j, k].Xn
                    if abs(val) > 1e-6:
                        allocation[i][j][k] = int(round(val))
                        cluster_contributions[k][j] += allocation[i][j][k]

        # 솔루션 별 클러스터 거리 비교 (단순 비교 용, 목적함수에 포함 X)
        # 클러스터별 TPC 계산 (거리기반)
        cluster_TPCs = []
        for k in range(num_clusters):
            nodes_in_k = [i for i in range(n_nodes) if contribution[i, k].Xn > 1e-6]

            tpc = 0.0
            for i in range(len(nodes_in_k)):
                for j in range(i + 1, len(nodes_in_k)):
                    ni, nj = nodes_in_k[i], nodes_in_k[j]
                    tpc += dist_np[ni][nj]  # 단순 거리 합
            cluster_TPCs.append(tpc)

        # 출력
        print(f"목적함수 값: {model.PoolObjVal:.2f} (총 TPC: 분할 기반)")
        print("거리 기반 TPC (Pairwise Distance Only):")
        for k, tpc in enumerate(cluster_TPCs):
            print(f" - 클러스터 {k}: 거리기반 TPC = {tpc:.2f}")

        # 솔루션 저장
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

# def visualize_clusters_graph(allocation, center_nodes, dist_matrix, num_clusters=4):
#     """
#     클러스터 결과를 네트워크 그래프로 시각화하는 함수
#     """
#     G = nx.Graph()
#     pos = {}  # 노드 좌표

#     # 노드 생성
#     for i in range(len(dist_matrix)):
#         G.add_node(i)
    
#     # 거리 기반 layout
#     spring_layout_pos = nx.spring_layout(G, weight=None, seed=42)
#     for i in range(len(dist_matrix)):
#         pos[i] = spring_layout_pos[i]
    
#     # 노드별 소속 클러스터 추출
#     node_clusters = {i: set() for i in range(len(dist_matrix))}
#     for i in allocation:
#         for j in allocation[i]:
#             for k in allocation[i][j]:
#                 if abs(allocation[i][j][k]) > 0:
#                     node_clusters[i].add(k)

#     # 색상 팔레트
#     color_map = plt.cm.get_cmap('tab10', num_clusters)
    
#     # 노드 시각화
#     for k in range(num_clusters):
#         nodes_k = [i for i in range(len(dist_matrix)) if k in node_clusters[i]]
#         nx.draw_networkx_nodes(G, pos,
#                                nodelist=nodes_k,
#                                node_color=[color_map(k)] * len(nodes_k),
#                                label=f'Cluster {k+1}',
#                                node_size=400,
#                                edgecolors='black')
    
#     # 중심 노드 강조
#     for k, center_node in center_nodes.items():
#         if center_node in G.nodes:
#             nx.draw_networkx_nodes(G, pos,
#                                    nodelist=[center_node],
#                                    node_color='white',
#                                    edgecolors='red',
#                                    node_size=800,
#                                    linewidths=2)

#     # 라벨 추가
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

#     # 거리 기반 edge (optionally show only short links)
#     for i in range(len(dist_matrix)):
#         for j in range(i+1, len(dist_matrix)):
#             if dist_matrix[i][j] < 80:  # 가까운 노드만 연결
#                 G.add_edge(i, j, weight=1.0)
#     nx.draw_networkx_edges(G, pos, alpha=0.2)

#     plt.title("Node Split Result")
#     plt.axis('off')
#     plt.legend()
#     plt.show()

def plot_cluster_on_map(allocation, num_clusters):
    city_names = [
        '춘천시', '원주시', '강릉시', '동해시', '태백시', '속초시', '삼척시',
        '홍천군', '횡성군', '영월군', '평창군', '정선군', '철원군',
        '화천군', '양구군', '인제군', '고성군', '양양군'
    ]

    city_coords = {
        "춘천시": (37.8813, 127.7298), "원주시": (37.3422, 127.9207), "강릉시": (37.7519, 128.8761),
        "동해시": (37.5244, 129.1145), "태백시": (37.1641, 128.9852), "속초시": (38.2044, 128.5912),
        "삼척시": (37.4456, 129.1652), "홍천군": (37.6968, 127.8881), "횡성군": (37.4877, 127.9843),
        "영월군": (37.1833, 128.4655), "평창군": (37.3705, 128.3891), "정선군": (37.3793, 128.6602),
        "철원군": (38.1464, 127.3137), "화천군": (38.1066, 127.7062), "양구군": (38.1054, 127.9892),
        "인제군": (38.0676, 128.1676), "고성군": (38.3796, 128.4672), "양양군": (38.0760, 128.6285)
    }

    # 노드별 클러스터 정보
    node_clusters = {i: set() for i in range(len(city_names))}
    for i in allocation:
        for j in allocation[i]:
            for k in allocation[i][j]:
                if abs(allocation[i][j][k]) > 0:
                    node_clusters[i].add(k)

    # 도시별 클러스터 리스트
    city_to_clusters = {}
    for i, name in enumerate(city_names):
        clusters = list(node_clusters[i])
        city_to_clusters[name] = clusters

    # gdf_points: 도시 중심 포인트용
    records = []
    for name, (lat, lon) in city_coords.items():
        clusters = city_to_clusters.get(name, [])
        for k in clusters:
            records.append({
                'city': name,
                'cluster': k,
                'geometry': Point(lon, lat)
            })
    gdf_points = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # 행정구역 GeoDataFrame
    gdf = gpd.read_file("C:/Users/Admin/Desktop/강원도 shp/sig.shp", encoding="euc-kr")
    gdf['SIG_CD'] = gdf['SIG_CD'].astype(str)

    # 강원도만 추출
    gangwon = gdf[
        (gdf['SIG_KOR_NM'].isin(city_names)) &
        (gdf['SIG_CD'].str.startswith('51'))
    ]

    # gdf의 CRS가 없는 경우 지정
    if gdf.crs is None:
        gdf.set_crs(epsg=5179, inplace=True)
    
    # gdf_points를 gdf의 CRS로 변환
    gdf_points = gdf_points.to_crs(gdf.crs)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for k in range(num_clusters):
        ax = axes[k]

        # 전체 배경 지도
        gangwon.plot(ax=ax, color='beige', edgecolor='black', alpha=0.3)

        # 클러스터 k에 포함된 도시들
        cluster_cities = [city for city, clusters in city_to_clusters.items() if k in clusters]

        # 클러스터에 해당하는 행정구역만 주황색으로 표시
        gangwon_k = gangwon[gangwon['SIG_KOR_NM'].isin(cluster_cities)]
        gangwon_k.plot(ax=ax, color='orange', edgecolor='black', alpha=0.8)

        # 중심 포인트와 라벨 (선택)
        # gdf_k = gdf_points[gdf_points['cluster'] == k]
        # gdf_k.plot(ax=ax, color='red', markersize=80, edgecolor='black')
        centroids = gangwon_k.geometry.centroid
        for x, y, label in zip(centroids.x, centroids.y, gangwon_k['SIG_KOR_NM']):
            ax.text(x, y, label, fontsize=9, ha='center', va='center')

        ax.set_title(f"구역 {k + 1}")
        ax.axis('off')

    plt.suptitle("구역 분할 결과", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# 메인 실행
if __name__ == "__main__":
    print("분할 가능한 균형 클러스터링 최적화 시작")
    print(f"입력 데이터 shape: {fixed_net_demand.shape}")
    
    # Gurobi로 최적화 실행
    all_solutions = solve_divisible_balanced_clustering_gurobi(fixed_net_demand, dist_matrix, num_clusters=num_clusters)
    
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
            
            # 시각화 실행
            # visualize_clusters_graph(allocation, center_nodes, dist_matrix, num_clusters=num_clusters)
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

            # 결과 시각화
            plot_cluster_on_map(allocation, num_clusters=num_clusters)
            
        else:
            print("균형이 맞는 솔루션이 없습니다.")
            
    else:
        print("분할 가능한 균형 클러스터링에 실패했습니다.")
