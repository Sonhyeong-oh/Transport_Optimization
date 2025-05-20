import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
import warnings
import random
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
*** 용어 설명 ***

잔여 물량 = 해당 군집으로 배송 진행 시 초과된 or 부족한 물량
불균형 점수 = 잔여 물량의 물품 별(열 별) 절댓값을 모두 더한 값

ex) 군집 경로로 모두 배송한 후의 사과, 배, 바나나 = [2, -1, 0] : 잔여 물량
사과는 2만큼 남음(공급 과잉), 배는 1개 부족함(공급 부족), 바나나는 수요와 공급이 일치했음
불균형 점수 = |2| + |-1| + |0| = 3

방문 경로 = 왼쪽에서 오른쪽으로 배송 진행
'''

# ---------------------------------------------------------------------------------------------------------------
# ------------------------------------------ 클러스터링에 필요한 함수 선언 ------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

import numpy as np
import random
from collections import defaultdict

# 공급, 수요 벡터 랜덤 생성 함수
def generate_fixed_net_demand(n_nodes, n_features=3, low=-3, high=3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)  # 재현성 위해 시드 고정
    return torch.randint(low=low, high=high + 1, size=(n_nodes, n_features), dtype=torch.int32)

# K-medoids 알고리즘 함수
def k_medoids(dist_matrix, n_clusters, random_state=0):
    model = KMedoids(
        n_clusters=n_clusters,
        metric='precomputed',  # 거리 행렬 직접 제공
        method='alternate',    # or 'pam' for classic method
        init='k-medoids++',    # 초기값 선택 방식
        random_state=random_state
    )
    model.fit(dist_matrix)
    
    return {
        'medoids': model.medoid_indices_.tolist(),
        'labels': model.labels_.tolist(),
        'clusters': {
            m: [i for i, label in enumerate(model.labels_) if label == cid]
            for cid, m in enumerate(model.medoid_indices_)
        }
    }



# === 불균형 점수 계산 ===
def supply_demand_imbalance_score(fixed_net_demand, labels):
    total_imbalance = 0
    for cid in np.unique(labels):
        nodes = np.where(labels == cid)[0]
        subtotal = fixed_net_demand[nodes].sum(axis=0)
        imbalance = np.abs(subtotal).sum()
        total_imbalance += imbalance
    return total_imbalance


# === 클러스터 내부 편도 경로 계산 및 반환 ===
def intra_cluster_greedy_path(cluster_nodes, dist_matrix):
    if len(cluster_nodes) <= 1:
        return cluster_nodes, 0.0
    unvisited = set(cluster_nodes)
    current = unvisited.pop()
    path = [current]
    total = 0.0
    while unvisited:
        next_node = min(unvisited, key=lambda x: dist_matrix[current][x])
        total += dist_matrix[current][next_node]
        path.append(next_node)
        current = next_node
        unvisited.remove(current)
    return path, total


# === 클러스터 평가 ===
def evaluate_clustering(fixed_net_demand, dist_matrix, n_clusters=4, trials=10, threshold=100):
    best_results = []

    for seed in range(trials):
        result = k_medoids(
            dist_matrix=dist_matrix,
            n_clusters=n_clusters,
            random_state=seed
        )

        labels = result['labels']

        # -1 (할당되지 않은 노드)가 포함된 경우 silhouette 계산이 불가능하므로 예외 처리
        if len(set(labels)) <= 1 or -1 in labels:
            sil_score = -1
        else:
            try:
                sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
            except:
                sil_score = -1

        imbalance_score = supply_demand_imbalance_score(fixed_net_demand, labels)

        best_results.append({
            'seed': seed,
            'labels': labels,
            'silhouette': sil_score,
            'imbalance': imbalance_score,
            'medoids': result['medoids'],
            'unassigned': labels.count(-1)  # 할당되지 않은 노드 수
        })

    sorted_results = sorted(best_results, key=lambda x: (-x['silhouette'], x['imbalance'], x['unassigned']))
    return sorted_results

# === 목적함수 계산 함수 ===
def calculate_objective(
    fixed_net_demand, labels, dist_matrix,
    lambda_dist=1.0,
    penalty_singleton=5000,
    penalty_unconnected=50000
):
    total_imbalance = 0
    total_distance = 0
    penalty = 0

    for cid in np.unique(labels):
        nodes = np.where(labels == cid)[0].tolist()
        # 클러스터 내 노드가 2개 이하라면 패널티 부여
        if len(nodes) <= 2:
            penalty += penalty_singleton
            continue

        subtotal = fixed_net_demand[nodes].sum(axis=0)
        imbalance = np.abs(subtotal).sum()
        total_imbalance += imbalance

        path, tour_length = intra_cluster_greedy_path(nodes, dist_matrix)
        total_distance += tour_length

        # ⚠️ 경로 중 연결 안 된 노드가 있다면 패널티 부여
        for i in range(len(path) - 1):
            if np.isinf(dist_matrix[path[i]][path[i+1]]):
                penalty += penalty_unconnected
                break

    return total_imbalance + lambda_dist * total_distance + penalty

# === 노드를 다른 클러스터로 옮기는 알고리즘 ===
# simulate_annealing 알고리즘에서 군집을 최적화할 때 사용
def generate_neighbor(labels, n_clusters):
    new_labels = labels.copy()
    idx = np.random.randint(0, len(labels))
    current_cluster = labels[idx]
    new_cluster = random.choice([c for c in range(n_clusters) if c != current_cluster])
    new_labels[idx] = new_cluster
    return new_labels


# === 담금질 기법 (전역 최적화 문제에 대한 일반적인 확률적 메타 알고리즘) ===
# K-Medoids로 생성된 군집을 수요와 거리를 함께 고려하여 최적화 하는 알고리즘
'''
temp = 무작위성 정도 (높을수록 나쁜 결과도 수용, 낮을수록 좋은 결과만 수용)
cooling = 냉각률 (temp를 조정)
초반 temp를 10으로 설정하여 무작위 탐색을 진행하도록하고, 그 후 냉각률을 곱해 좋은 결과로 수렴하도록 함함
'''
def simulated_annealing(fixed_net_demand, init_labels, dist_matrix, lambda_dist=1.0,
                        n_clusters=4, max_iter=100000, init_temp=10.0, cooling=0.995):
    current_labels = init_labels.copy()
    current_score = calculate_objective(fixed_net_demand, current_labels, dist_matrix, lambda_dist)
    best_labels = current_labels.copy()
    best_score = current_score
    temp = init_temp

    for step in range(max_iter):
        neighbor = generate_neighbor(current_labels, n_clusters)
        neighbor_score = calculate_objective(fixed_net_demand, neighbor, dist_matrix, lambda_dist)
        delta = neighbor_score - current_score

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_labels = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_labels = current_labels.copy()
                best_score = current_score

        # 학습 횟수가 많아질 때 temp가 과도하게 낮아지는 것을 방지
        temp = max(temp * cooling, 1e-6)

    return best_labels, best_score

def multiple_runs_sa(fixed_net_demand, init_labels, dist_matrix, 
                     n_clusters, lambda_dist, max_iter, n_runs=10):
    best_overall_labels = None
    best_overall_score = float('inf')

    for i in range(n_runs):
        labels, score = simulated_annealing(
            fixed_net_demand, init_labels, dist_matrix,
            lambda_dist=lambda_dist, n_clusters=n_clusters, max_iter=max_iter
        )
        if score < best_overall_score:
            best_overall_score = score
            best_overall_labels = labels.copy()

    return best_overall_labels, best_overall_score


# -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ 클러스터링 시각화, 결과 출력 함수 선언 ------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# === 각 클러스터 별 경로 출력 함수 ===
def print_cluster_details_and_paths(fixed_net_demand, labels, dist_matrix, G, pos):
    print(f"\n✅ 수급 불균형 점수(전체): {supply_demand_imbalance_score(fixed_net_demand, labels):.2f}")
    for cluster_id in np.unique(labels):
        members = np.where(labels == cluster_id)[0].tolist()

        # 0이 클러스터에 있으면 포함, 아니면 제외
        if 0 in members:
            path_nodes = members.copy()
        else:
            path_nodes = [n for n in members if n != 0]

        subtotal = fixed_net_demand[members].sum(axis=0)
        imbalance = np.abs(subtotal).sum()
        path, tour_length = intra_cluster_greedy_path(path_nodes, dist_matrix)

        print(f"\n클러스터 {cluster_id}: {members}")
        print(f"  🔹 잔여 물량 벡터: {subtotal.tolist()} → 불균형 점수: {imbalance}")
        print(f"  🔹 편도 이동 거리: {tour_length:.2f}")
        print(f"  🔹 방문 경로: {path}")
        visualize_path_on_graph(G, pos, labels, path, cluster_id, tour_length, fixed_net_demand)


# === 군집 그래프 + 경로 화살표 시각화 ===
def visualize_path_on_graph(G, pos, labels, path, cluster_id, tour_length, fixed_net_demand):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")
    plt.figure(figsize=(10, 7))

    legend_patches = []

    for cid in unique_labels:
        nodes = np.where(labels == cid)[0]
        color = cmap(cid % 10)

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=[color],
            node_size=600
        )
        legend_patches.append(mpatches.Patch(color=color, label=f"Cluster {cid}"))

        # 📌 노드 위에 수요 벡터 표시 (해당 클러스터만)
        if cid == cluster_id:
            for node in nodes:
                demand = fixed_net_demand[node].tolist()
                demand_text = str(demand)
                plt.text(pos[node][0], pos[node][1] + 0.07, demand_text,
                         fontsize=8, ha='center', va='bottom', color='black')

    # 현재 클러스터의 불균형 점수 계산
    nodes = np.where(labels == cluster_id)[0]
    subtotal = fixed_net_demand[nodes].sum(axis=0)
    imbalance = np.abs(subtotal).sum()

    # 기본 엣지 및 라벨
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.2)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    # 빨간 경로 강조
    edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edge_list,
        edge_color='red',
        width=4.0,
        arrows=True,
        arrowsize=30,
        connectionstyle='arc3,rad=0.15'
    )

    # 제목에 거리 + 불균형 점수 출력
    plt.title(f"Cluster {cluster_id} Path (Distance={tour_length:.2f}, Imbalance={imbalance:.2f})")

    # 범례
    plt.legend(handles=legend_patches, fontsize='small', loc='best', frameon=False)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# === 클러스터링 된 전체 지도를 출력 ===
def visualize_clusters(G, pos, labels, title):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")

    plt.figure(figsize=(8, 6))

    legend_patches = []  # 수동 범례 구성용

    for cluster_id in unique_labels:
        nodes = np.where(labels == cluster_id)[0]
        color = cmap(cluster_id % 10)
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes,
            node_color=[color], node_size=600
        )
        # 범례 패치 추가 (마커 크기 수동 지정)
        legend_patches.append(mpatches.Patch(color=color, label=f"Cluster {cluster_id}"))

    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.2)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    plt.title(title)
    plt.axis('off')

    # 범례 크기 조절 (마커 작게 설정됨)
    plt.legend(handles=legend_patches, fontsize='small', loc='best', frameon=False)

    plt.tight_layout()
    plt.show()
