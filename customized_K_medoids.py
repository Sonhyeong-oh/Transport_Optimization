import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
import warnings
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

# K-medoids 알고리즘에서 거리 제한이 가능하도록 수정
def constrained_k_medoids(dist_matrix, n_clusters, max_iter=10000, threshold=100, random_state=0):
    np.random.seed(random_state)
    n_samples = dist_matrix.shape[0]

    # 1. 초기 medoid 선택
    medoids = np.random.choice(n_samples, size=n_clusters, replace=False).tolist()

    for iteration in range(max_iter):
        clusters = defaultdict(list)

        # 2. 제약 조건을 고려한 할당
        for i in range(n_samples):
            min_dist = float('inf')
            best_medoid = None
            for m in medoids:
                if dist_matrix[i][m] < threshold:  # 제약 조건 확인
                    if dist_matrix[i][m] < min_dist:
                        min_dist = dist_matrix[i][m]
                        best_medoid = m
            if best_medoid is not None:
                clusters[best_medoid].append(i)

        # 제약 위반 확인: 클러스터 내 모든 쌍의 거리가 threshold 미만인지
        def is_valid_cluster(members):
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    if dist_matrix[members[i]][members[j]] >= threshold:
                        return False
            return True

        # 유효한 클러스터만 유지
        clusters = {m: v for m, v in clusters.items() if is_valid_cluster(v)}

        # 3. 새로운 Medoid 선택
        new_medoids = []
        for cluster in clusters.values():
            min_total_dist = float('inf')
            best_candidate = None
            for i in cluster:
                total_dist = sum(dist_matrix[i][j] for j in cluster)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_candidate = i
            new_medoids.append(best_candidate)

        # 4. 종료 조건 확인
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids

    # 최종 라벨링
    labels = [-1] * n_samples
    for label, m in enumerate(medoids):
        for i in clusters.get(m, []):
            labels[i] = label

    return {
        'medoids': medoids,
        'labels': labels,
        'clusters': clusters,
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
        result = constrained_k_medoids(
            dist_matrix=dist_matrix,
            n_clusters=n_clusters,
            threshold=threshold,
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
                        n_clusters=4, max_iter=1000, init_temp=10.0, cooling=0.995):
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

# ------------------------------------------------------------------------------------------------
# ------------------------------------- 클러스터링 코드 --------------------------------------------
# ------------------------------------------------------------------------------------------------

# 파라미터 지정
'''
목적함수 = 잔여 물량 + lambda_dist * 거리 + penalty(클러스터 내 노드 개수가 2개 이하일 때 부여)
이 목적함수가 최소가 되게 하도록 작동
lambda_dist = 1 : 잔여 물량과 거리를 똑같이 고려
lambda_dist < 1 : 잔여 물량을 더 중요하게 고려
lambda_dist > 1 : 거리를 더 중요하게 고려
'''
# K-Medoids
n_trials = 1000 # K-Medoids 클러스터링 실험 횟수
threshold = 100 # 거리 제한 (거리가 100 이상인 군집이 생성되지 않도록 함함)

# 최적화
lambda_dist = 0.5 # 목적함수 중 거리 감소 중요도 비율 지정 
n_clusters = 3 # 군집 수 (가용 차량 댓수로 치환 가능)
max_iter = 300000 # 군집 조합 탐색 횟수
edges_added = 0 # 이미 연결된 노드 수
num_link = 30 # 연결할 링크(도로) 갯수 지정


# === 데이터 준비 ===
# 각 노드별 수요, 공급 행렬 (행 = 노드(마을) / 열 = 품목)
fixed_net_demand = torch.tensor([
    [ 0,   0,  0], [-3,  -2,  2], [-2,   4,  0], [ 2,  -3,  2], [ 4,  -2, -2],
    [ 0,   1, -1], [-1,   2, -1], [ 1,  -1,  0], [-2,   0,  1], [ 2,   2, -3]
    # [ 0,   0,  1], [-1,   1, -1], [ 1,   1,  1], [ 0,  -2,  2], [-2,  -1,  1],
    # [ 2,  -2,  0], [-1,   2,  1], [ 3,  -1, -1], [ 0,   1, -2], [-1,   0,  2], [ 1,  -1, -1]
], dtype=torch.int32)

size = fixed_net_demand.shape[0]
rng = np.random.default_rng(seed=42)
dist_matrix = np.full((size, size), np.inf)
np.fill_diagonal(dist_matrix, 0.0)


'''
도로 데이터를 임의로 생성
실제 도로 데이터를 구하면 이 코드는 삭제하고 위의 fixed_net_demand처럼 직접 입력
'''
while edges_added < num_link:
    i, j = rng.integers(0, size, size=2)
    if i != j and dist_matrix[i][j] == np.inf:
        dist = rng.uniform(1.0, 3.0) # 1~3 중 unifrom dist로 하나 선택
        dist_matrix[i][j] = dist_matrix[j][i] = dist
        edges_added += 1

# 그래프 구성
G = nx.Graph()
for i in range(size):
    for j in range(i + 1, size):
        if np.isfinite(dist_matrix[i][j]) and dist_matrix[i][j] > 0:
            G.add_edge(i, j, weight=1 / dist_matrix[i][j])
            
pos = nx.spring_layout(G, seed=42, weight='weight')

# 거리 → 유사도 → 거리
tf_dist_matrix = np.where(np.isinf(dist_matrix), 100.0, dist_matrix)
dist_sim = np.exp(-tf_dist_matrix)
combined_dist = 1 - dist_sim
np.fill_diagonal(combined_dist, 0.0)


# 클러스터링
# best = KMedoids 결과
results = evaluate_clustering(fixed_net_demand.numpy(), dist_matrix=combined_dist, 
                              n_clusters=n_clusters, trials=n_trials, threshold=threshold)
best = results[0]
labels = best['labels']

import numpy as np
import copy
import random

init_labels = best['labels']
optimized_labels, optimized_score = simulated_annealing(
    fixed_net_demand.numpy(),
    init_labels,
    tf_dist_matrix, 
    lambda_dist=lambda_dist,
    n_clusters=n_clusters,
    max_iter=max_iter
)

# 각 클러스터별 결과 출력 & 시각화
visualize_clusters(G, pos, optimized_labels, title="Metaheuristic Optimization Result")
print_cluster_details_and_paths(fixed_net_demand.numpy(), optimized_labels, tf_dist_matrix, G, pos)

# 총 불균형 점수 계산
imbalance_score = supply_demand_imbalance_score(fixed_net_demand.numpy(), optimized_labels)

print("\n✅ 목적함수 점수:", optimized_score)
print(f"📊 불균형 점수 총합: {imbalance_score}")
