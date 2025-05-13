'''
수정해야 할 사항
1. 실제 데이터로 돌려보기
2. multiple_run_sa 함수의 최적 n_runs 찾기
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
import warnings
import random
import Clustering_function as cf
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
lambda_dist = 0.3 # 목적함수 중 거리 감소 중요도 비율 지정 
n_clusters = 5 # 군집 수 (가용 차량 댓수로 치환 가능)
max_iter = 100000 # 군집 조합 탐색 횟수
n_runs = 3 # 최적화 시행 횟수


# === 데이터 준비 ===
# 각 노드별 수요, 공급 행렬 (행 = 노드(마을) / 열 = 품목)
fixed_net_demand = cf.generate_fixed_net_demand(n_nodes=18)
print(fixed_net_demand)
size = fixed_net_demand.shape[0]
rng = np.random.default_rng(seed=42)


# 시(군)청 별 거리 데이터
dist_matrix = torch.tensor([
    #춘천 원주  강릉  동해  태백 속초  삼척  홍천 횡성  영월  평창 정선  철원  화천 양구  인제  고성  양양
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
])


# 그래프 구성
G = nx.Graph()
for i in range(size):
    for j in range(i + 1, size):
        if np.isfinite(dist_matrix[i][j]) and dist_matrix[i][j] > 0:
            G.add_edge(i, j, weight=1 / dist_matrix[i][j])
            
pos = nx.spring_layout(G, seed=42, weight='weight')

# 거리 → 유사도 → 거리
dist_sim = torch.exp(-dist_matrix)
combined_dist = 1 - dist_sim
combined_dist.fill_diagonal_(0.0)


# 클러스터링
# best = KMedoids 결과
results = cf.evaluate_clustering(fixed_net_demand.numpy(), dist_matrix=combined_dist, 
                              n_clusters=n_clusters, trials=n_trials, threshold=threshold)
best = results[0]
labels = best['labels']

cf.visualize_clusters(G, pos, labels, title="K-Medoids Result")
cf.print_cluster_details_and_paths(fixed_net_demand.numpy(), labels, dist_matrix, G, pos)

init_labels = best['labels']
best_labels, best_score = cf.multiple_runs_sa(
    fixed_net_demand, init_labels, dist_matrix,
    n_clusters=n_clusters, lambda_dist=lambda_dist, 
    max_iter = max_iter, n_runs = n_runs
)

# 각 클러스터별 결과 출력 & 시각화
cf.visualize_clusters(G, pos, best_labels, title="Metaheuristic Optimization Result")
cf.print_cluster_details_and_paths(fixed_net_demand.numpy(), best_labels, dist_matrix, G, pos)

# 총 불균형 점수 계산
imbalance_score = cf.supply_demand_imbalance_score(fixed_net_demand.numpy(), best_labels)

print("\n✅ 목적함수 점수:", best_score)
print(f"📊 불균형 점수 총합: {imbalance_score}")
