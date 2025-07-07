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
limit = 15 # 적재 한도 (수요량, 공급량 제한 / 수요량 분할 범위 : -limit ~ 0 / 공급량 분할 범위 : 0 ~ limit)
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
    [  -3,   -2,   -4],
    [   6,   -2,   -2],
    [  -9,   -9,   -1],
    [  -4,   -3,    6],
    [   8,   -8,  -10],
    [   4,   -4,  -10],
    [  10,    5,    2],
    [  -4,   -4,   -5],
    [  -3,   10,    6],
    [   1,  -10,    1],
    [  -9,   -6,  -10],
    [  -8,    8,    4],
    [   0,  -10,   -4],
    [  -2,   50,    9],
    [  10,   -3,    6],
    [   1,   -3,   -3],
    [  -6,    1,    9],
    [ -10,    7,    6],
    [   8,    0,   -8],
    [ -32,   -3,    0],
    [  -4,    6,    9],
    [   1,   -2,   -6],
    [  10,    2,   -2],
    [   4,   -1,   27],
    [   1,    1,    9],
    [   8,    4,   -9],
    [  -7,    2,   -4],
    [  -4,    5,    6],
    [   9,  -10,   -8],
    [  10,   -4,    1],
    [   4,    4,    1],
    [  10,   -8,   -9],
    [   9,    2,    8],
    [   5,  -10,   -5],
    [   0,    1,   -6],
    [  -8,   10,    6],
    [  -1,   -6,   -9],
    [   0,   -2,   -2],
    [  -5,   -8,    1]
], dtype=torch.float32)

# 노드 간 거리 행렬
dist_matrix = torch.tensor([
    [0.00, 0.19, 0.49, 0.24, 0.53, 1.27, 0.49, 0.42, 0.81, 0.73, 0.88, 1.60, 1.05, 0.56, 0.44, 0.93, 1.39, 3.74, 3.46, 2.12, 1.84, 0.64, 4.69, 3.44, 2.39, 4.09, 4.11, 4.27, 2.62, 8.66, 10.03, 13.21, 8.65, 20.01, 7.81, 15.95, 18.31, 7.07, 15.12],
    [0.19, 0.00, 0.42, 0.29, 0.56, 1.25, 0.61, 0.58, 0.83, 0.89, 0.94, 1.77, 1.04, 0.46, 0.30, 0.86, 1.22, 3.71, 3.36, 2.31, 1.81, 0.82, 4.81, 3.62, 2.28, 4.07, 4.09, 4.23, 2.44, 8.62, 10.19, 13.34, 8.69, 19.98, 7.63, 15.81, 18.42, 7.24, 15.08],
    [0.49, 0.42, 0.00, 0.30, 0.28, 0.83, 0.98, 0.63, 0.46, 0.86, 0.64, 1.73, 1.46, 0.84, 0.21, 0.44, 1.02, 4.11, 3.67, 2.46, 1.39, 1.05, 4.56, 3.69, 1.91, 3.65, 3.67, 4.63, 2.31, 9.02, 10.46, 13.11, 8.30, 19.56, 7.46, 16.00, 18.80, 7.16, 14.66],
    [0.24, 0.29, 0.30, 0.00, 0.29, 1.04, 0.71, 0.37, 0.57, 0.65, 0.65, 1.54, 1.28, 0.75, 0.35, 0.70, 1.29, 3.97, 3.64, 2.19, 1.61, 0.75, 4.53, 3.45, 2.20, 3.85, 3.87, 4.50, 2.56, 8.89, 10.16, 13.06, 8.42, 19.78, 7.73, 16.08, 18.52, 7.00, 14.89],
    [0.53, 0.56, 0.28, 0.29, 0.00, 0.76, 0.98, 0.45, 0.28, 0.62, 0.38, 1.46, 1.58, 1.02, 0.47, 0.49, 1.27, 4.26, 3.90, 2.26, 1.32, 0.93, 4.30, 3.45, 2.00, 3.56, 3.58, 4.79, 2.56, 9.18, 10.29, 12.84, 8.13, 19.49, 7.69, 16.27, 18.76, 6.89, 14.61],
    [1.27, 1.25, 0.83, 1.04, 0.76, 0.00, 1.74, 1.18, 0.50, 1.22, 0.63, 1.82, 2.29, 1.66, 1.01, 0.42, 1.21, 4.92, 4.40, 2.83, 0.57, 1.66, 4.03, 3.84, 1.36, 2.82, 2.84, 5.45, 2.39, 9.83, 10.91, 12.58, 7.50, 18.74, 7.33, 16.51, 19.50, 6.96, 13.85],
    [0.49, 0.61, 0.98, 0.71, 0.98, 1.74, 0.00, 0.67, 1.26, 0.90, 1.27, 1.62, 0.74, 0.63, 0.91, 1.41, 1.81, 3.41, 3.31, 1.86, 2.31, 0.46, 4.88, 3.25, 2.88, 4.53, 4.56, 3.94, 2.97, 8.31, 9.61, 13.33, 9.02, 20.47, 8.16, 15.92, 17.82, 7.02, 15.59],
    [0.42, 0.58, 0.63, 0.37, 0.45, 1.18, 0.67, 0.00, 0.68, 0.31, 0.62, 1.19, 1.38, 0.98, 0.72, 0.95, 1.65, 4.07, 3.86, 1.83, 1.72, 0.48, 4.29, 3.08, 2.46, 3.90, 3.93, 4.60, 2.92, 8.97, 9.85, 12.79, 8.36, 19.84, 8.09, 16.38, 18.32, 6.66, 14.98],
    [0.81, 0.83, 0.46, 0.57, 0.28, 0.50, 1.26, 0.68, 0.00, 0.74, 0.23, 1.48, 1.86, 1.28, 0.67, 0.37, 1.28, 4.53, 4.13, 2.39, 1.05, 1.15, 4.11, 3.50, 1.81, 3.28, 3.30, 5.06, 2.56, 9.45, 10.45, 12.66, 7.86, 19.21, 7.63, 16.43, 19.00, 6.82, 14.33],
    [0.73, 0.89, 0.86, 0.65, 0.62, 1.22, 0.90, 0.31, 0.74, 0.00, 0.59, 0.89, 1.64, 1.28, 0.99, 1.08, 1.88, 4.31, 4.16, 1.65, 1.70, 0.56, 4.00, 2.83, 2.55, 3.80, 3.83, 4.83, 3.17, 9.20, 9.71, 12.49, 8.16, 19.74, 8.31, 16.68, 18.33, 6.36, 14.89],
    [0.88, 0.94, 0.64, 0.65, 0.38, 0.63, 1.27, 0.62, 0.23, 0.59, 0.00, 1.25, 1.92, 1.40, 0.85, 0.60, 1.51, 4.62, 4.28, 2.21, 1.11, 1.08, 3.92, 3.28, 1.98, 3.28, 3.30, 5.15, 2.79, 9.53, 10.28, 12.47, 7.78, 19.22, 7.85, 16.63, 18.91, 6.59, 14.36],
    [1.60, 1.77, 1.73, 1.54, 1.46, 1.82, 1.62, 1.19, 1.48, 0.89, 1.25, 0.00, 2.35, 2.13, 1.88, 1.85, 2.73, 4.90, 4.93, 1.16, 2.14, 1.17, 3.33, 2.03, 3.17, 3.88, 3.91, 5.41, 4.02, 9.70, 9.16, 11.71, 7.83, 19.68, 9.11, 17.53, 18.14, 5.47, 14.91],
    [1.05, 1.04, 1.46, 1.28, 1.58, 2.29, 0.74, 1.38, 1.86, 1.64, 1.92, 2.35, 0.00, 0.66, 1.30, 1.89, 1.95, 2.70, 2.58, 2.38, 2.86, 1.18, 5.62, 3.81, 3.20, 5.11, 5.13, 3.23, 2.87, 7.61, 9.72, 14.06, 9.70, 21.01, 7.98, 15.23, 17.55, 7.67, 16.10],
    [0.56, 0.46, 0.84, 0.75, 1.02, 1.66, 0.63, 0.98, 1.28, 1.28, 1.40, 2.13, 0.66, 0.00, 0.66, 1.26, 1.33, 3.26, 2.90, 2.48, 2.23, 1.03, 5.25, 3.86, 2.55, 4.48, 4.50, 3.79, 2.38, 8.18, 10.18, 13.77, 9.14, 20.36, 7.56, 15.40, 18.18, 7.59, 15.44],
    [0.44, 0.30, 0.21, 0.35, 0.47, 1.01, 0.91, 0.72, 0.67, 0.99, 0.85, 1.88, 1.30, 0.66, 0.00, 0.60, 0.95, 3.91, 3.46, 2.53, 1.57, 1.07, 4.77, 3.81, 1.98, 3.82, 3.84, 4.44, 2.21, 8.83, 10.47, 13.31, 8.50, 19.71, 7.39, 15.80, 18.72, 7.33, 14.80],
    [0.93, 0.86, 0.44, 0.70, 0.49, 0.42, 1.41, 0.95, 0.37, 1.08, 0.60, 1.85, 1.89, 1.26, 0.60, 0.00, 0.93, 4.51, 3.99, 2.73, 0.97, 1.41, 4.36, 3.86, 1.51, 3.22, 3.24, 5.03, 2.20, 9.42, 10.78, 12.92, 7.91, 19.12, 7.27, 16.17, 19.22, 7.16, 14.22],
    [1.39, 1.22, 1.02, 1.29, 1.27, 1.21, 1.81, 1.65, 1.28, 1.88, 1.51, 2.73, 1.95, 1.33, 0.95, 0.93, 0.00, 4.23, 3.43, 3.48, 1.55, 2.02, 5.24, 4.71, 1.33, 3.60, 3.61, 4.73, 1.29, 9.05, 11.42, 13.78, 8.44, 19.26, 6.44, 15.31, 19.50, 8.09, 14.31],
    [3.74, 3.71, 4.11, 3.97, 4.26, 4.92, 3.41, 4.07, 4.53, 4.31, 4.62, 4.90, 2.70, 3.26, 3.91, 4.51, 4.23, 0.00, 1.49, 4.48, 5.48, 3.79, 8.22, 5.78, 5.56, 7.72, 7.74, 0.53, 4.51, 4.92, 9.74, 16.49, 12.39, 23.49, 8.71, 13.34, 15.98, 9.75, 18.53],
    [3.46, 3.36, 3.67, 3.64, 3.90, 4.40, 3.31, 3.86, 4.13, 4.16, 4.28, 4.93, 2.58, 2.90, 3.46, 3.99, 3.43, 1.49, 0.00, 4.86, 4.89, 3.76, 8.15, 6.27, 4.69, 7.03, 7.04, 1.75, 3.34, 5.71, 11.07, 16.63, 11.86, 22.51, 7.22, 12.69, 17.45, 10.20, 17.52],
    [2.12, 2.31, 2.46, 2.19, 2.26, 2.83, 1.86, 1.83, 2.39, 1.65, 2.21, 1.16, 2.38, 2.48, 2.53, 2.73, 3.48, 4.48, 4.86, 0.00, 3.23, 1.49, 4.01, 1.42, 4.19, 5.04, 5.07, 4.96, 4.74, 9.06, 8.08, 12.02, 8.80, 20.78, 9.92, 17.55, 16.99, 5.35, 16.03],
    [1.84, 1.81, 1.39, 1.61, 1.32, 0.57, 2.31, 1.72, 1.05, 1.70, 1.11, 2.14, 2.86, 2.23, 1.57, 0.97, 1.55, 5.48, 4.89, 3.23, 0.00, 2.19, 3.79, 4.10, 1.13, 2.26, 2.28, 6.00, 2.55, 10.38, 11.29, 12.29, 6.97, 18.17, 7.22, 16.82, 20.03, 6.94, 13.28],
    [0.64, 0.82, 1.05, 0.75, 0.93, 1.66, 0.46, 0.48, 1.15, 0.56, 1.08, 1.17, 1.18, 1.03, 1.07, 1.41, 2.02, 3.79, 3.76, 1.49, 2.19, 0.00, 4.45, 2.84, 2.92, 4.34, 4.37, 4.32, 3.26, 8.66, 9.41, 12.88, 8.71, 20.28, 8.45, 16.37, 17.85, 6.57, 15.43],
    [4.69, 4.81, 4.56, 4.53, 4.30, 4.03, 4.88, 4.29, 4.11, 4.00, 3.92, 3.33, 5.62, 5.25, 4.77, 4.36, 5.24, 8.22, 8.15, 4.01, 3.79, 4.45, 0.00, 3.44, 4.79, 3.43, 3.46, 8.74, 6.34, 13.01, 10.24, 8.56, 5.06, 17.36, 10.68, 20.53, 20.20, 3.67, 12.88],
    [3.44, 3.62, 3.69, 3.45, 3.45, 3.84, 3.25, 3.08, 3.50, 2.83, 3.28, 2.03, 3.81, 3.86, 3.81, 3.86, 4.71, 5.78, 6.27, 1.42, 4.10, 2.84, 3.44, 0.00, 5.18, 5.45, 5.48, 6.23, 6.00, 10.14, 7.35, 10.83, 8.48, 20.70, 11.13, 18.95, 16.87, 3.98, 16.09],
    [2.39, 2.28, 1.91, 2.20, 2.00, 1.36, 2.88, 2.46, 1.81, 2.55, 1.98, 3.17, 3.20, 2.55, 1.98, 1.51, 1.33, 5.56, 4.69, 4.19, 1.13, 2.92, 4.79, 5.18, 0.00, 2.42, 2.43, 6.05, 1.72, 10.36, 12.26, 13.17, 7.31, 17.93, 6.09, 16.04, 20.70, 8.05, 12.98],
    [4.09, 4.07, 3.65, 3.85, 3.56, 2.82, 4.53, 3.90, 3.28, 3.80, 3.28, 3.88, 5.11, 4.48, 3.82, 3.22, 3.60, 7.72, 7.03, 5.04, 2.26, 4.34, 3.43, 5.45, 2.42, 0.00, 0.03, 8.24, 4.12, 12.61, 12.80, 11.14, 4.89, 15.94, 7.46, 18.34, 22.02, 7.09, 11.09],
    [4.11, 4.09, 3.67, 3.87, 3.58, 2.84, 4.56, 3.93, 3.30, 3.83, 3.30, 3.91, 5.13, 4.50, 3.84, 3.24, 3.61, 7.74, 7.04, 5.07, 2.28, 4.37, 3.46, 5.48, 2.43, 0.03, 0.00, 8.25, 4.13, 12.63, 12.83, 11.15, 4.88, 15.92, 7.44, 18.34, 22.05, 7.12, 11.06],
    [4.27, 4.23, 4.63, 4.50, 4.79, 5.45, 3.94, 4.60, 5.06, 4.83, 5.15, 5.41, 3.23, 3.79, 4.44, 5.03, 4.73, 0.53, 1.75, 4.96, 6.00, 4.32, 8.74, 6.23, 6.05, 8.24, 8.25, 0.00, 4.94, 4.39, 9.84, 16.98, 12.92, 23.98, 8.94, 13.00, 15.71, 10.20, 19.02],
    [2.62, 2.44, 2.31, 2.56, 2.56, 2.39, 2.97, 2.92, 2.56, 3.17, 2.79, 4.02, 2.87, 2.38, 2.21, 2.20, 1.29, 4.51, 3.34, 4.74, 2.55, 3.26, 6.34, 6.00, 1.72, 4.12, 4.13, 4.94, 0.00, 9.04, 12.56, 14.82, 8.99, 19.17, 5.19, 14.33, 20.25, 9.35, 14.18],
    [8.66, 8.62, 9.02, 8.89, 9.18, 9.83, 8.31, 8.97, 9.45, 9.20, 9.53, 9.70, 7.61, 8.18, 8.83, 9.42, 9.05, 4.92, 5.71, 9.06, 10.38, 8.66, 13.01, 10.14, 10.36, 12.61, 12.63, 4.39, 9.04, 0.00, 11.23, 20.97, 17.31, 28.21, 11.88, 11.15, 13.69, 13.94, 23.22],
    [10.03, 10.19, 10.46, 10.16, 10.29, 10.91, 9.61, 9.85, 10.45, 9.71, 10.28, 9.16, 9.72, 10.18, 10.47, 10.78, 11.42, 9.74, 11.07, 8.08, 11.29, 9.41, 10.24, 7.35, 12.26, 12.80, 12.83, 9.84, 12.56, 11.23, 0.00, 14.04, 15.11, 27.45, 17.70, 22.19, 10.57, 7.84, 23.11],
    [13.21, 13.34, 13.11, 13.06, 12.84, 12.58, 13.33, 12.79, 12.66, 12.49, 12.47, 11.71, 14.06, 13.77, 13.31, 12.92, 13.78, 16.49, 16.63, 12.02, 12.29, 12.88, 8.56, 10.83, 13.17, 11.14, 11.15, 16.98, 14.82, 20.97, 14.04, 0.00, 7.86, 16.73, 18.52, 29.08, 24.52, 7.14, 14.17],
    [8.65, 8.69, 8.30, 8.42, 8.13, 7.50, 9.02, 8.36, 7.86, 8.16, 7.78, 7.83, 9.70, 9.14, 8.50, 7.91, 8.44, 12.39, 11.86, 8.80, 6.97, 8.71, 5.06, 8.48, 7.31, 4.89, 4.88, 12.92, 8.99, 17.31, 15.11, 7.86, 0.00, 12.35, 11.30, 23.04, 25.24, 7.59, 8.11],
    [20.01, 19.98, 19.56, 19.78, 19.49, 18.74, 20.47, 19.84, 19.21, 19.74, 19.22, 19.68, 21.01, 20.36, 19.71, 19.12, 19.26, 23.49, 22.51, 20.78, 18.17, 20.28, 17.36, 20.70, 17.93, 15.94, 15.92, 23.98, 19.17, 28.21, 27.45, 16.73, 12.35, 0.00, 18.20, 30.51, 37.56, 19.77, 5.00],
    [7.81, 7.63, 7.46, 7.73, 7.69, 7.33, 8.16, 8.09, 7.63, 8.31, 7.85, 9.11, 7.98, 7.56, 7.39, 7.27, 6.44, 8.71, 7.22, 9.92, 7.22, 8.45, 10.68, 11.13, 6.09, 7.46, 7.44, 8.94, 5.19, 11.88, 17.70, 18.52, 11.30, 18.20, 0.00, 12.49, 24.65, 14.12, 13.36],
    [15.95, 15.81, 16.00, 16.08, 16.27, 16.51, 15.92, 16.38, 16.43, 16.68, 16.63, 17.53, 15.23, 15.40, 15.80, 16.17, 15.31, 13.34, 12.69, 17.55, 16.82, 16.37, 20.53, 18.95, 16.04, 18.34, 18.34, 13.00, 14.33, 11.15, 22.19, 29.08, 23.04, 30.51, 12.49, 0.00, 23.74, 22.89, 25.79],
    [18.31, 18.42, 18.80, 18.52, 18.76, 19.50, 17.82, 18.32, 19.00, 18.33, 18.91, 18.14, 17.55, 18.18, 18.72, 19.22, 19.50, 15.98, 17.45, 16.99, 20.03, 17.85, 20.20, 16.87, 20.70, 22.02, 22.05, 15.71, 20.25, 13.69, 10.57, 24.52, 25.24, 37.56, 24.65, 23.74, 0.00, 18.34, 32.95],
    [7.07, 7.24, 7.16, 7.00, 6.89, 6.96, 7.02, 6.66, 6.82, 6.36, 6.59, 5.47, 7.67, 7.59, 7.33, 7.16, 8.09, 9.75, 10.20, 5.35, 6.94, 6.57, 3.67, 3.98, 8.05, 7.09, 7.12, 10.20, 9.35, 13.94, 7.84, 7.14, 7.59, 19.77, 14.12, 22.89, 18.34, 0.00, 15.70],
    [15.12, 15.08, 14.66, 14.89, 14.61, 13.85, 15.59, 14.98, 14.33, 14.89, 14.36, 14.91, 16.10, 15.44, 14.80, 14.22, 14.31, 18.53, 17.52, 16.03, 13.28, 15.43, 12.88, 16.09, 12.98, 11.09, 11.06, 19.02, 14.18, 23.22, 23.11, 14.17, 8.11, 5.00, 13.36, 25.79, 32.95, 15.70, 0.00],
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
    model.setParam('TimeLimit', 500)
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
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Point

    city_names = [
        '봉의동', '요선동', '낙원동', '중앙로1가', '중앙로2가', '중앙로3가', '옥천동', '조양동', '죽림동',
        '운교동', '약사동', '효자동', '소양로1가', '소양로2가', '소양로3가', '소양로4가', '근화동', '우두동',
        '사농동', '후평동', '온의동', '교동', '퇴계동', '석사동', '삼천동', '칠전동', '송암동', '신동', '중도동',
        '신북읍', '동면', '동산면', '신동면', '남면', '서면', '사북면', '북산면', '동내면', '남산면'
    ]

    city_coords = {
        "봉의동": (37.8844, 127.7305),
        "요선동": (37.8851, 127.7286),
        "낙원동": (37.8820, 127.7258),
        "중앙로1가": (37.8825, 127.7291),
        "중앙로2가": (37.8801, 127.7279),
        "중앙로3가": (37.8753, 127.7216),
        "옥천동": (37.8868, 127.7352),
        "조양동": (37.8811, 127.7329),
        "죽림동": (37.8779, 127.7264),
        "운교동": (37.8787, 127.7348),
        "약사동": (37.8767, 127.7286),
        "효자동": (37.8733, 127.7422),
        "소양로1가": (37.8934, 127.7340),
        "소양로2가": (37.8892, 127.7287),
        "소양로3가": (37.8839, 127.7255),
        "소양로4가": (37.8790, 127.7225),
        "근화동": (37.8847, 127.7147),
        "우두동": (37.9173, 127.7395),
        "사농동": (37.9149, 127.7228),
        "후평동": (37.8786, 127.7536),
        "온의동": (37.8710, 127.7180),
        "교동": (37.8832, 127.7377),
        "퇴계동": (37.8434, 127.7434),
        "석사동": (37.8693, 127.7647),
        "삼천동": (37.8748, 127.7061),
        "칠전동": (37.8530, 127.7062),
        "송암동": (37.8530, 127.7059),
        "신동": (37.9220, 127.7405),
        "중도동": (37.8899, 127.7016),
        "신북읍": (37.9601, 127.7536),
        "동면": (37.8891, 127.8446),
        "동산면": (37.7729, 127.7823),
        "신동면": (37.8091, 127.7055),
        "남면": (37.7385, 127.5971),
        "서면": (37.8981, 127.6433),
        "사북면": (38.0105, 127.6437),
        "북산면": (37.9693, 127.9094),
        "동내면": (37.8371, 127.7843),
        "남산면": (37.7792, 127.6213)
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

    # 도시 중심 포인트
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
    gdf = gpd.read_file("C:/Users/Admin/Desktop/춘천시 shp/emd.shp", encoding="euc-kr")
    gdf['EMD_CD'] = gdf['EMD_CD'].astype(str)
    gangwon = gdf[
        (gdf['EMD_KOR_NM'].isin(city_names)) &
        (gdf['EMD_CD'].str.startswith('5111'))
    ]
    if gdf.crs is None:
        gdf.set_crs(epsg=5179, inplace=True)
    gdf_points = gdf_points.to_crs(gdf.crs)

    # 시각화
    ncols = 2
    nrows = (num_clusters + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten()

    for k in range(num_clusters):
        ax = axes[k]
        gangwon.plot(ax=ax, color='beige', edgecolor='black', alpha=0.3)

        cluster_cities = [city for city, clusters in city_to_clusters.items() if k in clusters]
        gangwon_k = gangwon[gangwon['EMD_KOR_NM'].isin(cluster_cities)]
        gangwon_k.plot(ax=ax, color='orange', edgecolor='black', alpha=0.8)

        # 행정구역 위에 행정구역명 표시
        # centroids = gangwon_k.geometry.centroid
        # for x, y, label in zip(centroids.x, centroids.y, gangwon_k['EMD_KOR_NM']):
        #     ax.text(x, y, label, fontsize=9, ha='center', va='center')

        # 클러스터 내 행정구역명을 subplot 오른쪽에 출력
        # 여기서 범례 위에 노드 수 함께 표시
        cluster_label = f"소속 노드 수: {len(cluster_cities)}개\n" + "\n".join(cluster_cities)
        ax.text(1.05, 0.5, cluster_label, transform=ax.transAxes,
            fontsize=9, va='center', ha='left',
            bbox=dict(boxstyle="round", fc="white", ec="black"))

        ax.set_title(f"구역 {k + 1}")
        ax.axis('off')

    for i in range(num_clusters, len(axes)):
        axes[i].axis('off')  # 남는 subplot 숨김

    plt.suptitle("구역 분할 결과", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
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
