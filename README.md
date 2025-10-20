<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>

## Transportation Optimization using Integer Programming & RL

강화학습 연산량 감소를 위한 사전 클러스터링 코드 구현

- 파일 설명
* K-medodis : K-medoids 알고리즘으로 1차 클러스터링 후 수요, 공급 균형을 위한 2차 최적화
* Genetic : 유전 알고리즘으로 수요, 공급 균형과 거리 단축 동시 최적화
* Integer-programming : 정수 계획법으로 완벽한 수요, 공급 균형 달성 후 클러스터 별 총 이동거리 최소화 해를 선택
* Gurobi : 선형 계획법, 정수 계획법 등을 푸는 알고리즘, 자세한 소스코드는 비공개
* RL_transport_optimization : 군집화 후 차량 배차 및 이동 경로 최적화 알고리즘 (강화학습 기반)


# ✅ Description
# K-medoids
1. K-Medoids를 사용한 클러스터링 후 수요, 공급에 맞게 노드 재배치

   (Clustering fucntion 파일을 import 하여 K-meodoids 파일 실행)
   
   ![image](https://github.com/user-attachments/assets/3c96abbc-89c6-4592-9c8d-2cef288fbe1b)


# Genetic
1. 유전 알고리즘을 사용해 거리와 수요, 공급 균형을 3:7 비율로 중요도를 부여해 클러스터링
2. 노드를 분할하여 중복 노드 허용 클러스터링 구현

   (GAcluster_model과 GAcluster_utils를 import 하여 GAcluster 실행)

   * 결과 출력 예시

     ![terminal result 3-1](https://github.com/user-attachments/assets/bdc7271f-7284-46f2-bf66-ffc534529069)

   * 알고리즘 문제점
     1. 각 클러스터별 fixed_net_demand의 열 합이 정확히 0이 되지 않음. (근사치로 구해짐)
     2. 시간이 오래 걸림

        
# Integer_programming
1. 수요, 공급 샘플 데이터 수정
2. 정수 계획법을 사용해 각 클러스터의 수요, 공급 합이 0이 되도록 구현 성공
3. 유전 알고리즘 소요 시간 : 10 ~ 20분 vs. 정수 계획법 소요 시간 : 0.1초 내외

   * 결과 출력 예시

     ![image](https://github.com/user-attachments/assets/2ade8c3b-f011-40b4-b6b0-f305e30875ad)

   * 알고리즘 문제점
     1. fixed_net_demand의 값이 커지면(세자리 이상) 해를 구하지 못함.


# Gurobi_Clustering.py
1. Gurobipy 패키지를 사용해 클러스터링 구현
2. 수요, 공급량 스케일에 관계없이 잘 작동함.
3. 패키지 개발사 웹사이트 링크 : https://www.gurobi.com/

   * 결과 출력 예시

     ![image](https://github.com/user-attachments/assets/11590e7e-ce55-400c-ab4a-6d1191bac95c)

# Gurobi_Clustering_fixed, Gurobi_gangwon, Gurobi_chuncheon
1. Gurobi_Clustering 코드가 클러스터링 시 거리를 고려하도록 수정 → 더 다양한 솔루션 탐색하는 효과
2. Gurobi_ga과
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/9608453a-e48b-4619-9815-78c5b263a298" />


# Paper
게재 예정
