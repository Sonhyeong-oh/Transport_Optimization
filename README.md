# Dohwa-OFORD

![도화오포드 로고](https://github.com/user-attachments/assets/dca41e5b-5bb4-4e19-bc52-766451dbccd6)

강화학습 연산량 감소를 위한 사전 클러스터링 코드 구현

# ✅ 작업 사항
# 25.05.12 (월)
1. K-Medoids를 사용한 클러스터링 후 수요, 공급에 맞게 노드 재배치

   (Clustering fucntion 파일을 import 하여 K-meodoids 파일 실행)
   
   ![image](https://github.com/user-attachments/assets/3c96abbc-89c6-4592-9c8d-2cef288fbe1b)


# 25.05.21 (수)
1. 유전 알고리즘을 사용해 거리와 수요, 공급 균형을 3:7 비율로 중요도를 부여해 클러스터링
2. 노드를 분할하여 중복 노드 허용 클러스터링 구현

   (GAcluster_model과 GAcluster_utils를 import 하여 GAcluster 실행)

   * 텍스트 출력 결과
     
   ![terminal result](https://github.com/user-attachments/assets/1bb3e4ed-65a3-48aa-b722-5693ab6e2b22)

   ![image](https://github.com/user-attachments/assets/1ac55459-8d45-4ad0-8444-a65bb89635b5)

   * 시각화
     
   ![클러스터링 시각화](https://github.com/user-attachments/assets/f9dd5c50-ee87-4ab8-b8c5-301b0d0bd96b)


