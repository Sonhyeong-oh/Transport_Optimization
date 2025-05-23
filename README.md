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

     ![terminal result 3-1](https://github.com/user-attachments/assets/bdc7271f-7284-46f2-bf66-ffc534529069)
     ![terminal result 3-2](https://github.com/user-attachments/assets/7bffa6f7-e58a-48d8-943a-85313f56eace)


# 25.05.21 (수)
1. 수요, 공급 샘플 데이터 수정
2. 정수 계획법을 사용해 각 클러스터의 수요, 공급 합이 0이 되도록 구현 성공
3. 유전 알고리즘 소요 시간 : 10 ~ 20분 vs. 정수 계획법 소요 시간 : 0.1초 내외

   * 텍스트 출력 결과

  ![image](https://github.com/user-attachments/assets/601b464b-a00a-44bd-a3a5-cdbe040e37f8)![image](https://github.com/user-attachments/assets/65d2e82c-5eed-4818-887f-f872e0550f71)![image](https://github.com/user-attachments/assets/6ce96e85-c542-4929-bec8-606a7c4c944e)





