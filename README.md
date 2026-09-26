# 6th_KAMP
제 6회 k-인공지능 제조데이터 분석 경진대회

#### A방식 : raw data에 대한 임의의 라벨링이 필요하다.

raw data --> (비지도 학습 e.g. isolation forest) --> 라벨링 된 data ---> (지도 학습) --> learning

근거 : welding data set.xlsx 의 result tap을 보면, 날짜 별로 defect type과 갯수 정보를 줬기 때문에 이를 버리면 안된다.

#### B방식 : raw data에 대한 임의 라벨링은 정확도 이슈가 있다.

raw data --> (비지도 학습 e.g. isolaton forest) --> learning 

근거 : 임의 라벨링에 오류가 있을 위험을 감수하지 말고 비지도 학습에만 맡긴다.
