># 머신러닝 
## 머신러닝의 정의
- 구체적으로 프로그래밍하지 않아도 스스로학습하여 임무를 수행할 수 있는 능력을 컴퓨터가 갖도록 구현하는
Ai의 한 분야이다.
- AI >= MACHINELEARNING >=Deep Learning의 관계이다
*밴다이어그램 생각

## 머신러닝의 학습방법 종류

## 1. 지도학습
- 레이블된 데이터
- 직접적인 피드백
- 출력 및 미래예측

## 2. 비지도 학습
- 레이블 및 타깃 없음
- 피드백 없음
- 데이터에서 숨겨진 구조 찾기

## 3. 강화학습
- 결정 과정
- 보상 시스템
- 연속된 행동에서 학습

![예시](https://mblogthumb-phinf.pstatic.net/MjAyMDAzMjBfNjIg/MDAxNTg0NjgxMzkwNDA3.phXztVskcaZOCMWsWZEbLRUCMFajGyEbS_5umsbJ7nMg.7zJtOoA7QIlRS6upLzUmd_N8U7AF_2iQsS-t7QO7HBwg.PNG.k0sm0s1/untitled.png?type=w800)

>> ## 머신 러닝의 학습방법
  1. 지도학습
   - Classification: 데이터를 여러개의 class로 분류
   - Regrssion: 어떤 데이터의 특징을 토대로 값 예측
   - 라벨(타겟)데이터가 존재하는 학습방식임.
![예시](https://static.javatpoint.com/tutorial/machine-learning/images/regression-vs-classification-in-machine-learning.png)

 2. 비지도학습
 - clustering
 - Dimentionality Reduction(차원 축소):
저차원 표현이 고차원 원본 데이터의 의미있는 특성을 이상적으로 원래 차원에 가깝게 유지할 수 있도록 고차원 공간에서 저차원 공간으로 데이터를 변환하는것 >>>아직 이해가 잘안가는 개념이므로 우선 넘어가자
![클러스터링 예시](https://media.geeksforgeeks.org/wp-content/uploads/merge3cluster.jpg)
![차원 축소 예시](https://d3i71xaburhd42.cloudfront.net/8dc7a7af1685d6667d24f013ecc5fceeb2bcc689/7-Figure2-1.png)
 3. 강화 학습 
   - 상과 벌이라는 reward를 주면 reward를 통해 상을 최대화 하고 벌을 최소화 시키는 방식
![강화학습 예시](https://blog.kakaocdn.net/dn/efV5EL/btqCO1QgkC1/1p7K6JmbdcPsrckiHhtXKK/img.png)


>> ## 기존 프로그래밍과 머신러닝의 차이
1. 기존 프로그래밍
- 규칙과 데이터를 통해서 해답을 도출한다

2. 머신러닝
- **데이터와 해답을 통해서 규칙성을 발견한다.**
- 데이터 수집 - 모델생성 -모델 학습 - 예측의 단계를 통해 데이터의 정보를 분석하여 규칙을 발견하고 이를통해 데이터에서 유의미한 상관관계를 예측하는등의 효과를 얻을 수 있다.

![이미지](https://timetodev.co.kr/Upload/TextEditor/mlprogramming.PNG)


>> # 머신 러닝의 모델링 과정
# 머신러닝 모델링 과정

-  데이터 전처리 -> 데이터 셋 분리 -> 모델생성 및 학습 -> 예측 수행 -> 평가

 1. 데이터 전처리: 불필요한 column을 제거, 데이터를 변환, nan을 제거 또는 대치  # 정보를 가공하기 쉬운상태로 만듦

 2.  데이터 셋 분리: 학습데이터,검증데이터                                      # 정보를 분류함
 3. 모델 생성 : sklearn api -> model 객체 생성                                # 정보의 가공방식을 결정
 4.  학습: model.fit()                                                        # 정보의 가공방식을 대입하여 정보가공
 5.  예측: model.predict(data) -> 예측값                             
 6.  평가 : 실제모델이 어느정도의 성능을 갖고있는가 측정





># 사이킷 런 시작
## 사이킷 런의 특징
- 다양한 머신러닝 알고리즘을 구현한 파이썬 라이브러리
-  심플하고 일관성있는 API및 예제,유용한 온라인 문서제공
-  머신러닝 관련 알고리즘과 개발을 위한 프레임워크 및 API 제공

## scikit-learn 주요 모듈
| 모듈 | 설명 |
|------|------|
| `sklearn.datasets` | 내장된 예제 데이터 세트 |
||Data Cleasing & Feature Engineering
| `sklearn.preprocessing` | 다양한 데이터 전처리 기능 제공 (변환, 정규화, 스케일링 등) |
| `sklearn.feature_selection` | 특징(feature)를 선택할 수 있는 기능 제공 | 
| `sklearn.feature_extraction` | 특징(feature) 추출에 사용 |
||모형 성능 평가와 개선|
| `sklearn.model_selection` | 교차 검증을 위해 데이터를 학습/테스트용으로 분리, 최적 파라미터를 추출하는 API 제공 (GridSearch 등)
| `sklearn.metrics` | 분류, 회귀, 클러스터링, Pairwise에 대한 다양한 성능 측정 방법 제공 (Accuracy, Precision, Recall, ROC-AUC, RMSE 등) |
| `sklearn.pipeline` | 특징 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 묶어서 실행할 수 있는 유틸리티 제공 |
||지도학습(Supervised Learning) 알고리즘|
| `sklearn.linear_model` | 선형 회귀, 릿지(Ridge), 라쏘(Lasso), 로지스틱 회귀 등 회귀 관련 알고리즘과 SGD(Stochastic Gradient Descent) 알고리즘 제공 |
| `sklearn.svm` | 서포트 벡터 머신 알고리즘 제공 |
| `sklearn.neighbors` | 최근접 이웃 알고리즘 제공 (k-NN 등)
| `sklearn.naive_bayes` | 나이브 베이즈 알고리즘 제공 (가우시안 NB, 다항 분포 NB 등) |
| `sklearn.tree` | 의사 결정 트리 알고리즘 제공 |
| `sklearn.ensemble` | 앙상블 알고리즘 제공 (Random Forest, AdaBoost, GradientBoost 등) |
||비지도학습(Unsupervised Learning) 알고리즘|
| `sklearn.decomposition` | 차원 축소 관련 알고리즘 지원 (PCA, NMF, Truncated SVD 등)
| `sklearn.cluster` | 비지도 클러스터링 알고리즘 제공 (k-Means, 계층형 클러스터링, DBSCAN 등)



>># Liner regression
- 종속변수 y와 한개 이상의 독립 변수 X와의 선형 관계를 모델링한 것이다.
- 독립변수: 원인이 되는 열 (일반적으로 feature)
input Data라고도 부른다.
- 종속변수: 결과가 되는 열 
Target: 예측치(정답) 추구하는 목표를 의미한다

ex) 당뇨병환자의 질병 여부를 검사하는 데이터에서 환자가 당뇨에 걸렸으면 1, 안걸렸으면
0인 경우 이를 나타내는 데이터가 Target 데이터라 볼 수 있다.
 Target은 Label, Class와 동의어

- 선형 관계의 modeling은 1차로 이루어진 직선을 구하는 것
- Input와 Label 데이터의 **관계를 가장 잘 설명하는 최적의 직선**을 찾아냄으로써 독립 변수와 종속 변수 사이의 관계를 도출해 내는 과정

 ## 데이터를 활용한 머신러닝 모델링(선형 regression)시 확인 사항
1. Dataset.data를 통해 내부 값을 확인
2. Dataset.target을  통해 목적 데이터를 확인
3. Dataset.data.shape를 통해 vector형태인지 확인
4. Dataset.feature_names를 통해 특징 정보를 확인 
5. 이후에 데이터프레임으로 만들어준다.
6. df.info()를 통해 데이터타입 및 null값 확인
7. df["target"]  = dataset.target 을 통해 데이터프레임에 타겟 열을 새로 만들고 그 값을 타겟
데이터로 채워줄 수 있다.
8. 만들어진 target 데이터를 df["target"].nunique()를 통해 몇개나 되는지 새로 설정해줄수있다.


## simple linear regression
- 독립 변수가 1개인 예시
- 변수가 하나인 직선 (변수 x하나임 )
- $$f(x_i) = wx_i + b$$
<img src="https://nbviewer.jupyter.org/github/engineersCode/EngComp6_deeplearning/blob/master/images/residuals.png
" width="400" height="300" />

- Goal : 예측한 값과 실제 데이터가 가장 비슷한 직선을 찾는것
- model 이 예측한 값 : $f(x_i)$
- 실제 데이터 : $y$ 입니다.  

- Cost Function(에러로 이해가능)
    - 실제 데이터(위 그림에서 빨간 점) 과 직선 사이의 차이를 줄이는 것이 우리의 목적 
$$\text{cost function} = \frac{1}{N}\sum_{i=1}^n (y_i - f(x_i))^2$$

 최솟값을 찾으려는 이유? **cost function이 가장 낮은 최솟값을 찾으려는 이유는 정확히 말하면 에러를 최소화시키게 해주는 파라미터를 찾는 것이다. 그리고 이 과정이 머신러닝에서 학습(learning)을 하는 과정이다. 이를 통해 오류가 가장적은 모델을 만들수 있다.**

$$f(x_i) = wx_i + b$$ 
를 미분한값이 0이되는 지점이 cost function이 최솟값이 되는 지점이다. >> **이차함수 그래프에서 깊이가 가장 낮은곳을 가르킬떄가 미분한 값이 0이되는 지점**이고 costfunction이 최솟값이 되는지점이다.


>> ## (중요) 최솟값을 찾는 방법(gradient descent)
### **당장 이해가 안되더라도 반복해서 읽어보기**
- 최솟값을 찾는 이유는 위에 설명했듯 **오류가 가장적은 이상적인 모델을 만들기 위해서**이다. 
- 이를 위해 선형회귀식 y= wx + b에서 오류가 가장 적게 하는 w,b 두 parameter의 값을 찾아야한다.
- 이를 위해 **목적함수, 편미분, 학습률을 활용한다.**
 1. 목적함수: 결과를 얻기 위해 처리를 해야할 대상 이 경우에는 오류를 최소화 하고자하는 것이기때문에 목점함수는 error이다 
$$\text{cost function} = \frac{1}{N}\sum_{i=1}^n (y_i - f(x_i))^2$$ 
오차 제곱의 평균을 의미한다.

이 목적함수에서 gradient 정보(여기선 기울기)를 추출해야한다.

$$\text{cost function} = \frac{1}{N}\sum_{i=1}^n (y_i - f(x_i))^2$$

그리고 그 기울기를 추출하기 위해 편미분을 해줘야한다. 
 
 2. 편미분:

 3. 학습률: hyper parmetrer를 의미한다. 사용자가 직접 지정한다 이것을 목적함수를 편미분한 값에 곱해준다.
- w에 대해서 편미분한 값에 곱하면 w의 값을 알수 있고
- b에 대해서 편미분한 값에 곱하면 b의 값을 알 수 있다.
- 이렇게 해서 뽑아낸 w와 b의 값을 새로운 w와 b의 값으로 업데이트하여 최초의 선형회귀식에 넣어서 y값을 구한다. 그 후, 이 과정을 계속해서 반복한다.
## 위 과정을 코드로 나타낸 예시
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# 데이터 불러오기.
bike = pd.read_csv(r'C:\Users\user\Downloads\SeoulBikeData.csv', encoding='cp949')
X = np.array(bike.iloc[:,3]).reshape((-1,1))
y = np.array(bike.iloc[:,1]).reshape((-1,1))

# 랜덤으로 600개만 가져오기.
idx = [random.randint(0, len(X)) for i in range(600)]
X = X[idx]
y = y[idx]

# 초기에 w 와 b 를 설정.
# learning rate 를 0.005 로 설정.
w = 1  # 기울기 
b = 1  # 상수(편향)
alpha = 0.005

# 예측값과 실제값을 보기 위한 plotting function 을 define.
def plot_(y, y_hat):
    
    real=plt.scatter(X, y, label='y', s=5, color='red')
    pred=plt.scatter(X, y_hat, label='y_hat', s=5, color='blue')
    plt.xlabel('temperature')
    plt.ylabel('demand bike')
    plt.legend((real, pred), ('real','predict'))
    
    plt.show()

# iteration 100 번을 하며 학습을 진행함.
for i in range(100):

    y_hat = w * X + b

    error= np.abs(y_hat - y).mean()
    
    w_update = alpha * ((y_hat - y) * X).mean() #w에 대해서 편미분한값
    b_update = alpha * (y_hat - y).mean()  #b에 대해서 편미분한값
    
    w = w - w_update #새롭게 업데이트될 기울기 값
    b = b - b_update# 새롭게 업데이트될 상수값
    

    if i%10 == 0:  #10단위로 그래프로 표시 
        plot_(y, y_hat)
```