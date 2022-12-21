> # 머신러닝 

>> ## 선형회귀 모델의 오류측정 방법
### 1. MAE(Mean Absolute Error)
- 실제 값과 예측 값의 차이를 절대값으로 변환해 평균화함
- MAE는 에러에 절대값을 취하기 떄문에 **에러의 크기 그대로 반영된다.** 그러므로 예측 결과물의 에러가 10이 나온것이 5로 나온것보다 2배가 나쁜 도메인에 쓰기 적합한 산식이다.
- **즉, 에러에 따른 손실이 선형적으로 올라갈 때 적합하다.**

```python
from sklearn.metrics import mean_absolute_error
mean_absolte_error(y_test,y_pred)
```
![MAE](https://blog.kakaocdn.net/dn/bGEngr/btqPWZxFSQR/kMkdAx0exHjw2HA8Eezbkk/img.jpg)

### 2. MSE(Mean Squared Error)
- 실제 값과 예측 값의 차이를 제곱해서 평균화
- 예측값과 실제값 차이의 **면적의 합**이다
- **제곱해서 평균을 구하는 방식이기 때문에 오류 편차가 클수록
수치가 많이 늘어나게 된다.**
```python
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test, y_pred)
```
![MSE](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F7iz5k%2FbtqPQt7tjwK%2FwW4Xugkr8jYlAJ33o9WxhK%2Fimg.jpg)

### 3. RMSE(Root Mean Squared Error)
- 2.의 방식은 실제 오류 평균보다 더커지는 특성탓에 MSE에 루트를 씌운 RMSE가 활용된다.
- 에러에 제곱을 하기 때문에 에러가 크면 클수록 그에 따른 가중치가 높이 반영된다.
- **에러에 따른 손실이 기하 급수적으로 올라가는 상황에 적합하다.**
```python
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(y_test, y_pred) 
np.sqrt(MSE)
```
### 4. MSLE(Mean Squared Log Error)
- MSE에 로그를 적용해준 지표이다.
```python
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y_test, y_pred)
```

### 5. MAPE(Mean Absolute Percentage Error)
- MAE를 퍼센트로 변환한 것
- 모델에 대한 편향이 존재한다.
```python
def MAPE(y_test, y_pred):
	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 
    
MAPE(y_test, y_pred)
```
![MAPE](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbT9d4i%2FbtqPTDBKfSs%2FNJYQp4VNSiPKqmzBz2Bds0%2Fimg.jpg)

### 6. MPE(Mean Percentage Error)
- MAPE에서 절대값을 제외한 지표이다.
- 모델이 underperformance(+) 인지 overperformance(-) 인지 판단

```python
def MAE(y_test, y_pred): 
	return np.mean((y_test - y_pred) / y_test) * 100)
    
MAE(y_test, y_pred)
```
![MPE](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCmKgn%2FbtqPYSLBLub%2F6oSuG6bvsuxnkwGuCrFMm0%2Fimg.jpg)

>> # 교차검증
## 1. Test vs Training
- 모델을 Test셋과 Training셋으로 분리하여 sample에 얼마나 일반화 될지 알 수 있다
- 보통데이터(1만개) 80%는 Traning set , 20%는 Test set으로 떼어 놓는다.
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(DATA이름, DATA_Target, test_size=0.2, random_state =100)
```
- 1) random_state: Random seed 번호, 번호를 지정하면 동일한 데이터 세트로 분리해 준다. 
따로설정 안해주면 실행할때마다 랜덤으로 데이터를 분리함 번호는 아무거나 지정해줘도 상관이없다.
- 2) test_size: 전체 데이터 셋에서 테스트 데이터 크기를 얼마나 샘플링 할것인가
- 3) train_size: test_size의 남은 값 
- shuffle: 데이터 분리전 섞을지 결정 (difault값?)

## 2. Validation set(검증 셋)의 필요성
- 모델 검정을 위해 train ,Test set만 이용시 사실상 
Test set이 Validation set이 된다.
이 경우, Test set을 이용해 모델 성능을 확인하고 파라미터를 수정하였기 때문에, Test set에서만 잘 동작하는
모델이 완성된다. 


- 검증 세트를 통해 모델을 선정하는 과정
   1. Training Set으로 Model 학습
   2. 학습된 Model을 Validation set으로 평가
   3. Validation Set로 가한 결과에 따라 모델 조정하고 다시 학습
   4. 가장 우수한 결과를 보이는 모델을 선택
   5. 그 모델을 이용해 Test set으로 평가

---최종적으로 test set으로 평가를 해주는것이다.---
- **위 과정을 통해 Overfitting을 방지할 수 있다.**
Validation Set의 결과와 Training Set의 결과와 차이가 벌어지면 Overfitting이다.

## overfitting이란? 
표본내 성능은 좋으면서 표본외 성능이 상대적으로 많이 떨어지는 경우를 말한다. 학습에 쓰였던 표본 데이터에 대해서는 잘 종속 변수의 값을 잘 추정하지만 새로운 데이터를 주었을 때 전혀 예측하지 못하기 때문에 예측 목적으로는 쓸모없는 모형이 된다.
![overfitting](https://miro.medium.com/max/1100/0*H377j9pbSHLQhkNd.webp)

## 교차 검증방법(k-fold방식)
### 1. 단순 교차검증 cross_val_score
![cross_val_score](https://t1.daumcdn.net/cfile/tistory/99D179345C666A2D20)
- 데이터를 여러 개의 부분집합으로 분할한 후, 각 분할마다 하나의 폴드를 테스트용으로 사용하고 나머지 폴드는 훈령용으로 사용한다.
- 이 과정을 반복하여 각 분할마다 정확도를 측정한다.
- 사이킷 런에서는 단순 교차검증을 위한 함수를 제공한다
``` python
cross_val_score(model,X,y,scoring=None,cv=None)
- model:회귀 분석 모형
- X: 독립 변수 데이터
- y: 종속 변수 데이터
- scoring: 성능 검증에 사용할 함수 이름
- cv: 교차검증 생성기 객체 또는 숫자(폴드 갯수 의미)
#None이면 기본적으로 3 따로 숫자를 입력하면 그 숫자만큼의 폴드를 생성한다.
score = cross_val_score()

#최종적으로 평균을 내어 정확도를 간단히 한다.
print('교차검증 평균 : %.3f' %(score.mean()))
```

### 2. 계층별 k-겹 교차검증 
- 데이터가 편향(한 곳에 몰렸을 경우)단순 교차 검증으로는 성능 평가가 잘 이뤄지지 않을 수 있다.
- 이럴 땐, stratified k-fold cross-validation을 사용한다.
  ```python
  from sklearn.model_selection import StaratifiedkFold
  skf = StratifiedkFold(n_splits=10, shuffle=True,random_state= 0)
  score = cross_val_score(logreg,data,target,cv=skf)
  print(score.mean)
  - StaratifiedkFold(n_splits,shuffle,random_state)
  - n_splits은 몇 개로 분할할지를 정하는 매개변수
  - shuffle은 기본값False 대신 True를 넣으면 Fold나누기전 무작위로 섞는다
  - corss_val_score함수의 cv매개변수에 넣으면 된다.



참고
- 일반적으로 회귀에는 k-겹 교차검증을, 분류에는 StratifiedlFold를 사용한다.
- 또한 cross_val_score함수에는 KFold의 매개변수를 제어할 수 없으므로 따로 KFold객체를 만들고 매개변수를 조정하여 cross_val_score의 cv매개변수에 넣어야한다.

### 3.임의분할 교차검증
mglearn.plots.plot_shuffle_split()

- 임의분할 교차검증은 train set과 test set의 크기를 유연하게 조절해야 할 때 유용
- train_size와 test_size에 정수를 입력하면 해당 수만큼 데이터포인트의 개수가 정해지며, 만일 실수를 입력하면 비율이 정해진다.
```python
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(train_size=0.5,test_size=0.5 , random_state = 0 ,n_splits = 8)
score = cross_val_score(logreg,data,target,cv=shuffle_split)
```




