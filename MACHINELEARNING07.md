> # 머신러닝
>> ## 서포트 벡터 머신(SVM)
## SVM
---------
- 회귀, 분류 , 이상치 탐지등에 사용되는 지도학습의 방법
- 결정경계(분류를 위한 기준선)를 정의하는 모델이다. 
- 각 지지 벡터 사이의 마진이 가장 큰 방향으로 학습이 이뤄진다.
- 지지 벡터 까지의 거리와 지지 벡터의 중요도를 기반으로 예측을 수행한다.
- ![support vector machine](https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Svm_separating_hyperplanes.png/220px-Svm_separating_hyperplanes.png)
- H3은 두 클래스의 점들을 제대로 분류하고 있지않다.
- H1가 H2는 두 클래스의 점들을 분류하는데 H2가 H1보다 더 큰 마진을 갖고 분류한다는 것을 알 수 있다.

   #파랑색이 H1 빨간색이 H2임.

### SVM을 이용한 회귀 모델과 분류 모델
  
```python
 from sklearn.svm import SVR, SVC # SVC  classification, SVR regression

model = SVR()
model.fit(X_train , y_train)
## SVR모델

model = SVC()
model.fit(X_train,y_train)
## SVC모델
 ```

### 커널기법
- 입력 데이터를 고차원 공간에 사상해서 비선형 특징을 학습할 수 있도록 확장하는 방법(아래 설명 참조)
- scikit-learn에서는 Linear, Polynomial, RBF(Radial Basis Function)등 다양한 커널 기법을 지원

![그림7](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99F2D73359EACDE930)
- SVM은 선형 SVM이든 RBF 커널 SVM이든 항상 선형으로 데이터를 분류하는 것이 기본적인 전략이다. 

- 그러나 위의 그림처럼 선형으로 분류가 어려운 경우가 생기는데 이를 해결하기 위해 데이터를 고차원 공간에서 사상해서 이를 선형으로 나눌 수 있게 만드는 것이 커널 기법이다.

![그림8](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F990CD33359E9EC961E)

- 위 그림처럼 저차원공간에서는 선형으로 나눌수 없었던 데이터를 고차원공간으로 사상하니 나눌 수 있게 되었다!.
```python
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(x_train, y_train)
# 선형 커널
print(f'Linear SVR Train Data Score: {linear_svr.score(x_train, y_train)}')
print(f'Linear SVR Test Data Score: {linear_svr.score(x_test, y_test)}')

polynomial_svr = SVR(kernel = 'poly')
polynomial_svr.fit(x_train, y_train)
# 폴리메니어 커널
print(f'Polynomial SVR  Train Data Score: {polynomial_svr.score(x_train, y_train)}')
print(f'Polynomial SVR Test Data Score: {polynomial_svr.score(x_test, y_test)}')

rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(x_train, y_train)
# rbf 커널 (가장 많이 사용하고 성능이 좋다고 알려진 커널임)
print(f'RBF SVR Train Data Score: {rbf_svr.score(x_train, y_train)}')
print(f'RBF SVR Test Data Score: {rbf_svr.score(x_test, y_test)}')
```

### 매개변수 튜닝
- SVM은 사용하는 커널에 따라 다양한 매개변수 설정 가능
- 매개변수를 변경하면서 성능변화를 관찰

 - C는 얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는지를 결정
  - 작을 수록 많이 허용하고, 클 수록 적게 허용한다.
  -  C값을 낮게 설정하면 이상치들이 있을 가능성을 크게 잡아 일반적인 결정 경계를 찾아내고, 높게 설정하면 반대로 이상치의 존재 가능성을 작게 봐서 좀 더 세심하게 결정 경계를 찾아낸다.


|파라미터|default	| 설명|
|-------|-------|-------|
|C	|1.0|	오류를 얼마나 허용할 것인지 (규제항) 클수록 하드마진, 작을수록 소프트마진에 가까움|
|kernel	|'rbf' (가우시안 커널)| 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'|
|degree	| 3| 	다항식 커널의 차수 결정 |
|gamma	|'scale'|	결정경계를 얼마나 유연하게 그릴지 결정 클수록 오버피팅 발생 가능성 높아짐|
|coef0	|0.0	|다항식 커널에 있는 상수항 r

```python 
polynomial_svc = SVC(kernel = 'poly', degree=2, C=0.1, gamma='auto')
polynomial_svc.fit(x_train, y_train)
## poly 커널의 매개변수 예시

rbf_svc = SVC(kernel = 'rbf', C=2.0, gamma='scale')
rbf_svc.fit(x_train, y_train)
##rbf 커널의 매개변수 예시

# 각 커널마다 다른종류의 매개변수를 사용할 수 있음을 알아둬야한다.
```

### SVM과 데이터 전처리
- **SVM은 입력 데이터가 정규화 되어야 좋은 성능을 보임**
- 주로 모든 특성 값을 [0, 1] 범위로 맞추는 방법을 사용
- scikit-learn의 StandardScaler 또는 MinMaxScaler를 사용해 정규화

- 정리


