> # 머신러닝 
>> ## 학습곡선
### Overfitting/Underfitting
- 과대 적합이란 모델이 훈련세트에서는 좋은 성능을 내지만 검증 세트에서는 낮은 성능을 내는 경우를 말한다.
- 과소 적합은 훈련세트와 검증 세트의 성능에는 차이가 크지 않지만 모두 낮은 성능을 내는 경우를 의미한다.

### 나타나는 이유?
- 매개 변수(파라미터)가 많고 표현력이 높은 경우
- 훈련 데이터가 너무 적은 경우

### 훈련세트의 크기와 과대/과소적합 분석
![이미지](https://velog.velcdn.com/images/prislewarz/post/6b324d40-04b2-4dad-ad39-723e41d7954f/image.png)
![이미지2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc3ijh4%2FbtqDJfx9Ahw%2F0ux2dBo5rj5trCcCu705ek%2Fimg.png)
1. 과소적합
   - 훈련 세트와 검증세트의 성능 간격은 가까우나 성능 자체가 낮다. 
   - 편향이 큰 상태다(high bias)
   - 모델이 충분히 복잡하지않아 훈련데이터에 있는 패턴을 모두 잡아내지 못한 현상을 나타낸다.

2. 과대적합
  - 훈련세트와 검증세트의 측정한 성능 간격이 크다
  - 분산이 크다(high bias)라고 표현한다.
  - 훈련세트에 충분히 다양한 패턴의 샘플이포함되지 않아 검증세트에 제대로 적응하지 못한것을 의미한다.

>> ## 정규화 선형회귀 방법
### 정규화란?
- 회귀 모형이 과도하게 최적화되는 현상을 막는 방법을 의미한다.

### 정규화 선형회귀의 종류 

### 1. Ridge 회귀
- 가중치들의 제곱합을 최소화 하는것을 추가적인 제약 조건으로한다.
- 기존 선형 모델에 규제항을 추가해 overfitting을 해결한다
- 릿지 회귀는 회귀계수를 0에 가깝게 하지만 0으로 만들지는 않는다.

$$\begin{equation} RidgeMSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^{p} w_i^2
\end{equation}$$

- $\alpha$: 사용자가 지정하는 매개변수
  - $\alpha$가 크면 규제의 효과가 커지고, $\alpha$가 작으면 규제의 효과가 작아짐
  - $\alpha$가 없으면 일반적인 선형 회귀와 같다.

### 실습 예제(보스턴 집값)

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# 시작전 모듈 import
x, y = load_boston(return_X_y = True) #return_X_y는 데이터 셋에 저장된 데이터와 타겟을 x,y에 바로 로드해줌
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = Ridge(alpha = 0.2) # 릿지모델을 만들어줌 (매개변수 0.2로 지정)
model.fit(x_train, y_train)
# 훈련 데이터와 테스트 데이터의 성능을 볼수 있음.
print(f'Train Data Score: {model.score(x_train, y_train)}')
print(f'Test Data Score: {model.score(x_test, y_test)}')
# 그래프로 만들어주기 위해 함수 지정
import matplotlib.pyplot as plt
def plot_boston_price(expected, predicted):
  plt.figure(figsize=(8,4))
  plt.scatter(expected, predicted)
  plt.plot([5, 50], [5, 50], '--r') 
  plt.xlabel('True price ($1,000s)')
  plt.ylabel('Predicted price ($1,000s)')
  plt.tight_layout()

x_test_predict = model.predict(x_test) 
plot_boston_price(y_test, x_test_predict) # 함수 실행
```
### Ridge 알파 계수를 반복문으로 만들어본 코드
```python
models = [Ridge(alpha = i)for i in np.arange(0,1,0.1)] #0부터 1까지 0.1씩 알파계수로 넣어줌
trained_models = [] #학습된 모델들이 들어올 빈 리스트 만들어줌

for model in models:
    model.fit(a_train,b_train)
    print(f'alpha: {model.alpha}',f'Train Data Score : {model.score(a_train,b_train)}',f'Test Data Score : {model.score(a_test,b_test)}')
    trained_models.append(model)
```

### 2. 라쏘 회귀 (Lasso Regression)
- Lasso(Least Absolute Shrinkage and Selection Operator) 회귀모형은 가중치의 절대값의 합을 최소화하는 것을 추가적인 제약 조건으로 한다. **참고(릿지는 가중치들의 제곱합을 최소화하는것이 제약조건)**
$$\begin{equation}
LassoMSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^{p} |w_i|
\end{equation}$$

- 릿지와 마찬가지로 매개변수인 $\alpha$값을 통해 규제 강도 조절 가능

### 실습 예제 (보스턴 집값)

```python
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# 시작전 모듈 import
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 모델 생성 및 알파값 지정
model = Lasso(alpha = 0.1) #알파값이 릿지때보다 낮게 설정됐음을 확인 일반적으로 라쏘에서는 릿지보다 더 낮게 설정한다.
# model = Lasso(alpha = 0.01)
# model = Lasso(alpha = 0.001)
model.fit(x_train, y_train)

print(f'Train Data Score: {model.score(x_train, y_train)}')
print(f'Test Data Score: {model.score(x_test, y_test)}')
x_test_predict = model.predict(x_test)
plot_boston_price(y_test, x_test_predict)
```

### 3. 신축망
- 릿지 회귀와 라쏘 회귀, 두 모델의 모든 규제를 사용하는 선형 모델. 
- 데이터 특성이 많거나 서로 상관 관계가 높은 특성이 존재 할 때 위의 두 모델보다 좋은 성능을 보여 준다.

$$\begin{equation}
ElasticMSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) + \alpha \rho \sum_{i=1}^{p} |w_i| + \alpha (1 - \rho) \sum_{i=1}^{p} w_i^2
\end{equation}$$

- $\alpha$: 규제의 강도를 조절하는 매개변수
- $\rho$: 라쏘 규제와 릿지 규제 사이의 가중치를 조절하는 매개변수

l1_ratio = 0 (L2 규제만 사용) =  Ridge

l1_ratio = 1 (L1 규제만 사용) = Lasso

0 < l1_ratio < 1 (L1 and L2 규제의 혼합사용)