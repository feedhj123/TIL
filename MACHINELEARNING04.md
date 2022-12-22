> # 머신러닝
>> ## 다항회귀
- 입력 데이터를 비선형 변환 후 사용
- 단, 모델 자체는 **선형 모델**
$$\begin{equation}
\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_1^2 + w_5 x_2^2
\end{equation}$$

- 차수가 높아지면 좀 더 복잡한 데이터 학습이 가능하다.

### 다항 회귀의 구현

- 사이킷런에서는 다항회귀 api를 제공하지 않는다.
- PolynomialFeatures로 피처들을 변환 후, LinerRegression을 순착 적용해야한다.
- Pipe라인을 이용해 위의 2과정을 한번에 할 수도 있다.

### 예제
### 과정 1 
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

n = 100

x = 6 * np.random.rand(n, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.rand(n, 1)

plt.scatter(x, y, s=5)
```
![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNDTfk%2FbtqF5AAFSTk%2FIB5TxaDeOKXpzZBHbvrQuK%2Fimg.png) 
- 상기 예제는 이렇게 곡선(비선형)으로 나타나므로 일반적 선형회귀로는 해결이 불가능해 다항회귀를 이용해야한다.
### 과정 2
``` python
# PolynomialFeatures를 이용해 다항식으로 데이터를 변환해준다
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False) # 기본적인 다항식의 형태를 만들어준다.
x_poly = poly_features.fit_transform(x)
#이렇게 만들어진 다항식 모델에 상기의 x를 넣어서 새로운 데이터를 생성해준다.
```

### 과정3
```python
from sklearn.linear_model import LinearRegression
# model.coef_, model.intercept_
model = LinearRegression()
model.fit(x_poly, y) # 다항회귀 모델에 그렇게 변형한 데이터와 기존 y값을 넣고 학습시킨다.
```
- 선형 회귀 모델을 만들고, 다향화한 x의 데이터와 기존 y값을
모델에 fit시켜준다.

### pipeline을 이용한 방법
```python
from sklearn.pipeline import make_pipeline   
#pipe라인 함수를 이용해 일련의 과정을 한번에 진행해준다
model_lr = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),LinearRegression())
# PolynomialFeatures와 LinearRegression이 동시에 적용된 모델이다.
model_lr.fit(x, y)

## 그렇게 생선된 모델에 x와 y를 fit시키고 그래프를 그려 이를 확인할 수 있다.
xx = np.linspace(-3, 3, 100)
y_pred = model_lr.predict(xx[:, np.newaxis])
plt.plot(xx, y_pred)
plt.scatter(x, y, s=5)
```

