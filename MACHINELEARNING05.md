> # 머신러닝
>> ## Logistic_Regression(로지스틱 회귀)
------------------

## 로지스틱 회귀
- 연속적인 Lable data가 아닌 이산적인 Lable data의 학습에 용이하다.
- 회귀라는 단어를 사용하고 있으나, 사실상 분류 문제에 더 많이 활용함.
- Logistic regression을 진행하기 위해서는 출력 값을 0과 1의 값으로 변경해야 한다. 
- Score 를 0 과 1사이의 값으로 변경하기 위해 `Sigmoid(logistic) function` 을 사용 (분류를 해야하므로 값의 변경이 필요한 것)# 시그모이드 함수를 로지스틱 함수라고도함.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Logistic regression을 진행할 때 입력 데이터를 $x$, 실제 class 값을 $y$, 예측된 출력 값을 $\hat{y}$라고 하면 $x$는 두가지 변환을 거쳐서 $\hat{y}$가 된다. 
$$z = wx + b$$
$$\hat{y} = \sigma(z)$$

- 목적
  - $\hat{y}$가 실제 $y$와 가장 가깝게 되도록 하는 $w$와 $b$를 찾는 것 
  즉, 예측값이 실제 값과 가장 가깝게 하는 가중치와 편향을 찾는 것이다.


z = sympy.Symbol('z', real=True)

logistic = 1/(1+ sympy.exp(-z))
logistic
로지스틱 함수(시그모이드 함수)를 수식으로 나타낸 모양이다.

## Logistic loss function
### Cost Function
- Linear regression -> MSE 
- Logistic regression -> ?

- Logistic regression MSE 를 적용하면 convex(볼록)한 형태가 아니다. 

- linear regression에서의 MSE
$$\frac{1}{n} \sum_{i=1}^n (y_i - (wx_i + b))^2$$


- logistic regression에서의 MSE
$$\frac{1}{n} \sum_{i=1}^n (y_i - \sigma(wx_i + b))^2$$

### 상기 수식을 코드로 나타낸 예시
$w = 1, b=0$ , $(x, y) : (-1, 2), (-20, -1), (-5, 5)$ 일 때 cost function 

badloss = (2 - 1/(1+ sympy.exp(-z)))**2 + \
          (-1 - 1/(1+ sympy.exp(-20*z)))**2  + \
          (5 - 1/(1+ sympy.exp(-5*z)))**2

### 로지스틱 회귀에서의 cost function 최솟값
- 일반 선형회귀와 같이 경사하강법을 사용하게 되면 global cost minimum이 아닌 local cost minimum을 구하게 된다.
![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F27465D46592521CE13)

## 이를 해결하기 위한 방법?

### cost function $L$
- cross entropy function 을 사용함
- 이진분류 model에서의 cost function은 다음과 같은 함수를 사용한다. 
$$ L = -y \log(a) + (y-1)\log(1-a) $$
- 이제 실제로 차이가 클 때 $L$ 값도 커질까? 
- $y=1$인 경우 $L = -\log(a)$
- $a$ : 예측한값
- $f(a)$ : costfunction 의 값
- 정답 class 가 1인 경우 predict 값이 0에 가까워지면 costfunction의 값은 점점 커지고
- 정답 class 가 1인 경우 predict 값이 1에 가까워지면 costfunction의 값은 0에 가까워 진다. 

즉, 실제값에 대한 예측 정확도가 올라갈수록  costfunction은 0에 가까워진다.


이를 이용하여 gd방식을 이용하여 최적의 parameters를 찾아 볼 수 있으며, 이 과정은 3.00 Logistic_Regression을 참고하여보자.