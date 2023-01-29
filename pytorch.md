> # PYTORCH


## PYTORCH란?
- 2017년 초, 공개된 딥러닝 프레임워크로 루아언어로 개발되었던 토치를 페이스북에서 파이썬 버전으로 내놓은 것

- 간결하고 빠른 구성으로 주목받고 있다.

## Pytorch의 동적 신경망
- 훈련을 반복할 때마다 네트워크 변경이 가능한 신경망을 의미
- 예를 들어 학습중에 은닉층 추가나 제거등 모델의 네트워크 조작이 가능하다.
- 연산그래프를 정의하는 것과 동시에 값도 초기화가 동시에 되는 **Defin by Run**방식을 사용 따라서 연산그래프와 연산을 분리해서 생각할 필요가 없어 코드이해가 쉽다.
- 이에 반해 Tensorflow는 모델을 만들어주고 값을 따로 넣어주는 **Define and Run** 방식을 사용한다.

![define by and run](./img_1/%EC%BA%A1%EC%B2%982.PNG)

## pytorch의 구성요소
1. torch: GPU를 지원하는 텐서 패키지

2. torch.autograd: 자동 미분 패키지

3. torch.nn: 신경망 구축 및 훈련 패키지(데이터 구조나 레이어등의 라이브러리)

4. torch.multiprocessing: 파이썬 멀티프로세싱 패키지

5. torch.optim: SGD를 중심으로한 파라미터 최적화 알고리즘 제공

6. torch.utils: DataLoader및 데이터 조작등 기타 유틸리티 기능을 제공

7. torch.onnx: ONNX(Open Neural Network Exchange). 서로 다른 프레임워크 간의 모델을 공유할 때 사용

## Pytorch vs tensorflow
- ### 모듈의 차이
![비교](./img_1/%EC%BA%A1%EC%B2%983.PNG)

- ### 레이어의 차이
![레이어](./img_1/%EC%BA%A1%EC%B2%984.PNG)

- ### 훈련방식의 차이
![트레인1](./img_1/%EC%BA%A1%EC%B2%985.PNG)

![트레인2](./img_1/%EC%BA%A1%EC%B2%986.PNG)

>> # PYTORCH 코드 구현

## 1.torch.Tensor와 torch.tensor

```python
- "torch.Tensor"
    - 클래스 (Class)
    - int 입력시 float로 변환
    - torch 데이터 입력시 입력 받은 데이터의 메모리 공간을 사용
    - list, numpy 데이터 입력 시 입력 받은 데이터를 복사하여
      새롭게 torch.Tensor를 만든 후 사용
- "torch.tensor"
    - 함수 (Function)
    - int 입력시 int 그대로 입력
    - 입력 받은 데이터를 새로운 메모리 공간으로 복사 후 사용

짤막한 TIP
Class : 앞글자가 대문자로 시작

Function: 앞글자가 소문자로 시작
```

- **중요한 Point**: torch.Tensor는 torch값이 들어올때는 메모리 공간을 그대로 사용하기 때문에 원본 데이터를 수정하면 수정값이 다른 작업에도 영향을 미침

- torch.tensor의 경우 애초에 데이터를 복사하여 사용하기 때문에 원본데이터를 수정해도 다른 작업에 영향을 주지않음

- torch.Tensor의 경우라고 하더라도 list나 numpy자료형의 경우 데이터를 복사하여 사용하기 때문에 다른 작업에 영향을 끼치지않음


## 2. PYTORCH의 연산
- torch.add : 더하기
- torch.sub : 빼기
- torch.mul : 곱하기
- torch.div : 나누기

```python
Quiz: ((4*2)-(1+2)) - 5 를 계산해주세요!

A = torch.Tensor([4])
B = torch.Tensor([2])
C = torch.Tensor([1])
D = torch.Tensor([2])
E = torch.Tensor([5])

# 1줄에 torch함수 하나씩만 사용하세요!
out1 = torch.mul(A,B)
out2 = torch.add(C,D)
out3 = torch.sub(out1,out2)

output = torch.sub(out3,E)

print("result = {}".format(output))
```

- torch.mm ,torch.matmul: 내적곱을 수행할 떄, 활용할 수 있다. 다만 matmul은 broadcast가 지원이 된다.

```python
# (3, 2) 크기의 X 텐서와 (2, 2) 크기의 Y 텐서를 생성한다.
X = torch.Tensor([[1, 4], 
                  [2, 5], 
                  [3, 6]])

Y = torch.Tensor([[7, 9], 
                  [8, 10]])

# 행렬의 곱셈을 한다.
print(torch.mm(X, Y)) # 브로드캐스팅을 지원하지않는다 
print()
print(X.mm(Y))

```

-  max & argmax / min & argmin: 최댓값,최솟값 크기 및 위치 구하기.

```python

# 텐서의 모든 원소중 최대값 및 최대값의 위치 구하기
print("Z max:", torch.max(Z))
print("Z argmax:", torch.argmax(Z))

# 텐서의 모든 원소중 최소값 및 최소값의 위치 구하기
print("Z min:", torch.min(Z))
print("Z argmin:", torch.argmin(Z))

# 차원 지정시 지정된 차원을 기준으로 차원이 축소되며, 최대값 및 위치 혹은 최소값 및 위치를 튜플로 반환한다.

Z_max, Z_argmax = torch.max(Z, dim=1)
Z_min, Z_argmin = torch.min(Z, dim=1)
print("Z max:\n", Z_max)
print("Z argmax:\n", Z_argmax)
print()
print("Z min:\n", Z_min)
print("Z argmin:\n", Z_argmin)

```



## 3. 텐서로의 변환 및 되돌아가기
  ```python
  # list 로부터 2x3 텐서 생성
x_list = [[1, 2, 3], [4, 5, 6]]
x = torch.Tensor(x_list)
print(x)

# numpy array 로부터 2x3 텐서 생성
x_numpy = np.array([[1, 2, 3], [4, 5, 6]])
x = torch.Tensor(x_numpy) # float형으로 출력
print(x)
print(type(x))

----- 텐서변환 ------


# .tolist()
x_back2list = x.tolist() # 같은 level(위치)에 있는 데이터끼리 묶어준다.
print(type(x_back2list))

# .numpy()
x_back2numpy = x.numpy()
print(type(x_back2numpy))

---- 다시 원래 형태로 복귀 ------

```

## 4. Pytorch GPU사용
- pytorch에서 GPU를 사용하려면 device 정보를 텐서에 **'string'**타입으로 전달해줘야 한다
- 'cuda':GPU사용
- 'cpu':CPU사용

```python
#@title
# 기본 device 정보
print("텐서 x 의 device:", x.device) # cpu

device = 'cuda'
# GPU 사용
x = x.to(device)
print("device 정보 전달 후, 텐서 x 의 device:", x.device)

device = 'cpu'
# CPU 사용
x = x.to(device)
print("device 정보 전달 후, 텐서 x 의 device:", x.device)
```


## 5. 랜덤 텐서 생성하기
- torch.manual_seed: 동일한 결과를 만들도록 seed고정

- torch.rand:[0,1]사이의 랜덤 텐서 생성

- torch.randn: 평균=0, 표준편차=1인 정규분포로부터 랜덤 텐서 생성

- torch.randint: [최저값(low),최대값(high),형태(size)] 사이에서 랜덤 정수 텐서 생성

추가적으로

- torch.zeros_like : 입력 텐서와 같은 크기,타입,디바이스로 0으로 채운 텐서 생성

- torch.ones_like: 입력 텐서와 같은 크기, 타입, 디바이스로 1로 채운 텐서 생성
```python
torch.manual_seed(777)
# 랜덤 숫자로 구성된 크기가 2x3 인 텐서 생성
# 0과 1사이의 랜덤한 숫자
print("torch.rand\n-------------")
x = torch.rand(2, 3)
print(x)
print()

# 평균=0, 표준편차=1 정규분포에서 생성
print("torch.randn\n-------------")
x = torch.randn(2, 3)
print(x)
print()

# 0과 8 사이의 정수형 랜덤한 숫자
print("torch.randint\n-------------")
x = torch.randint(low=0, high=8, size=(2, 3))
print(x)
print()

# GPU를 사용하고 크기가 x 와 같은 0으로 채워진 텐서 생성
x_zeros = torch.zeros_like(x.cuda())
print(x_zeros.device)
print(x_zeros)


Quiz: 0부터 9사이의 랜덤 정수 3 * 4크기의 행렬을 만들고, 다른 행렬은 디바이스로 1로 채워진 동일한 크기의 텐서를 생성한 후 두 행렬을 더해서 결과를 출력해주세요.

A = torch.randint(low=0, high=9, size=(3, 4))
B = torch.ones_like(A)

output =  torch.add(A,B)
print(output)
```

## 6. Tensor의 type 알아보기

```python
# 실수형 텐서
a = torch.FloatTensor(np.array([[1, 2, 3], 
                                [4, 5, 6]]))

# 정수형 텐서
b = torch.LongTensor(np.array([[1, 2, 3], 
                               [4, 5, 6]]))

# 8 bit 정수형
c = torch.ByteTensor([True, False, True, True])
---반환시 True는 1로 False는 0으로 표시---
# 불리언형 텐서
d = torch.BoolTensor([True, False, True, True])
```

## 7. 텐서의 조작 
- 텐서 역시 다른 자료형과 마찬가지로
슬라이싱과 같은 조작이 가능하다.

- data.flatten()을 통해 차원을 축소시킬 수 있다.

```python
슬라이싱 예시

# 예시 텐서
torch.manual_seed(777)
x = torch.randint(0, 10, size=(2, 3, 4))

x는 (2, 3, 4) 크기를 가지는 텐서다. 

첫번째 차원의 0번 원소, 두번째 차원의 2번 원소, 세번째 차원의 3번 원소를 선택하려면 다음과 같이 한다.

x[0,2,3]

인덱싱으로 생각하여 각 크기의 n-1까지가 최대크기라고 생각하면 이해가 쉬움

3차원 기준으로,

x[]안의값의 첫번째는 깊이 2번째는 가로(행)
3번째는 세로(열)로 이해할 수 있다.

```

- view: 텐서 크기를 알맞게 변화해야할 때 자주 사용한다. 변화되는 차원 크기의 총 곱은 원래 차원크기의 총 곱과 일치해야한다.

```python

# 크기가 (2, 3, 4) 3차원 텐서를 (2, 2, 6) 으로 변경
x_viewed1 = x.view(2,2,6)

# 텐서 시각화
print("original tensor: ", x.size())
mask = torch.ones_like(x)
draw_tensor(mask, x)

print("reshaped tensor: ", x.size())
mask = torch.ones_like(x_viewed1)
draw_tensor(mask, x_viewed1)


# "-1"을 사용하면 나머지 차원을 알아서 계산 해준다. 단, 2곳 이상 동시사용은 불가능

# 크기가 (2, 3, 4) 3차원 텐서를 (2, 1, 12) 으로 변경
x_viewed2 = x.view(-1,1,12)
# 1,12 고정되면 -1이 자동적으로 설정

# 텐서 시각화
print("original tensor: ", x.size())
mask = torch.ones_like(x)
draw_tensor(mask, x)

print("reshaped tensor: ", x_viewed2.size())
mask = torch.ones_like(x_viewed2)
draw_tensor(mask, x_viewed2)

```
- 텐서의 인덱싱 역시 가능하다(torch.index_select 함수 사용 가능)

```python
A = torch.Tensor([[1, 2],
                  [3, 4]])

# [1, 3]만 출력해봅시다.

# torch.index_select 함수를 써서 해보세요!
output = torch.index_select(A, 1, torch.tensor([0])) #입력텐서 ,axis , index
output = output.view(-1)
output

# 파이썬 리스트 인덱싱과 비슷한 방법으로 해보세요!
output = A[:,0]
output
```


- Permute : 차원의 위치를 바꿀때, 주로 사용(텐서 전체 모양 바꿀때 유용)

```python
x = torch.zeros(2,3,4)

# (2, 3, 4) 크기 텐서의 차원 크기를 (4, 3, 2)로 바꾼다. rank, shape
x_permuted = x.permute(2,1,0)

인덱스 순서로 재정렬 했다고 이해하면 편하다

x의 인덱스 위치 2는 :4
x의 인덱스 위치 1은: 3
x의 인덱스 위치 0은 : 2
```

- transpose: permute의 특별한 케이스로 주로 2개 차원을 교환하여 바꿀 떄, 사용한다.

```python
# (2, 3, 4) 크기 텐서의 첫번째 차원과 두번째 차원이 바뀐다.
x_transposed = x.transpose(1,0)

# 텐서 시각화
print("original tensor: ", x.size())
mask = torch.ones_like(x)
draw_tensor(mask, x)

print("reshaped tensor", x_transposed.size())
mask = torch.ones_like(x_transposed)
draw_tensor(mask, x_transposed)
```

- Squeeze and unsqueeze
  - squeeze: 텐서의 크기가 1인 차원을 지운다. 차원을 특정하면(숫자를 이용해서)그 차원의 크기가 1이면 지우고 아니면 그냥 둔다.

  - unsqueeze: 해당하는 숫자 차원에 크기 1인 차원을 늘린다.
  np.newaxis와 비슷한 효과

```python
# 크기가 (2, 1, 3, 4, 1) 인 5차원 텐서를 생성한다
x = torch.rand((2, 1, 3, 4, 1))
print(x.size())

# 모든 차원에서 크기가 1인 차원을 squeeze 한다.
print(x.squeeze().size())  # 크기확인
torch.Size([2, 3, 4])

# 두번째 차원(크기 = 1)을 squeeze 한다.
print(x.squeeze(4).size())  # 크기확인
torch.Size([2, 1, 3, 4])
# 4번째 차원에 크기를 1 추가, 6차원 텐서가 된다.
print(x.unsqueeze(4).size())  # 크기확인
torch.Size([2, 1, 3, 4, 1, 1])
```

- cat과 stack
    - cat의 경우 텐서를 합친다는 느낌이 강하다. 지정한 차원방향의 두텐서가 같아야한다.
     
```python
  torch.manual_seed(777)
# 크기가 (2, 3) 인 A, B 텐서를 만든다
A = torch.rand((2, 3))
B = torch.rand((2, 3))

# 첫번째 차원을 기준으로 텐서를 concatenate 한다.
AB_cated = torch.cat([A, B], dim=0) # dim = 0 가로기준 dim = 1 세로 기준
print(A)
print(B)
print(AB_cated)

```

   - stack: 텐서들을 쌓는다는 느낌이 강하다. 각 리스트 안에 있는 지정된 차원을 unsqueeze한 다음, cat을 사용하는 것과 같다.

```python
# 첫번째 차원을 기준으로 텐서를 stack 한다.
AB_stacked = torch.stack([A, B], dim=0)
print("torch.stack([A, B], dim=0)\n")
print(AB_stacked)
print("----"*10)
print("torch.cat([A.unsqueeze(0), B.unsqueeze(0)], dim=0)\n")
# 각 텐서를 첫번째 차원 기준으로 unsqueeze 후, cat 한것과 같은 결과
AB_unsqueeze_cat = torch.cat([A.unsqueeze(0), B.unsqueeze(0)], dim=0)
print(AB_unsqueeze_cat)

```





>> ## PYTORCH 모듈 

### 1. torch.nn
- torch.nn 공식문서 읽기[공식](https://pytorch.org/docs/stable/nn.html)

### 1-1 nn.linear
- y= wx + b의 linear transformation을 구현해놓은 것

활용예제(텐서 크기 변환)
```python
import torch
from torch import nn

---모듈 import ---

X = torch.Tensor([[1, 2],
                  [3, 4]])

# TODO : tensor X의 크기는 (2, 2)입니다
#        nn.Linear를 사용하여서 (2, 5)로 크기를 바꾸고 이 크기를 출력하세요!

linear = nn.Linear(2,5)
output = linear(X)
output.size()

```
### 1-2 nn.identity
- 입출력값이 동일한 텐서를 출력함 

```python
import torch
from torch import nn

X = torch.Tensor([[1, 2],
                  [3, 4]])

# TODO : nn.Identity를 생성해 X를 입력시킨 후 나온 출력값이 X와 동일한지 확인해보세요!
identity = nn.Identity()
output = identity(X)
output
```
### 1-3. nn.Module 클래스
- 커스텀 모델 제작을 위한 클래스

- pythorch의 다양한 기능들을 조합하여 모델을 만들 수 있도록 이런 일련의 기능들을 한 곳에 모아 하나의 모데롤 추상화할 수 있게 도와준다.

- nn.module 자체는 빈 상자로 이해할 수 있으며 어떠한 것을 채워놓느냐에 따라 역할이 달라질 수 있다. 그 예시는 다음과 같다. 

  
  - `nn.Module`이라는 상자에 `기능`들을 가득 모아놓은 경우 `basic building block`
  - `nn.Module`이라는 상자에 `basic building block`인 `nn.Module`들을 가득 모아놓은 경우 `딥러닝 모델`
  - `nn.Module`이라는 상자에 `딥러닝 모델`인 `nn.Module`들을 가득 모아놓은 경우 `더욱 큰 딥러닝 모델`

### nn.module 모델 제작 예시
- 더하기 연산모델
```python
import torch
from torch import nn

# TODO : Add 모델을 완성하세요!
class Add(nn.Module):
    def __init__(self):
        # TODO : init 과정에서 반드시 super 관련 코드가 들어가야함
        super().__init__()

    def forward(self, x1, x2):
        # TODO : torch.add 함수를 이용해 더하기 연산 구현
        output = torch.add(x1, x2)

        return output



x1 = torch.tensor([1])
x2 = torch.tensor([2])

add = Add() # 클래스 불러오기 
output = add(x1, x2)

output #3
```
Q.어째서 사용자 지정 클래스 작성시 super 관련 코드가 들어가야하나요? 

A: python 환경에서  상위 클래스 생성자 혹은 초기화자는 자동으로 호출 되지 않습니다. 
따라서 nn.module class 자체가 초기화 되도록 super호출이 필요합니다.

python 3을 사용하는 경우, super()호출에 인자가 따로 필요하지 않고 단순히 super().__init__()으로 족합니다.

- torch.sequential: 모듈들을 하나로 묶어 순차적으로 호출하고 싶을 때 사용
   - 묶어놓은 모듈을 차례대로 수행하기 때문에 실행순서가 정해져있는 기능들을 하나로 묶어놓기가 좋다.
  
```python
import torch
from torch import nn

# TODO : 다음의 모듈(Module)을 읽고 이해해보세요!
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value

# TODO : 위에 모듈(Module)과 nn.Sequential를 이용해서
#        입력값 x가 주어지면 다음의 연산을 처리하는 모델을 만들어보세요!
#        y = x + 3 + 2 + 5
calculator = nn.Sequential(Add(3),
                           Add(2),
                           Add(5))


# 아래 코드는 수정하실 필요가 없습니다!
x = torch.tensor([1])

output = calculator(x)

output # 11
```

- nn.modulelist(): python의 리스트처럼 모듈들을 모아두고 그때그때 원하는 것만 indexing해서 사용하고 싶은 경우 이것을 사용할 수 있다.

```python 
import torch
from torch import nn

# TODO : 다음의 모듈(Module)을 읽고 이해해보세요!
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


# TODO : Calculator 모델을 완성하세요!
class Calculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_list = nn.ModuleList([Add(2), Add(3), Add(5)])

    def forward(self, x):
        # TODO : self.add_list에 담긴 모듈들을 이용하여서
        #        y = ((x + 3) + 2) + 5 의 연산을 구현하세요!

        x = self.add_list[1](x)  # 위에서 modulelist에 담긴 모듈add를 인덱싱으로 불러와서 사용하고 있다.
        x = self.add_list[0](x)
        x = self.add_list[2](x)
        
        return x


# 아래 코드는 수정하실 필요가 없습니다!
x = torch.tensor([1])

calculator = Calculator()
output = calculator(x)

output # 11
```

- torch.nn.ModuleDict
  - 파이썬의 dict처럼 특정 모듈을 key값을 이용해 보관해놓을 수 있다.

```python
import torch
from torch import nn

# TODO : 다음의 모듈(Module)을 읽고 이해해보세요!
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


# TODO : Calculator 모델을 완성하세요!
class Calculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_dict = nn.ModuleDict({'add2': Add(2),
                                       'add3': Add(3),
                                       'add5': Add(5)})

    def forward(self, x):
        # TODO : self.add_dict에 담긴 모듈들을 이용하여서
        #        y = ((x + 3) + 2) + 5 의 연산을 구현하세요!

        x = self.add_dict['add3'](x)
        x = self.add_dict['add2'](x)
        x = self.add_dict['add5'](x)
        
        return x


# 아래 코드는 수정하실 필요가 없습니다!
x = torch.tensor([1])

calculator = Calculator()
output = calculator(x)

output # 11
```

- torch.parameter 구현
```python
import torch
from torch import nn
from torch.nn.parameter import Parameter


# TODO : Linear 모델을 완성하세요!
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # TODO : W, b parameter를 생성하세요! 모두 1로 초기화해주세요!
        self.W = Parameter(torch.ones((out_features, in_features)))
        self.b = Parameter(torch.ones(out_features))

    def forward(self, x):
        output = torch.addmm(self.b, x, self.W.T) # 곱셈 + 덧셈 동시에 수행

        return output


# 아래 코드는 수정하실 필요가 없습니다!
x = torch.Tensor([[1, 2],
                  [3, 4]])

linear = Linear(2, 3)
output = linear(x)


output 
#output == torch.Tensor([[4, 4, 4],
                     # [8, 8, 8]])):
```
- buffer? : 일반적인 tensor와 다르게 값이 업데이트 되지 않는다해도 저장하고 싶은 Tensor가 있을때, buffer에 등록하면 모델을 저장할 때, 해당 tensor들도 같이 저장할 수 있다.

```python
import torch
from torch import nn
from torch.nn.parameter import Parameter


# TODO : Model 모델을 완성하세요!
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.parameter = Parameter(torch.Tensor([7]))
        self.tensor = torch.Tensor([7])

        # TODO : torch.Tensor([7])를 buffer이라는 이름으로 buffer에 등록해보세요!
        self.register_buffer('buffer', torch.Tensor([7]), persistent=True)



# 아래 코드는 수정하실 필요가 없습니다!
model = Model()

try:
    buffer = model.get_buffer('buffer')
    if buffer == 7:
        print("🎉🎉🎉 성공!!! 🎉🎉🎉\n")
        print("🎉 이제 buffer에 등록된 tensor는 모델이 저장될 때 같이 저장될거예요! 🎉")
        print(model.state_dict())
    else:
        print("다시 도전해봐요!")
except:
    print("다시 도전해봐요!"
```

### Tensor vs Parameter vs Buffer

- "Tensor"
    - ❌ gradient 계산
    - ❌ 값 업데이트
    - ❌ 모델 저장시 값 저장
- "Parameter"
    - ✅ gradient 계산
    - ✅ 값 업데이트
    - ✅ 모델 저장시 값 저장
- "Buffer"
    - ❌ gradient 계산
    - ❌ 값 업데이트
    - ✅ 모델 저장시 값 저장


> # 부가학습

## 다차원 배열?
|   이름 | 차원  | 표기  |
|---|---|---|
| 스칼라  | 0  | 1  |
|  벡터 |   1|  [1,2,3] |
| 행렬  |  2 |  [[1,2],[3,4]] |
|  텐서 |  임의 | [[.....[1,2],[3,4]].....]  |

![참고이미지](./img_1/%EC%BA%A1%EC%B2%98.PNG)


## iter tools(cartesian prod_)
- 주어진 행렬 혹은 리스트의 모드 경우의 수를 출력한다.

```python
import itertools
a = [1, 2]
b = [4, 5]
list(itertools.product(a,b))


import torch
tensor_a = torch.tensor(a)
tensor_b = torch.tensor(b)
torch.cartesian_prod(tensor_a, tensor_b) # 모든 경우의 수를 다 출력

```

## Torch autograd(미분)

- 예시1
$$
y = w^2 \\ 
z = 10*y + 50 \\
z = 10*w^2 + 50 
$$

```python
w = torch.tensor(2.0, requires_grad = True) # True이면 미분을 하겠다는 말임.
y = w ** 2
z = 10 * y + 50
z.backward() #역전파
w.grad 

```

- 예시2
$$ Q = 3a^3 - b^2  $$
```python

a = torch.tensor([2., 3.], requires_grad = True) # 미분을 할지 안할지
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3* a **3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

a.grad   

# a에대한 편미분을 실시하면 다른 변수(이 식에서는 b)는 상수 취급한다 따라서 남게되는 결과는 9a^2 이므로 
36(a=2),81(a=3)이 된다.
```