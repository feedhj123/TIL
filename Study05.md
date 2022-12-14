# 자료구조의 개념과 종류

## 자료구조의 개념
- 컴퓨터 프로그래밍 언어에서 효율적인 자료(데이터)형태를 의미한다.



## 자료구조의 종류
1. 단순 자료구조
2. **선형 자료구조** 중요
   -  리스트
   -  스택
   -  큐

3. **비선형 자료구조** 중요
   -  트리
   -  그래프

4. 파일 자료구조

>선형 자료구조 - 리스트
>> 선형 리스트
### 선형 리스트란?
- 데이터가 일정한 순서로 나열된 자료구조
- 순차리스트라고도 불림
- 데이터를 입력 순서대로 입력하는 일에 적당함

### 선형 리스트의 원리
### 1. 데이터의 삽입

----
  
  (1) 끝에 삽입하는 경우:
   - 선형리스트의 끝에 빈칸을 하나 추가해준다 >>>apend(none)
   - 이후 데이터값을 추가해준다.

  (2) 중간에 삽입하는 경우:
  - 끝에삽입하는 경우와 마찬가지로 빈칸을 하나 추가해준다
  - 이후 리스트의 끝에서부터 빈칸으로 뒷칸으로 땡겨서 빈공간을 확보해준 뒤,
  그곳에 데이터를 삽입해준다.

### 2. 데이터의 삭제 

---
   
   (1) 끝에 있는 데이터 삭제: del명령어를 통해 바로 삭제해준다.
   
   (2) 중간에 있는 데이터의 삭제:
   삭제하고자하는 데이터를 빈칸(None)값으로 변환해준뒤, 데이터를 끝에서부터 한칸씩 앞으로 땡겨서 받아준다. 작업이 끝난후 리스트의 마지막 부분은 빈데이터값(none)만 남게되는데, 이를 삭제해준다.

### 3. 선형리스트의 일반적인 구현
----

(1) 선형리스트 생성 함수를 제작
```python
Data = [] #데이터값을 추가해줄 빈리스트

def add_data(dataset):
    Data_append(None) #빈데이터를 먼저 집어넣어줌
    Dlen = len(Data) #리스트의 길이를측정
    Data[Dlen-1] = dataset # 0번째부터 순차적으로 데이터를 채워줄 수 있는 함수 완성
```
(2) 선형 리스트 데이터 삽입함수
```python
def insert_data(position, friend):
    katok.append(None)
    kLen = len(katok)
    #position은 데이터를 넣을 위치 friend는 넣어야할 데이터값을 의미
    for i in range(kLen-1, position, -1): #리스트의 인덱싱은 0부터시작하므로 새로 추가하는 빈데이터값에 기존데이터의 마지막 데이터를 넣기위해 범위설정을 klen-1로한것임.
        katok[i] = katok[i-1]
        katok[i-1] = None
        
    katok[position] = friend
```

(3) 선형 리스트 삭제함수
```python
def delete_data(poistion):
    katok[poistion] = None
    kLen = len(katok)
    #삭제할 데이터가 위치한 리스트를 먼저 None값으로 바꾸고 시작함
    for i in range(poistion+1, kLen, 1): #삭제한 데이터의 바로뒤에거부터 앞으로 데이터를 채워야하므로 position+1을 범위의 시작으로 설정
        katok[i-1] = katok[i]
        katok[i] = None

    del (katok[kLen-1])
```


>선형 자료구조 - 리스트
>> 단순 연결 리스트
### 단순연결리스트란?
- 노드(데이터+링크)들이 물리적으로 떨어진 곳에 위치
- 노드의 번지도 순차적이진 않음
- 하지만 연결(링크)을 따라가면 선형 리스트의 순서와 동일함.

### 단순 연결리스트의 간단 구현

### 1. 노드의 생성과 연결 함수
---

```python
class Node() : #노드라는 데이터형을만들어주는 클래스
    def __init__(self): #>>데이터형 생성시 자동으로 실행되는 부분
        self.data = None
        self.link = None

```

### 2. 노드들을 연결시켜 연결리스트를 생성하는 함수
---

```python
def printNodes(start):
    current = start ##current는 현재 가르키는 노드 위치를 말함 맨처음이니current 또한 start에 머무름
    print(current.data, end=' ')
    while (current.link != None): ##current.link가 none값이 아닐때까지라는 얘기는 none값인 마지막 노드가 나올때까지 이하 작업을 반복하라는 의미임.
        current = current.link
        print(current.data, end=' ')
    print()
```

