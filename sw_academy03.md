# 가장 빠른 문자열 타이핑
- 접근 방식
바로 전에 풀었던 문자열찾기와 유사하다고 생각했다. 
풀기 전 생각했던 점.
1. 주어진 문자열은 그 크기(len)가 어떻든 count로 세면 1씩 셀 수 있다.
2. 전체 문장길이에서 주어진문자열을 count로 센 것을 빼면 hot_key를 뺀 순수 타이핑 길이가 나온다
3. 이러한 순수 타이핑 길이에 카운트한 값을 더해주면 답이 나올 것이다.
```python
T = int(input()) 
for i in range(1,T+1):
        sentence, hot_key = input().split() # 문자열과 문장을 동시에 받아와준다.
        new_sentence = (sentence.replace(hot_key,'*')) #접근방식 2와 3을 동시에 진행해줄려고 이렇게 했다. 문장에서 문자열을 그키가 어떻든 len = 1로 받을 수 있는 *로 바꿔주면서 전체길이를 한번에 구할 수 있게끔 세팅을 해줬다.
        print(f'#{i} {len(new_sentence)}') 
```


# 빌딩 높이 찾기

- 접근 방식
풀기전 생각 했던 점.
1. 기본적인 Logic을 먼저 생각하려 애씀. n-2 , n-1 < n > n+1 , n+2 인 상황이어야 count를 할 수 있음.
2. 1을 통해서 코드를 짜기 위해서는 범위 설정이 중요했음 n(가로축 길이)이 0,1이거나 99,100인거처럼 양쪽 끝단에 있는경우 그냥 range(1,n)이런식으로는 오류가 뜰 것이기 때문
3. 처음에는 if문으로 하나하나 n-2,n-1 < n > n+1, n+2를 설정하려 했으나 시간이 너무 오래걸린다고 생각이들어서 다른 방법을 찾기로함.
4. 결국 주어진 범위에서 n이 가장 크면 되는 것이기 때문에 max()함수를 사용하여 이를 한번에 구하기로함.
5. 시야가 뚫린 범위는 결국, n - 나머지 범위중 가장 큰값이기 때문에 이를 마지막에 호출하면 된다고 생각했음.

```python
for i in range(1,11):
        N = int(input())
        height = list(map(int, input().split()))
        count = 0
        for j in range(2,N-2):
            Max_height = max([height[j-2],height[j-1],height[j+1],height[j+2]])
            if height[j] > Max_height:
                count += height[j] - Max_height
        print(f'#{i} {count}')
```