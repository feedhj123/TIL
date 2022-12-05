# PANDAS DAY 01

## 기초 통계 함수
1. sum()
2. mean()
3. std()
4. idmax()/idxmin()
5. cumsum()
6. describe

**기본적으로 열 단위로 작동한다(행 단위인 경우 axis=1)**

```python
df = pd.Data ([[1.4,np.nan]) , [7.1,-4.5],
              [np.nan,np.nan], [0.75,-1.3]],
              index = ['a'.'b','c','d'],
              columns = ['one','two'])
```
- df.sum() 행/열 단위 합산으로 계산하는 메서드
  -  열 단위로 계산하며 nan값이 있는 경우 그냥 무시하고 계산함
  -  다만, df.sum(skipna=False)로 하는 경우 NAN값으로 표시 가능
  -  axis=1을 붙여줄 시, 행 단위로 계산 가능
  -  df.열이름을 통해 열 하나씩도 출력 가능 

- df.mean() 행/열의 평균값을 계산하는 메서드
   - 상기와 동일

- df.std() 표준편차를 계산하는 메서드
   
   - 표준편차란 각각의 값의 평균값에 대한 편차를 말한다.


- df.idxmax()/df.idxmin()최댓값/최솟값을 갖는 **인덱스**확인

- df.cunsum(): 누적합을 계산하는 메서드

- df.describe(): 통계 정보를 요약하여 보여주는 메서드
   - 수치형이 아닌경우(ex:문자) 최댓값이나 최솟값이 아닌 count uniqe top freq등의 요약 통계를 나타내줌.
  