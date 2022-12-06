# PANDAS

## TimeSeries Data
- 시간의 흐름에 따라 변화하는 Data이다.
- DatetimeIndex 또는 PeriodIndex로 구성된 데이터셋이다.

1. to_datetime
   - 기본형 : pd.to_datetime(date,format=%)

2. sample data 생성
   - ts = pd.Timestamp("2022-01-01 00:00")
   - s_1 = pd.Series(100,index=[ts])>> 이런식으로
   시간객체를 시리즈의 인덱스로 삽입할 수 있다.

## Time Series의 함수들
- date_range : 기본형 pd.date_range("시작","끝",freq="함수")
- 함수목록
  1. s:초
  2. t:분
  3. H:시간
  4. D:일
  5. B:주말이 아닌 평일
  6. W:주(일요일)
  7. W-MoN:주(월요일)
  8. M: 각 달의 마지막 날
  9. MS: 각 달의 첫날
  10. BM: 주말이 아닌 평일 중에서 각 달의 마지막 날
  11. BMS: 주말이 아닌 평일 중에서 각 달의 첫날
  12. WOM-2THU: 각 달의 두번째 목요일
  13. Q-JAN:각 분기의 첫달의 마지막 날
  14. Q-DEC: 각 분기의 마지막 달의 마지막 날

## 시계열 데이터에서의 indexing과 slicing
  - Indexing : df[" "]날짜를 통해 인덱싱이 가능하다
  - Slicing: df["날짜":"날짜"] 마찬가지로 슬라이싱 역시 가능하다
  - loc와 iloc역시 동일한 방식으로 사용가능하다.

## 시계열 데이터에서의 이동
  - shift()를 이용해 데이터 값을 이동시킬 수 있다.
  - df.shift(양수,음수) 양수면 데이터값을 뒤로 이동시키고 음수면 앞의값으로 이동시킨다.

## 시간 간격 재조정
  - resample : 시간 간격을 재조정하는 기능이다
  - df.resample("함수")<< 함수에 해당하는 값으로 시계값 데이터의 시간간격을 재조정한다.

## dt접근자
 - datetime 자료형 시리즈에는 dt접근자가 존재한다
 - datetime 자료형이 가진 몇가지 유용한 속성과 메서드를 사용한다
 - 예를 들면 datetime 속성을가진 데이터 프레임 df가있다고치자

```pyhton
# df.year>>날짜에서 연만 따로추출할 수 있다
마찬가지로 month,day,weekday를 통해 년 월 일 요일등의 정보를 추출할 수 있다.
```
 - 또한 df.starttime함수는 시계열을 문자열로 변환시켜줄 수 있다.
```pyhton
df.startime("%y년 %m월 %d일")
위와 같은 코드를 통해 시계열을 문자열로 받을 수 있다.
```