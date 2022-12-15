## BeautifulSoup
- HTML 문서에서 원하는 부분을 추출해내는 라이브러리
- 'requests'는 HTML을 텍스트 형태로 출력할 뿐 실제로 HTML 태그를 다루지는 않는다.
- BeautifulSoup라이브러리가 위의 결과를 실제 코드로 변환해준다.
  
## BeautifulSoup의 메소드
- BeautifulSoup:
  - 문자열 HTML코드를 실제 HTML 코드로 변환 해주는 함수
  - BeautifulSoup(문자열,'html.parser')의 형태를 띈다.
  - 일반적으로 import BeautifulSoup as bs등의 형태로 사용한다

- find_all()
 - HTML코드에서 원하는 부분을 모두 가져오는 함수(리스트로 반환)
 -  원하는 부분 지정할때 태그와 selector를 이용한다
 -  예시
```python
# <div id="example1">
실제HTML코드.find_all("div") # 태그 이름
실제HTML코드.find_all(id="example1") # 선택자 정보

# <div id="example1">, <span class="example2">
실제HTML코드.find_all(["div", "span"]) # 태그 이름
실제HTML코드.find_all(attrs = {"id":"example1", "class":"example2"}) # 선택자 정보
```

- find()
  - 태그 혹은 선택자를 이용해 필요한 정보의 가장 앞에있는 부분을 반환한다


## URL 패턴
- 일반적으로 url 패턴은 query="검색값" page="페이지값"의 형태를 띈다 
- 따라서 실제로 웹주소를 받아와서 크롤링을 할때에는
- f"전체주소{query}"  , f"전체주소{page}"와 같이 f스트링을 이용한 방식으로 검색값이나 페이지값을 바꿔가며 원하는 정보를 찾을 수 있다.
```python
url = f"https://search.naver.com/search.naver?query={query}&nso=&where=blog&sm=tab_opt"

for query in range(10)
f"https://search.naver.com/search.naver?query={query}&nso=&where=blog&sm=tab_opt"
다음과 같이 반복문을 같이 활용해 줄 수도 있다.
```

### header를 사용한 URL 주소 불러오기
- 일부 웹사이트의 경우, 크롤링을 통해 주소를 불러와 접근하는 것을 비정상적인 접근으로 보고 검색을 일시적으로 차단하는 경우가 있다(주소값 안보내줌 ) 이 경우 headers 정보에 user-agent를 넣어주면 된다.

 ```python
 일반형
  url(requests해올 주소) ="url"
  url_raw = requests.get(url)
  soup = BeautifulSoup(url_raw.text,"html.parser")

 header가 필요한 경우.
  headers = {'User-Agent' : '유저정보'}
  url = '접속할 사이트의 주소'
  requests.get(url, headers = headers)

  ```
![User-Agent 불러오는법](https://blog.kakaocdn.net/dn/ccyxU3/btqCrN5UPid/koUH99PxRCeUPTbiZ3Xnuk/img.png)

### 크롤링과 데이터프레임
- 크롤링으로 받아온 데이터들을 데이터 프레임으로 변환 시킬 수 있다.

예시 
```python
url = f"https://kin.naver.com/search/list.naver?query={user_input}"
html = requests.get(url)
soup = BeautifulSoup(html.text, "html.parser")




ul_soup = soup.select("#s_content > div.section > ul") # 검색을 위해 큰 조각으로 하나선택
li_soup = ul_soup[0].find_all("li")   # 필요한정보들이 구분되어 있는 묶음이 하나의 리스트로 지정될 수 있는조각 확보(이후에 반복문 사용을 위해)
kin = []
for i in li_soup:
    title = i.find("dt").text.replace("\n", " ")
    date = i.find_all("dd")[0].text
    cont = i.find_all("dd")[1].text
    kin.append({"제목": title, "날짜":date, "내용":cont})
#빈 리스트를 만들어준뒤, 수집한 데이터들을 밸류값으로 묶어서 딕셔너리 형태로 만들어준뒤에, 리스트에 추가를 해준다.
#이후에 
pd.DataFrame(kin)
이와 같이 데이터 프레임의 형태로 변환시켜줄 수 있다.
```