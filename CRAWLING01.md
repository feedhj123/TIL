# 크롤링의 종류
## 1. 정적 크롤링
- 웹이 있는 정적 데이터 수집에 이용
   - 정적 데이터란 로그인같은 사전작업 없이 접근 가능한 데이터
   - 새로고침 하지 않는 이상 변하지 않는 데이터
   - 주소를 통해 요청하고 결과 전달후 종료

## 2. 동적크롤링
- 웹에 있는 동적데이터 수집에 이용
   - 동적 데이터는 입력,클릭,로그인 같이 페이지 이동시 얻을 수 있는 데이터
   -  단계적 접근 필요해 속도 느리지만 수집대상에 한계가 거의 없음

# 1. 정적 크롤링 도구
- requests: 간편한 http요청 처리하는 라이브러리 웹서비스 연결 위해 사용
- beautifulsoup: html 태그를 처리하는 라이브러리, 웹에 있는 데이터 중 필요한 데이터만  추출하기 위해 사용
- pd.read_html: html 내의 table만 추출할 수 있는 도구

# 2. 동적 크롤링 도구
- selenium: 웹드라이버 사용해 자동화 기능 실현하는라이브러리 웹에 접속해 action을 제어한다.

# 웹페이지와 HTML
## 1.HTML 태그
- 기본형
  
  <태그>내용<태그>

- 웹페이지의 시작과 끝을 의미하는 태그

    html  html

- 문서의 제목을 의미하는

   title title

- 웹에 실제 표시되는 내용을 의미

  body body

## 2.HTML 태그 종류

- ul: unordered list.
- li: list item(목록이 되는 실직적 태그)
- a: 하피어링크를 나타내는 태그
- p: 긴 글 뭉텅이
- table: 표를 나타내는 태그
- html태그: find("태그") 첫번째 태그만 검색
   find_all("태그")  전체 태그 검색후 "list"로 반환

- 예시 (find_all_li[0].find("a")['href'])  # 태그를 찾을떄는 find, 태그의 속성을 찾을때는 indexing을해준다.

# Selector(선택자)사용법
- 선택지 따라 데이터 찾는 코드차이
- id는 # class는 '.'을 붙여준다
- class 이름에 공백이 포함된 경우 공백은 '.'으로 대체하여 작성한다 
- 예시) div class ='hello python' -> div.hello.python


# BeautifulSoup
- HTML 문서에서 원하는 부분을 추출해내는 라이브러리
- 'requests'는 HTML을 텍스트 형태로 출력할 뿐 실제로 그 태그를 다루진 않는다.
- BeautifulSoup 라이브러리는 위의 텍스트 결과를 실제 HTML 코드로 바꿔준다.
- 이를 코드로 옮겨보면
  
  ```python
  url(requests해올 주소) ="url"
  url_raw = requests.get(url)
  soup = BeautifulSoup(url_raw.text,"html.parser")
  ```
  이런식의 3중구조를 띈다.