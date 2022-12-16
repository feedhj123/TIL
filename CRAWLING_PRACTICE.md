# 연습문제 벅스뮤직 크롤링
```python
from tqdm import tqdm
sdt = input("시작입 입력 : ")     #특정기간내 차트 진입 점수를 계산해야하므로 특정기간을 만들어준다.
edt = input("종료일 입력 : ")

dates = pd.date_range(sdt,edt)  # 시작전과 끝 날짜를 지정하면 그기간을 갖는 시계열 데이터로 변환해주는 함수
dates = [i.strftime("%Y%m%d") for i in dates] #문자열로 변환하여 변수안에 넣음

df_bugs = pd.DataFrame() # 후에 나온 결과값을 넣어줄 임의의 빈 데이터 프레임 만들어 놓기

for date in tqdm(dates):
       url="https://music.bugs.co.kr/chart/track/day/total?chartdate=" + str(date)

       html=requests.get(url)
       soup = BeautifulSoup(html.text,"html.parser")
       
       bugs_day = [] #데이터 프레임으로 만들기 위해 append함수를 이용해줄 빈 리스트를 가진 변수를 하나 생성
       tbody= soup.find("tbody")
       tr_soup = tbody.find_all("tr")

       scr=101   #점수의 초기값
       for tr in tr_soup:
              rank = tr.find("div",class_="ranking").get_text().split("\n")[1]
              title = tr.find("p",class_="title").get_text().replace("\n","")
              art= tr.find("p",class_="artist").get_text().replace("\n","")
              album=tr.find("a",class_="album").get_text().replace("\n","")
              scr -= 1  # 1회 추출시 -1만큼 감소 
              bugs_day.append([date,rank,title,art,album,scr])
       
       # bugs_day
       df = pd.DataFrame(bugs_day,columns=["날짜","순위","곡명","아티스트","앨범","점수"])
       df_bugs = pd.concat([df_bugs,df]) #그날 그날의 데이터로 만든 데이터프레임을 빈 데이터프레임에 합쳐줌 for문으로 받고있으므로, 빈데이터에 합치는 작업을 지속해서 반복함


df_bugs.reset_index(drop=True,inplace=True) #인덱스를 리셋해주지 않으면 인덱스가 계속 0~99로반복된다 실제로는 하루당 100곡씩 100*n일만큼 인덱스가 형성되어야 하므로 본 함수를
#사용하여 이를 만들어준다.
#df_bugs.info()
```






















# 연습문제 네이버 무비 평점 긁어오기 
```python
from multiprocessing import Pool
import multiprocessing

df = pd.DataFrame() # 반복문이 돌아가면서 채워질 빈 데이터프레임을 미리 만들어 둠
for j in range(2):  #전체 가져올 페이지수 
        url = f"https://movie.naver.com/movie/point/af/list.naver?&page={j+1}"
        response = requests.get(url)
        table_pd = pd.read_html(response.text)
        for k in range(10):
                index = table_pd[0]['번호'][k]  # 감상평 번호
                name = table_pd[0]['감상평'][k].split(" 별점 - 총 10점 중")[0] #영화제목
                write = table_pd[0]['감상평'][k].split(" 별점 - 총 10점 중")[1][:-3].split("  ")# 감상평
                data = table_pd[0]['감상평'][k].split(" 별점 - 총 10점 중")[1][:-3].split("  ") # 평점
                num_1 = int(data[0]) #숫자만 긁어오기
                num_2 = num_1
                if num_1  >= 7 : # 점수가 7점 이상인경우 점수2에 1점이 나오게 하는 함수
                        num_2 = 1 
                else : 
                        num_2 = 0
                if len(data) == 2: # 평점만 입력하고 감상평을 아예 안적는 사람들을 걸르게 하기위해서 if문을 써준다
                        A_1 = {"번호":int(index),"제목":name,"평점":int(data[0]),"감상평":write[1],"점수1":num_1,"점수2":num_2}
                        df_1 = pd.DataFrame.from_dict([A_1]) 
                        df = pd.concat([df, df_1],ignore_index = True )# ignore_index를 안써주면 인덱스가 계속 0으로나옴
```                        