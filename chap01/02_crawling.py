# 데이터를 수집하는 방법
# - Crawling : html 파일 읽어오기(수집)
# pip install BeautifulSoup4 : 설치
# pip install urlopen        : 설치

from bs4 import BeautifulSoup
from urllib.request import urlopen

# 웹 페이지를 읽고 soup을 만든다.
url = "https://www.nytimes.com/2018/03/15/world/europe/germany-food-bank-migrant-ban.html?hp&action=click&pgtype=Homepage&clickSource=image&module=photo-spot-region&region=top-news&WT.nav=top-news"

html = urlopen(url)
soup = BeautifulSoup(html.read(), "html.parser") # 반정형의 데이터를 읽어온다.

# 본문 읽어오기
paragraph_list = soup.find_all("p",{"class":"css-exrw3m evys1bk0"})

for data in paragraph_list:
    print(data.get_text())

url = "http://pythonscraping.com/pages/page1.html"

soup = BeautifulSoup(urlopen(url).read(), "html.parser")

print(soup.body) # 태그 포함해서 가져오기
print(soup.body.get_text()) # .get_text() 를 통해서 태그를 제외하고 텍스트만 가져오기.

url = "http://pythonscraping.com/pages/warandpeace.html"

soup = BeautifulSoup(urlopen(url).read(), "html.parser")

green_list = soup.find_all("span",{"class":"green"})

for text in green_list:
    print(text)




