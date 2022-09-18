import requests
from bs4  import BeautifulSoup
import bs4.element
import datetime
from gensim.summarization.summarizer import summarize

sid = "100"

base_url = "https://news.naver.com/main/ranking/popularDay.naver?mid=etc&sid1=111" \

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}

def news_crawling():

    # web page information
    res = requests.get(base_url, headers = headers)
    # parsing
    soup = BeautifulSoup(res.text, 'html.parser')
    lis4 = soup.find('ul', class_='rankingnews_list').find_all("li", limit=50)
    news_list=[]
    href = []
    img_src=[]

    for li in lis4:
        news_info = {
            "title" : li.a.string,
            "links" : li.a["href"],
            "upload_time" : li.span.string,
            "img_src" : li.img["src"]
        }
        news_list.append(news_info)
        href.append(news_info["links"])

    return news_list, href, img_src

def img_src(news_list):
    contents_links=[]
    img_list = []

    for news_link in news_list:
        link = news_link["links"]
        print(link)
        contents_links.append(link)
        res = requests.get(link, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        contents = soup.find("img", id = "img1")
        
        if contents["data-src"].endswith("RANKING") is not None:
            img_list.append(contents["data-src"])

    return img_list

def summary(news_list):
    contents_links=[]
    datas = []
    summary_list =[]
    for news_link in news_list:
        link = news_link["links"]
        contents_links.append(link)
        data = ""
        res = requests.get(link, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        contents = soup.find("div", id = "dic_area")
        for content in contents:
            if type(content) is bs4.element.NavigableString:
                
                data += content
                data += " "
        datas.append(data)
    for i in range(len(datas)):
        test = summarize(datas[i], word_count=20)
        summary_list.append(test)
    
    return summary_list