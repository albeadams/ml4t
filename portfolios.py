import requests
from bs4 import BeautifulSoup

AAPL = ['AAPL']
POPULAR = ['AAPL', 'TSLA', 'AMZN', 'AMD', 'FB', 'NFLX']

def SP500():
    url = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(url.text, 'html.parser')
    links = soup.find_all('a',{'class':'external text'})
    names = []
    for link in links:
        if 'sec' not in link['href']:
            names.append(link.text)
    return names