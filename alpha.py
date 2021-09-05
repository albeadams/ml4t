import requests
from bs4 import BeautifulSoup

from paper_config import ALPHA_VANTAGE_SECRET_KEY

url = requests.get('https://www.alphavantage.co/documentation/')
soup = BeautifulSoup(url.text, 'html.parser')

toc = ['time-series-data', 'fundamentals', 'fx', 'digital-currency', 'technical-indicators']

def get_toc(category='technical-indicators'):
    a = soup.find_all('a', href=True)
    for e in a:
        if e['href'] == '#' + category:
            li = e.parent.findChildren("ul", recursive=True)[0].text
            return li.split('\n')
        
        
def technical_indicator(type='SMA', 
                        symbol='AAPL', 
                        interval='daily', 
                        time_period='10', 
                        series_type='open'):
    query = 'https://www.alphavantage.co/query'
    url = f'{query}?function={type}&symbol={symbol}&interval={interval}&time_period={time_period}&series_type={series_type}&apikey=ALPHA_VANTAGE_SECRET_KEY'
    r = requests.get(url)
    data = r.json()
    return data


def time_series_values(function='TIME_SERIES_DAILY_ADJUSTED',
                        symbol='AAPL',
                       outputsize='full'):
    query = 'https://www.alphavantage.co/query'
    url = f'{query}?function={function}&symbol={symbol}&outputsize={outputsize}&apikey=ALPHA_VANTAGE_SECRET_KEY'
    r = requests.get(url)
    data = r.json()
    return data
