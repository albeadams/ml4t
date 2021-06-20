import time

import alpha
from portfolios import *

def populate_popular_stocks_high_usage_alphas(DataStore=None, interval='daily'):
    if DataStore == None:
        print('supply DataStore')
        return
    toc = alpha.get_toc()
    high_indicators = []
    for indicator in toc:
        if 'High Usage' in indicator:
            name = indicator.replace(' High Usage', '')
            high_indicators.append(name)
    status = 1
    for stock in POPULAR:
        for indicator in high_indicators:
            status = DataStore.add_indicator(stock, indicator, interval)
            if status == -1:
                break
        if status == -1:
            break
    if status == -1:
        time.sleep(60)
        populate_popular_stocks_high_usage_alphas(DataStore, interval)
        

def populate_popular_stocks_values(DataStore=None):
    if DataStore == None:
        print('supply DataStore')
        return
    
    status = 1
    for stock in POPULAR:
        status = DataStore.add_value(stock)
        if status == -1:
            break
    
    if status == -1:
        time.sleep(60)
        populate_popular_stocks_values(DataStore)