import time

import alpha
from portfolios import *

def populate_popular_stocks_high_usage_alphas(DataStore, interval='daily'):
    toc = alpha.get_toc()
    high_indicators = []
    for indicator in toc:
        if 'High Usage' in indicator:
            name = indicator.replace(' High Usage', '')
            high_indicators.append(name)
    status = 1
    for sym in POPULAR:
        for indicator in high_indicators:
            status = DataStore.add_indicator(sym, indicator, interval)
            if status == -1:
                break
        if status == -1:
            break
    if status == -1:
        time.sleep(60)
        populate_popular_stocks_high_usage_alphas(DataStore, interval)
        

def populate_popular_stocks_values(DataStore):
    status = 1
    for sym in POPULAR:
        status = DataStore.add_value(sym)
        if status == -1:
            break
    
    if status == -1:
        time.sleep(60)
        populate_popular_stocks_values(DataStore)


def get_value(DataStore, sym):
    status = DataStore.add_value(sym)
    while status == -1:
        time.sleep(60)
        status = DataStore.add_value(sym)
        
def get_indicators(DataStore, sym, interval='daily'):
    toc = alpha.get_toc()
    indicators = []
    for indicator in toc:
        if 'High Usage' in indicator:
            name = indicator.replace(' High Usage', '')
            indicators.append(name)
        elif indicator:
            indicators.append(indicator)
    status = 1
    for indicator in indicators:
        status = DataStore.add_indicator(sym, indicator, interval)
        while status == -1:
            time.sleep(60)
            status = DataStore.add_indicator(sym, indicator, interval)
        
def download_data(DataStore, sym):
    """get all day for a symbol (price, indicators, ... more to come)"""
    get_value(DataStore, sym)
    get_indicators(DataStore, sym)
    
        
    