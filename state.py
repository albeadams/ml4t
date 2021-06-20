import pandas as pd

from portfolios import *

class State(object):
    def __init__(self, DataStore=None, portfolio=None, value='adjusted close', indicators='all'):
        if DataStore == None:
            print('supply a DataStore')
            return
        if portfolio == None:
            print('specify a portfolio')
            return
        if indicators != 'all' and not isinstance(indicators, list):
            print('supply indicators as a list')
            return
        self.portfolio = portfolio
        self.value = value
        self.indicators = indicators
        self.DataStore = DataStore

        if not isinstance(portfolio, list):
            portfolio = portfolio.split(',')
            portfolio = [x.strip() for x in portfolio]
        if len(portfolio) > 1:
            print('use build_multiple_state(list) when specifying multiple symbols')
            return

        portfolio = portfolio[0]
        
        # get value (open, close, etc.)
        x,y = DataStore.read(portfolio + '_VAL')
        values = []
        for key, val in x.items():
            for k, v in val.items():
                if value in k:
                    values.append((key, v))

        df = pd.DataFrame(values, columns = ['Date',value])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # add alpha values
        symbols = []
        if indicators == 'all':
            symbols.append(DataStore.list_indicator_symbols(sym=portfolio))
            symbols = [item for sublist in symbols for item in sublist]
        else:
            for ind in indicators:
                symbols.append(portfolio + '_' + ind + '_daily')  # daily only for now

        for symbol in symbols:
            x = DataStore.read(symbol)
            df2 = pd.DataFrame.from_dict(x[0], orient='index')
            df = df.merge(df2, how='inner', left_index=True, right_index=True)
        self.df = df.fillna(method='ffill')
    
    def build_multiple_state(self,portfolio=AAPL):
        print('ToDo - add value')
        return
        # Pass in multiple as a list of strings ['AAPL', 'TSLA'] 
        # or comma separated string: 'AAPL, TSLA, AMZN'
        if portfolio == None:
            print('specify a portfolio')
            return
        if not isinstance(portfolio, list):
            portfolio = portfolio.split(',')
            portfolio = [x.strip() for x in portfolio]
        stocks = []
        for eachSymbol in portfolio:
            symbols = []
            symbols.append(DataStore.list_indicator_symbols(sym=eachSymbol))
            symbols = [item for sublist in symbols for item in sublist]
            df = pd.DataFrame()
            for symbol in symbols:
                x = DataStore.read(symbol)
                df2 = pd.DataFrame.from_dict(x[0], orient='index')
                df = df.merge(df2, how='outer', left_index=True, right_index=True)
            df = df.fillna(method='ffill')
            stocks.append(df)
        return stocks