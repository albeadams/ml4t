import pandas as pd

from portfolios import *

class History(object):
    def __init__(self, DataStore, symbol, dates=None, value='adjusted close', indicators='all'):
        assert indicators == 'all' or (indicators != 'all' and isinstance(indicators, list)), "supply indicators as list"
        self.symbol = symbol
        self.value = value
        self.indicators = indicators
        self.DataStore = DataStore

        if not isinstance(symbol, list):
            symbol = symbol.split(',')
            symbol = [x.strip() for x in symbol]
        if len(symbol) > 1:
            print('use build_multiple_state(list) when specifying multiple symbols')
            return

        symbol = symbol[0]
        
        # get value (open, close, etc.)
        x,y = DataStore.read(symbol + '_VAL')
        values = []
        for key, val in x.items():
            for k, v in val.items():
                if value in k:
                    values.append((key, v))

        df = pd.DataFrame(values, columns = ['Date',value])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        self.prices = df.astype(float)

        # add alpha values
        symbols = []
        if indicators == 'all':
            symbols.append(DataStore.list_indicator_symbols(sym=symbol))
            symbols = [item for sublist in symbols for item in sublist]
        else:
            for ind in indicators:
                symbols.append(symbol + '_' + ind + '_daily')  # daily only for now

        for symbol in symbols:
            x = DataStore.read(symbol)
            df2 = pd.DataFrame.from_dict(x[0], orient='index').astype(float)
            df = df.merge(df2, how='inner', left_index=True, right_index=True)

        df = df.fillna(method='ffill')
        df = df.sort_index()
        df1 = df.iloc[:,0].to_frame() # note: joining of dataframes not currently used
        del df1
        
        #normalizing
        df2 = df.iloc[:, 1:]/df.iloc[0,1:]
        self.indicators = df2
        
        #not normalizing
        self.indicators = df
        
        self.indicators.rename(columns={'adjusted close': 'Adj Close'}, inplace=True)
        self.indicators = self.indicators[[col for col in self.indicators if col not in ['Adj Close']] + ['Adj Close']]
        
        lastdate = min(self.prices.index[-1], self.indicators.index[-1])
        mask = (self.prices.index <= lastdate)
        self.prices = self.prices.loc[mask]
        mask = (self.indicators.index <= lastdate)
        self.indicators = self.indicators.loc[mask]
        
        earlydate = max(self.prices.index[0], self.indicators.index[0])
        mask = (self.prices.index >= earlydate)
        self.prices = self.prices.loc[mask]
        mask = (self.indicators.index >= earlydate)
        self.indicators = self.indicators.loc[mask]
        
        if dates is not None:
            mask = (self.prices.index > dates[0]) & (self.prices.index <= dates[1])
            self.prices = self.prices.loc[mask]
            mask = (self.indicators.index > dates[0]) & (self.indicators.index <= dates[1])
            self.indicators = self.indicators.loc[mask]
        
        #print('Available to use:\n  <classname>.prices\n  <classname>.indicators')