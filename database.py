import json
from arctic import Arctic
from pymongo import MongoClient
import pandas as pd

import alpha, portfolios

class Mongo:
    def __init__(self):
        self.client = MongoClient('localhost')
        self.db = self.client.admin

    def server_status(self):
        serverStatusResult = self.db.command("serverStatus")
        return serverStatusResult

    def get_databases(self):
        return self.client.list_database_names()

    def get_collections(self):
        d = dict((db, [collection for collection in self.client[db].list_collection_names()])
                 for db in self.client.list_database_names())
        print(json.dumps(d))
        
        
class DataStore:
    def __init__(self, name='NASDAQ'):
        self.store = Arctic('localhost')
        self.exceeded_limit = 'Our standard API call frequency is 5 calls per minute and 500 calls per day'
        self.nodata = []
        self.library = self.store[name]

    def create_library(self, library_name=None):
        if library_name is None:
            print('supply a library name')
            return
        self.store.initialize_library(library_name)
        
    def write_indicator(self, name=None, df=None, metadata=None):
        if name is None or df is None:
            print('supply name and/or dataframe')
            return
        final_md = {}
        for key, value in metadata.items():
            if '.' not in key:
                final_md[key] = value
        self.library.write(name, df, final_md)
        
    def write_value(self, name=None, df=None, metadata=None):
        if name is None or df is None:
            print('supply name and/or dataframe')
            return
        final_md = {}
        for key, value in metadata.items():
            k = key.replace('.', ':')
            final_md[k] = value
        self.library.write(name, df, final_md)
        
    def delete(self, name=None):
        # function not working - delete is not deleting library
        if name is None:
            print('supply a name to delete')
        self.library.delete(name)
        print(f'{name} deleted')
        
    def list(self):
        return self.store.list_libraries()
    
    def read(self, symbol=None):
        if symbol is None:
            print('supply a library symbol')
            return
        item = self.library.read(symbol)
        return item.data, item.metadata
    
    def add_indicator(self, stock, indicator, interval):
        combo = stock + '_' + indicator + '_' + interval
        instore = self.get_symbols()
        if combo not in instore and combo not in self.nodata:
            data = alpha.technical_indicator(type=indicator, symbol=stock, interval=interval)
            for key, value in data.items():
                if key == 'Note':
                    print('exceeded limit')
                    return -1
            if data:
                ta_name =''
                for key, value in data.items():
                    ta_name = key
                try:
                    md = data['Meta Data']
                except:
                    md = ''
                data = data[ta_name]
                self.write_indicator(name=combo, df=data, metadata=md)
                print(f'added {combo}')
            else:
                print(f'{combo} has no data')
                self.no_data(combo)
        return 1
    
    def add_value(self, stock):
        combo = stock + '_VAL'
        instore = self.get_symbols()
        if combo not in instore and combo not in self.nodata:
            data = alpha.time_series_values(symbol=stock)
            for key, value in data.items():
                if key == 'Note':
                    print('exceeded limit')
                    return -1
            if data:
                ta_name =''
                for key, value in data.items():
                    ta_name = key
                try:
                    md = data['Meta Data']
                except:
                    md = ''
                data = data[ta_name]
                self.write_value(name=combo, df=data, metadata=md)
                print(f'added {combo}')
            else:
                print(f'{combo} has no data')
                self.no_data(combo)
        return 1
                
    def update_indicator_symbol(self, stock, indicator, interval):
        self.delete(stock + '_' + indicator + '_' + interval)
        status = self.add(stock, indicator, interval)
        if status == -1:
            time.sleep(60)
            self.update_indicator_symbol(stock, indicator, interval)

    def update_indicator(self, sym='all', indicator='all', interval='daily'):
        stores = self.get_symbols()
        updates = 0
        for store in stores:
            intr = store[store.find('_', store.find('_')+1)+1:]
            ind = store[store.find('_')+1:store.find('_', store.find('_')+1)]
            s = store[:store.find('_')]
            if sym == 'all' and indicator == 'all':
                if intr == interval:
                    updates += 1
            elif sym == 'all':
                if ind == indicator and intr == interval:
                    updates += 1
            elif indicator == 'all':
                if s == sym and intr == interval:
                    updates += 1
            elif s == sym and intr == interval and ind == indicator:
                updates += 1

        print(f'Approximate time to update {updates} symbols = {(updates/5)-1} minutes')

        for store in stores:
            intr = store[store.find('_', store.find('_')+1)+1:]
            ind = store[store.find('_')+1:store.find('_', store.find('_')+1)]
            s = store[:store.find('_')]
            if sym == 'all' and indicator == 'all':
                if intr == interval:
                    self.update_indicator_symbol(s, ind, intr)
            elif sym == 'all':
                if ind == indicator and intr == interval:
                    self.update_indicator_symbol(s, ind, intr)
            elif indicator == 'all':
                if s == sym and intr == interval:
                    self.update_indicator_symbol(s, ind, intr)
            elif s == sym and intr == interval and ind == indicator:
                self.update_indicator_symbol(s, ind, intr)
        
    def get_symbols(self):
        return self.library.list_symbols()
    
    def no_data(self, symbol):
        self.nodata.append(symbol)

    def delete_all(self):
        for sym in self.get_symbols():
            self.delete(sym)
            
    def list_indicator_symbols(self, sym='all', indicator='all', interval='daily'):
        stores = self.get_symbols()
        fin_store = []
        for store in stores:
            if '_VAL' not in store:
                intr = store[store.find('_', store.find('_')+1)+1:]
                ind = store[store.find('_')+1:store.find('_', store.find('_')+1)]
                s = store[:store.find('_')]
                if sym == 'all' and indicator == 'all':
                    if intr == interval:
                        fin_store.append(store)
                elif sym == 'all':
                    if ind == indicator and intr == interval:
                        fin_store.append(store)
                elif indicator == 'all':
                    if s == sym and intr == interval:
                        fin_store.append(store)
                else:
                    if s == sym and intr == interval and ind == indicator:
                        fin_store.append(store)
        return fin_store
    
    def list_value_symbols(self, sym='all'):
        stores = self.get_symbols()
        fin_store = []
        for store in stores:
            if '_VAL' in store:
                check = sym + '_VAL'
                if sym == 'all':
                    fin_store.append(store)
                elif check == store:
                    fin_store.append(store)
        return fin_store