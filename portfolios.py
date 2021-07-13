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
            if link.text == 'ZTS':
                break
    return names


class Portfolio:
    def __init__(self, use_alpaca=True, cash=10000):
        self.use_alpaca = use_alpaca
        if use_alpaca:
            self.cash_remaining = int(alpaca.get_account()['cash'])
            self.positions = alpaca.get_positions() # check on how returned from alpaca...?????????
            self.position_amount = 0 # ????
        else:
            self.cash_remaining = cash
            self.positions = []
            self.position_amount = dict()
        print(f'Portfolio created - available cash: {self.cash_remaining}')
        
    def has_position(self, sym):
        # check logic - need amount as well
        if not self.use_alpaca: return sym in self.positions
        # below not correct... neet to test api...
        if sym in self.positions: # not sure format positions returned...
            return alpaca.get_position(sym)
        else:
            return "no position"
        
    def buy(self, sym, amount, current_price):
        if self.use_alpaca:
            alpaca.create_order(symbol=sym, 
                                qty=amount, 
                                side='buy')
        else:
            # this is testing with fake money
            cost = current_price * amount
            if cost > self.cash_remaining:
                amount = self.cash_remaining//current_price # go all in
                cost = current_price * amount
            if not self.has_position(sym):
                self.positions.append(sym)
                self.position_amount[sym] = amount
            else:
                self.position_amount[sym] += amount
            self.cash_remaining -= cost
            
    def sell(self, sym, amount, current_price):
        assert self.has_position(sym), f"No position in {sym}"
        if self.use_alpaca:
            alpaca.create_order(symbol=sym, 
                                qty=amount, 
                                side='sell')
        else:
            if amount > self.position_amount[sym]:
                print(f'Attempting to sell off more {sym} than have (have {self.position_amount[sym]}, selling {amount}). Correcting...')
                amount = self.position_amount[sym] # sell off all position
            profit = current_price * amount
            self.position_amount[sym] -= amount
            if self.position_amount[sym] <= 0: # no more position in symbol
                self.positions.remove(sym)
                self.position_amount.remove(sym)
            self.cash_remaining += profit
            
    def save(self):
        # ToDo : serialize
        pass
    
    def load(self):
        # ToDo : deserialize
        pass