import requests, json

from paper_config import *

def get_account():
    r = requests.get(ALPACA_ACCOUNT_URL, headers=HEADERS)
    return json.loads(r.content)

def create_order(symbol, qty, side, type, time_in_force):
    data = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': type,
        'time_in_force': time_in_force
    }
    r = request.post(ORDERS_URL, json=data, header=HEADERS)
    return json.loads(r.content)
    
def get_orders():
    r = requests.get(ORDERS_URL, header=HEADERS)
    return json.loads(r.content)