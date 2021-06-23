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
    r = request.post(ORDERS_URL, json=data, headers=HEADERS)
    return json.loads(r.content)
    
def get_orders():
    r = requests.get(ORDERS_URL, headers=HEADERS)
    return json.loads(r.content)

def get_positions():
    r = requests.get(ALPACA_POSITIONS_URL, headers=HEADERS)
    return json.loads(r.content)

def get_position(sym=None):
    assert sym is not None, "supply a symbol"
    r = requests.get(f'{ALPACA_POSITIONS_URL}/{sym}', headers=HEADERS)
    return json.loads(r.content)

def get_assets():
    r = requests.get(ALPACA_ASSETS_URL, headers=HEADERS)
    return json.loads(r.content)