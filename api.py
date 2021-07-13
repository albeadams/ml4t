from paper_config import *

ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
HEADERS = {'APCA-API-KEY-ID': ALPACA_API_KEY, 'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY}

#Endpoints
ALPACA_ACCOUNT_URL = f'{ALPACA_BASE_URL}/v2/account' # view account info
ALPACA_ORDERS_URL = f'{ALPACA_BASE_URL}/v2/orders'  # place order
ALPACA_ASSETS_URL = f'{ALPACA_BASE_URL}/v2/assets'  # get all assets
ALPACA_POSITIONS_URL = f'{ALPACA_BASE_URL}/v2/positions'  # get all positions
ALPACA_LATEST_VALUE_URL = f'{ALPACA_BASE_URL}/v2/stocks'

#Alpha Vantage : 
ALPHA_VANTAGE_KEY = ALPHA_VANTAGE_SECRET_KEY