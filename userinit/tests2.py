

import requests
from django.http import JsonResponse

def optimize_investment():
    url="https://assetallocate.onrender.com"
    data = {
        "Symbols" : stocks,
    }
    url = "https://amiastock.onrender.com/optimize"
    data = {
        "lifestyle_risk": 1,
        "expected_annual_roi": 50,
        "principal_amount": 800000
    }
    
    try:
        response = requests.post(url, json=data)
        print("r1")
        response.raise_for_status()
        print("r2")
  # Raise an error for bad status codes
        result = response.json()  
        print("r3")
   # Parse JSON response
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)
    print(result)
    return JsonResponse(result)

# optimize_investment()




import requests
from django.http import JsonResponse
#tech 50%, #Energy 13%
def optimize_investment1():
    url = "https://assetallocate.onrender.com/portfolio"
    data = {
        "Symbols": 'XOM,CVX,SLB,COP,HAL',
    }

    try:
        response = requests.post(url, json=data)
        print("r1")
        response.raise_for_status()  # Raise an error for bad status codes
        print("r2")
        result = response.json()  # Parse JSON response
        print("r3")
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)

    # Extracting the annual return from the response
    annual_return = result.get('annual_return', None)
    if annual_return is not None:
        print("Annual Return:", annual_return)
    else:
        print("Annual return not found in the response.")

    print(result)
    return JsonResponse(result)

optimize_investment1()
# import yfinance as yf
# from yahoo_fin import stock_info as si
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import pickle
# import numpy as np

# plt.style.use("dark_background")

# # List of stocks you own [take input from stock recommendation and allocation]
# stocks = ["AAPL", "NFLX", "META", "TSLA", "AMZN", "NVDA", "MSFT"]

# # How many shares you own?
# amounts = [20, 15, 20, 50, 5, 100, 5]

# # Get live prices and calculate values, handle possible errors
# values = []
# for i in range(len(stocks)):
#     try:
#         price = si.get_live_price(stocks[i])
#         values.append(price * amounts[i])
#     except Exception as e:
#         print(f"Error fetching price for {stocks[i]}: {e}")
#         values.append(0)

# # Get sectors, countries, and market caps, handle possible errors
# sectors = []
# countries = []
# market_caps = []
# for x in stocks:
#     ticker = yf.Ticker(x)
#     info = ticker.info
#     sectors.append(info.get('industry', 'Unknown'))
#     countries.append(info.get('country', 'Unknown'))
#     market_caps.append(info.get('marketCap', 0))

# cash = 40000
# etfs = ['IVV', 'XWD.TO']
# etfs_amounts = [30, 20]

# # Get ETF values, handle possible errors
# etfs_values = []
# for i in range(len(etfs)):
#     try:
#         price = si.get_live_price(etfs[i])
#         etfs_values.append(price * etfs_amounts[i])
#     except Exception as e:
#         print(f"Error fetching price for {etfs[i]}: {e}")
#         etfs_values.append(0)

# cryptos = ["ETH-USD", "BTC-USD", "ADA-USD"]
# crypto_amounts = [0.89, 0.34, 190]

# # Get crypto values, handle possible errors
# crypto_values = []
# for i in range(len(cryptos)):
#     try:
#         price = si.get_live_price(cryptos[i])
#         crypto_values.append(price * crypto_amounts[i])
#     except Exception as e:
#         print(f"Error fetching price for {cryptos[i]}: {e}")
#         crypto_values.append(0)

# # General distribution
# general_dist = {
#     'Stocks': sum(values),
#     'ETFs': sum(etfs_values),
#     'Cryptos': sum(crypto_values),
#     'Cash': cash
# }

# # Sector distribution
# sector_dist = {}
# for i in range(len(sectors)):
#     if sectors[i] not in sector_dist:
#         sector_dist[sectors[i]] = 0
#     sector_dist[sectors[i]] += values[i]

# # Country distribution
# country_dist = {}
# for i in range(len(countries)):
#     if countries[i] not in country_dist:
#         country_dist[countries[i]] = 0
#     country_dist[countries[i]] += values[i]

# # Market caps distribution
# market_caps_dist = {'small': 0.0, 'mid': 0.0, 'large': 0.0, 'huge': 0.0}
# for i in range(len(stocks)):
#     if market_caps[i] < 2000000000:
#         market_caps_dist['small'] += values[i]
#     elif market_caps[i] < 10000000000:
#         market_caps_dist['mid'] += values[i]
#     elif market_caps[i] < 1000000000000:
#         market_caps_dist['large'] += values[i]
#     else:
#         market_caps_dist['huge'] += values[i]

# # Save distributions
# with open('general.pickle', 'rb') as f:
#     general_dist = pickle.load(f)

# with open('industry.pickle', 'rb') as f:
#     sector_dist = pickle.load(f)

# with open('country.pickle', 'rb') as f:
#     country_dist = pickle.load(f)

# with open('market_caps.pickle', 'rb') as f:
#     market_caps_dist = pickle.load(f)


# # Plot distributions
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# fig.suptitle("Portfolio Diversification Analysis", fontsize=18)
# colors = list(mcolors.TABLEAU_COLORS.values())

# axs[0, 0].pie(general_dist.values(), labels=general_dist.keys(), autopct="%1.1f%%", pctdistance=0.85, colors=colors)
# axs[0, 0].set_title("General Distribution")

# axs[0, 1].pie(sector_dist.values(), labels=sector_dist.keys(), autopct="%1.1f%%", pctdistance=0.85, colors=colors)
# axs[0, 1].set_title("Industry Distribution")

# axs[1, 0].pie(country_dist.values(), labels=country_dist.keys(), autopct="%1.1f%%", pctdistance=0.85, colors=colors)
# axs[1, 0].set_title("Country Distribution")

# axs[1, 1].pie(market_caps_dist.values(), labels=market_caps_dist.keys(), autopct="%1.1f%%", pctdistance=0.85, colors=colors)
# axs[1, 1].set_title("Market Cap Distribution")

# plt.show()
