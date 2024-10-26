from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si
from django.shortcuts import render
import numpy as np
from django.shortcuts import render
import requests
from django.http import JsonResponse

import requests
from django.http import JsonResponse
import pandas as pd

# Function to calculate future value of a series of cash flows
def future_value_annual_investment(P, r, t):
    FV = 0
    for i in range(1, t + 1):
        FV += P * ((1 + r) ** i)
    return FV

# Asset allocation based on risk profiles
risk_profiles = {
    "Very Conservative": {"Stocks": 0.20, "Bonds": 0.50, "Cash": 0.30},
    "Conservative": {"Stocks": 0.45, "Bonds": 0.40, "Cash": 0.15},
    "Moderate": {"Stocks": 0.65, "Bonds": 0.30, "Cash": 0.05},
    "Aggressive": {"Stocks": 0.80, "Bonds": 0.15, "Cash": 0.05},
    "Very Aggressive": {"Stocks": 0.90, "Bonds": 0.05, "Cash": 0.05}
}

# Annual return assumptions
annual_returns = {"Stocks": 0.07, "Bonds": 0.03, "Cash": 0.01}
high_returns = {"Stocks": 0.10, "Bonds": 0.05, "Cash": 0.02}
low_returns = {"Stocks": 0.04, "Bonds": 0.01, "Cash": 0.005}
# Stock symbols corresponding to sectors
stock_symbols = {
    'Technology': ['MSFT', 'AAPL', 'GOOGL', 'NVDA'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'AMGN'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'C'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'NKE', 'F', 'SBUX'],
    'Energy': ['XOM', 'CVX', 'SLB', 'COP', 'HAL'],
    'Industrials': ['BA', 'GE', 'MMM', 'HON', 'CAT'],
    'Utilities': ['NEE', 'DUK', 'SO', 'EXC', 'AEP'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
    'Real Estate': ['AMT', 'CCI', 'SPG', 'EQIX', 'O'],
    'Telecommunications': ['VZ', 'T', 'TMUS']
}

def index(request):
    return render(request, 'estimation.html', {'risk_profiles': risk_profiles.keys()})

def calculate_portfolio(request):
    risk_level = request.GET.get('risk_level')
    current_age = int(request.GET.get('current_age'))
    retirement_age = int(request.GET.get('retirement_age'))
    monthly_investment = int(request.GET.get('monthly_investment'))
    selected_sector = request.GET.get('selected_sector')  # Get the selected sector
    print(f'hi{selected_sector}')
    years_to_invest = retirement_age - current_age
    annual_investment = monthly_investment * 12

    profile = risk_profiles[risk_level]
    yearly_values = []
    highest_values = []
    lowest_values = []

    # Compute the future value of the portfolio for each year and the total portfolio at retirement
    for year in range(1, years_to_invest + 1):
        yearly_total = 0
        highest_total = 0
        lowest_total = 0
        for asset, allocation in profile.items():
            yearly_total += future_value_annual_investment(annual_investment * allocation, annual_returns[asset], year)
            highest_total += future_value_annual_investment(annual_investment * allocation, high_returns[asset], year)
            lowest_total += future_value_annual_investment(annual_investment * allocation, low_returns[asset], year)
        
        yearly_values.append((year + current_age, yearly_total))
        highest_values.append((year + current_age, highest_total))
        lowest_values.append((year + current_age, lowest_total))

    df = pd.DataFrame(yearly_values, columns=['Age', 'Portfolio Value (₹)'])
    df_high = pd.DataFrame(highest_values, columns=['Age', 'Highest Value (₹)'])
    df_low = pd.DataFrame(lowest_values, columns=['Age', 'Lowest Value (₹)'])

    total_value = yearly_values[-1][1] if yearly_values else 0
    highest_value = highest_values[-1][1] if highest_values else 0
    lowest_value = lowest_values[-1][1] if lowest_values else 0
    center_value = yearly_values[len(yearly_values) // 2][1] if yearly_values else 0

    total_invested = annual_investment * years_to_invest
    total_profit = total_value - total_invested
    total_profit_percent = (total_profit / total_invested) * 100 if total_invested > 0 else 0

    # Optimization step to estimate portfolio worth in the selected sector
    if selected_sector in stock_symbols:
        symbols = ','.join(stock_symbols[selected_sector])  # Get symbols for the selected sector
        url = "https://assetallocate.onrender.com/portfolio"
        data = {
            "Symbols": symbols,
        }
        print('hi')
        print(data)

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()  # Parse JSON response
        except requests.exceptions.RequestException as e:
            return JsonResponse({"error": str(e)}, status=500)

        # Extracting the annual return from the response for the selected sector
        sector_annual_return = result.get('annual_return', None)
        print(sector_annual_return)
        if sector_annual_return is not None:
            sector_worth = total_value * (sector_annual_return / annual_returns['Stocks'])
        else:
            sector_worth = 0
    else:
        sector_worth = 0
    print(sector_worth)
    return JsonResponse({
        'chart_data': df.to_dict(orient='records'),
        'chart_data_high': df_high.to_dict(orient='records'),
        'chart_data_low': df_low.to_dict(orient='records'),
        'total_value': total_value,
        'highest_value': highest_value,
        'lowest_value': lowest_value,
        'center_value': center_value,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_profit_percent': total_profit_percent,
        'sector_worth': sector_worth
    })

# def portfolio_view(request, stocks, stock_amounts, cash, etfs, etfs_amounts, cryptos, crypto_amounts):
#     values = []
#     for i, stock in enumerate(stocks):
#         try:
#             price = si.get_live_price(stock)
#             values.append(price * stock_amounts[i])
#         except Exception as e:
#             print(f"Error fetching price for {stock}: {e}")
#             values.append(0)

#     sectors = []
#     countries = []
#     market_caps = []
#     for x in stocks:
#         ticker = yf.Ticker(x)
#         info = ticker.info
#         sectors.append(info.get('sector', 'Unknown'))
#         countries.append(info.get('country', 'Unknown'))
#         market_caps.append(info.get('marketCap', 0))

#     etfs_values = []
#     for i, etf in enumerate(etfs):
#         try:
#             price = si.get_live_price(etf)
#             etfs_values.append(price * etfs_amounts[i])
#         except Exception as e:
#             print(f"Error fetching price for {etf}: {e}")
#             etfs_values.append(0)

#     crypto_values = []
#     for i, crypto in enumerate(cryptos):
#         try:
#             price = si.get_live_price(crypto)
#             crypto_values.append(price * crypto_amounts[i])
#         except Exception as e:
#             print(f"Error fetching price for {crypto}: {e}")
#             crypto_values.append(0)

#     general_dist = {
#         'Stocks': sum(values),
#         'ETFs': sum(etfs_values),
#         'Cryptos': sum(crypto_values),
#         'Cash': cash
#     }

#     sector_dist = {}
#     for i, sector in enumerate(sectors):
#         sector_dist.setdefault(sector, 0)
#         sector_dist[sector] += values[i]

#     country_dist = {}
#     for i, country in enumerate(countries):
#         country_dist.setdefault(country, 0)
#         country_dist[country] += values[i]

#     market_caps_dist = {'small': 0, 'mid': 0, 'large': 0, 'huge': 0}
#     for i, cap in enumerate(market_caps):
#         if cap < 2000000000:
#             market_caps_dist['small'] += values[i]
#         elif cap < 10000000000:
#             market_caps_dist['mid'] += values[i]
#         elif cap < 1000000000000:
#             market_caps_dist['large'] += values[i]
#         else:
#             market_caps_dist['huge'] += values[i]

#     context = {
#         'general_dist': general_dist,
#         'sector_dist': sector_dist,
#         'country_dist': country_dist,
#         'market_caps_dist': market_caps_dist
#     }

#     return context

def portfolio_view(request):
    stocks = ["HDFC.NS", "RELIANCE.NS", "HINDUNILVR.NS", "ADANIENT.NS","AAPL", "NFLX", "META", "TSLA", "AMZN", "NVDA", "MSFT"]
    amounts = [100,10, 30,40,20, 15, 20, 50, 5, 100, 5]

    values = []
    for i in range(len(stocks)):
        try:
            price = si.get_live_price(stocks[i])
            values.append(price * amounts[i])
        except Exception as e:
            print(f"Error fetching price for {stocks[i]}: {e}")
            values.append(0)

    sectors = []
    countries = []
    market_caps = []
    for x in stocks:
        ticker = yf.Ticker(x)
        info = ticker.info
        sectors.append(info.get('industry', 'Unknown'))
        countries.append(info.get('country', 'Unknown'))
        market_caps.append(info.get('marketCap', 0))

    cash = 40000
    etfs = ['IVV', 'XWD.TO']
    etfs_amounts = [30, 20]

    etfs_values = []
    for i in range(len(etfs)):
        try:
            price = si.get_live_price(etfs[i])
            etfs_values.append(price * etfs_amounts[i])
        except Exception as e:
            print(f"Error fetching price for {etfs[i]}: {e}")
            etfs_values.append(0)

    cryptos = ["ETH-USD", "BTC-USD", "ADA-USD","SOL-USD", "MATIC-USD"]
    crypto_amounts = [0.89, 0.34, 190, 2000,200]

    crypto_values = []
    for i in range(len(cryptos)):
        try:
            price = si.get_live_price(cryptos[i])
            crypto_values.append(price * crypto_amounts[i])
        except Exception as e:
            print(f"Error fetching price for {cryptos[i]}: {e}")
            crypto_values.append(0)

    general_dist = {
        'Stocks': sum(values),
        'Gold/Silver': sum(etfs_values),
        'Cryptos': sum(crypto_values),
        'Cash': cash
    }

    sector_dist = {}
    for i in range(len(sectors)):
        if sectors[i] not in sector_dist:
            sector_dist[sectors[i]] = 0
        sector_dist[sectors[i]] += values[i]

    country_dist = {}
    for i in range(len(countries)):
        if countries[i] not in country_dist:
            country_dist[countries[i]] = 0
        country_dist[countries[i]] += values[i]

    market_caps_dist = {'small': 0.0, 'mid': 0.0, 'large': 0.0, 'huge': 0.0}
    for i in range(len(stocks)):
        if market_caps[i] < 2000000000:
            market_caps_dist['small'] += values[i]
        elif market_caps[i] < 10000000000:
            market_caps_dist['mid'] += values[i]
        elif market_caps[i] < 1000000000000:
            market_caps_dist['large'] += values[i]
        else:
            market_caps_dist['huge'] += values[i]

    context = {
        'general_dist': general_dist,
        'sector_dist': sector_dist,
        'country_dist': country_dist,
        'market_caps_dist': market_caps_dist
    }

    return render(request, 'portfolio.html', context)

from .models import InvestmentOptimization

#api call to ml model 
def optimize_investment(request):
    if request.method == 'POST':
        print(request.POST)  # See what data is being received
        lifestyle_risk_map = {'low': 0, 'mid': 1, 'high': 2}
        lifestyle_risk = lifestyle_risk_map.get(request.POST.get('lifestyle_risk'), 2)  # Default to 0 if key is missing
        # expected_annual_roi = int(request.POST.get('expected_annual_roi', 0))
        expected_annual_roi_str = request.POST.get('expected_annual_roi', '0')
        expected_annual_roi = int(expected_annual_roi_str) if expected_annual_roi_str else 0
        # monthly_investment = float(request.POST.get('monthly_investment', 0))
        monthly_investment_str = request.POST.get('monthly_investment', '0')
        monthly_investment = float(monthly_investment_str) if monthly_investment_str else 0
        current_age = int(request.POST.get('current_age', 0))
        retirement_age = int(request.POST.get('retirement_age', 0))
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        annual_salary = float(request.POST.get('annual_salary', 0))
        monthly_expenses = float(request.POST.get('monthly_expenses', 0))
        savings = float(request.POST.get('savings', 0))

        # Calculate the principal amount
        investment_period = retirement_age - current_age
        principal_amount = monthly_investment * 12 * investment_period

        #save to db
                # Save data to database
        investment_optimization = InvestmentOptimization(
            name=name,
            email=email,
            lifestyle_risk=lifestyle_risk,
            expected_annual_roi=expected_annual_roi,
            monthly_investment=monthly_investment,
            current_age=current_age,
            retirement_age=retirement_age,
            annual_salary=annual_salary,
            monthly_expenses=monthly_expenses,
            savings=savings,
            principal_amount=principal_amount,
            investment_period=investment_period
        )
        investment_optimization.save()
        print("Saved to db")
        # Prepare the data payload for the API
        data = {
            "name":name,
            "lifestyle_risk": lifestyle_risk,
            "expected_annual_roi": expected_annual_roi,
            "principal_amount": principal_amount,
            "investment_period": investment_period,
        }
        print(data)
        # Make a POST request to the external API
        url = "https://amiastock.onrender.com/optimize"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Check for HTTP issues
            result = response.json()     # Get JSON data
        except requests.exceptions.RequestException as e:
            return JsonResponse({"error": str(e)}, status=500)


        # Zane task : Calculate total expected value and display it in the graph


        # Parse the clusters to capture allocation and weights for each stock
        cluster_allocation = []
        allocation_descriptions = []  # List to hold allocation descriptions for each cluster
        for cluster_result, cluster_info in zip(result['results'], result['clusters']):
            symbols = cluster_info['Symbols'].split(', ')
            weights = cluster_result['Weights']['Weights']
            
            stock_details = {symbol: weights.get(symbol, 0.0) for symbol in symbols}
            cluster_data = {
                "name": "Cluster containing " + ', '.join(symbols),
                "total_allocation": cluster_info['Weights'],
                "stock_allocations": stock_details
            }
            cluster_allocation.append(cluster_data)

            # Create a string describing the stock allocations for the current cluster
            allocation_description = ', '.join([f"{symbol}: {weight:.2%}" for symbol, weight in stock_details.items()])
            allocation_descriptions.append(allocation_description)

    

        # Define the AI prompt template
        prompt_template = """
This portfolio belongs to an investor with {lifestyle_risk} (0 for low risk, 1 for mid, 2 for high risk) tolerance and an expected annual ROI of {expected_annual_roi}%. The principal amount is ${principal_amount}. The portfolio is divided into three clusters (potentially low-cap, mid-cap, and high-cap stocks).

* Cluster 1: {cluster_1_names} - Allocations: {cluster_1_allocations}
* Cluster 2: {cluster_2_names} - Allocations: {cluster_2_allocations}
* Cluster 3: {cluster_3_names} - Allocations: {cluster_3_allocations}

Analyze this portfolio allocation and explain why it might be suitable for this aggressive investor seeking a high ROI. Consider these aspects:

* Justification for allocation percentages in each cluster.
* How the risk-reward profile aligns with the aggressive investment strategy.
* Benefits of diversification across multiple capitalizations.

**Note:** 
* Avoid mentioning specific URLs or citing the source of the data.
* Focus on the analysis and justification for the aggressive portfolio allocation.
"""

        # Create the AI prompt using the template and data
        ai_prompt = prompt_template.format(
            lifestyle_risk=lifestyle_risk,
            expected_annual_roi=data.get('expected_annual_roi', 'N/A'),
            principal_amount=data.get('principal_amount', 'N/A'),
            cluster_1_names=', '.join(cluster_allocation[0]['stock_allocations'].keys()),
            cluster_1_allocations=allocation_descriptions[0],
            cluster_2_names=', '.join(cluster_allocation[1]['stock_allocations'].keys()),
            cluster_2_allocations=allocation_descriptions[1],
            cluster_3_names=', '.join(cluster_allocation[2]['stock_allocations'].keys()),
            cluster_3_allocations=allocation_descriptions[2]
        )
        # ai_summary = ai_summary_portfolio(ai_prompt)
        ai_summary = (ai_prompt)

      
        print(ai_summary)  # Optionally, you could log this or perform further actions based on the AI prompt
        # Prepare the context for the Django template
        context = {
            "lifestyle_risk": lifestyle_risk,
            "expected_annual_roi": expected_annual_roi,
            "principal_amount": principal_amount,
            "investment_period": investment_period,
            "monthly_investable_amount" :monthly_investment,
            'data': result,
            'cluster_allocation': cluster_allocation,
            'leftover_funds': result['leftover_funds'][0]['leftover funds'],
            'ai_summary' : ai_summary,
        }
        # total_expected_return_cluster_wise, end_of_year_principal_cluster_wise = calculate_total_expected_return_cluster_wise(context)
        # print(f"Hi {total_expected_return_cluster_wise}{end_of_year_principal_cluster_wise}")
        print(context)
        # context.update({
        #     'total_expected_return_cluster_wise': total_expected_return_cluster_wise,
        #     'end_of_year_principal_cluster_wise': end_of_year_principal_cluster_wise,
        # })
        
        return render(request, 'pages/test.html', context)
    else:
        # Show the form initially
        return render(request, 'wizard-book-room.html')
        
# def calculate_total_expected_return_cluster_wise(data):
#     total_expected_return_cluster_wise = {}
#     end_of_year_principal_cluster_wise = {}
#     principal_amount = data.get('principal_amount', 10000)  # Retrieve principal amount from data or default to 0
#     investment_period = data.get('investment_period', 10)  # Retrieve investment period from data or default to 0

#     if 'data' in data and 'results' in data['data'] and 'cluster_allocation' in data:
#         for cluster in data['data']['results']:
#             expected_return = cluster.get('Expected Annual Return', 30)  # Retrieve expected annual return or default to 0
#             symbols = cluster.get('Symbols', '')
            
#             for allocation in data['cluster_allocation']:
#                 if symbols in allocation['name']:
#                     cluster_weight = allocation.get('total_allocation', 0)  # Retrieve total allocation or default to 0
#                     break
            
#             normalized_weight = cluster_weight / principal_amount if principal_amount != 0 else 0
#             total_expected_return_cluster_wise[allocation['name']] = expected_return * normalized_weight
#             print(f"this is {investment_period}")
#             # Calculate end of year principal for each year
#             for year in range(1, investment_period + 1):
#                 end_of_year_principal = principal_amount * (1 + expected_return / 100) ** year
#                 end_of_year_principal_cluster_wise.setdefault(allocation['name'], []).append(end_of_year_principal)
    
#     return total_expected_return_cluster_wise, end_of_year_principal_cluster_wise


import google.generativeai as genai

def ai_summary_portfolio(aiprompt):
    # Configure the API key for the generative AI
    genai.configure(api_key="AIzaSyDntr0t9CC0OMlY91LFn2nJjnMlJn1dAZ8")  # Replace "YOUR_API_KEY_HERE" with your actual API key

    # Create the generation configuration without the invalid field
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }

    # Define the safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]

    # Initialize the generative model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # Ensure this is a valid model name
        safety_settings=safety_settings,
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])

    # Send a message to the model and get a response
    response = chat_session.send_message(aiprompt)

    # Return the response text
    return response.text

def base(request):
    return render(request, 'pages/test.html')


def form(request):
    return render(request, 'wizard-book-room.html')
# import requests
# from django.http import JsonResponse
# from django.shortcuts import render
# import requests
# from django.http import JsonResponse
# from django.shortcuts import render

# def monte_carlo_simulation(request):
#     if request.method == "POST":
#         symbols = request.POST.get('symbols')
#         sim_no = int(request.POST.get('sim_no', 10000))  # Default to 10000 if not provided
        
#         url = "https://montecarloapi.onrender.com/portfolio"
#         payload = {
#             "Symbols": symbols,
#             "sim_no": sim_no
#         }
        
#         try:
#             response = requests.post(url, json=payload)
#             response.raise_for_status()  # Raise an error for bad status codes
#             data = response.json()       # Parse JSON response
#             array_of_allocation = data.get("array_of_allocation", [])
#             return JsonResponse({"array_of_allocation": array_of_allocation})
#         except requests.exceptions.RequestException as e:
#             return JsonResponse({"error": str(e)}, status=500)
#     else:
#         return JsonResponse({"error": "Invalid request method"}, status=400)




# from django.shortcuts import render

# def calculate_future_value(pv, rate, n):
#     return pv * ((1 + rate) ** n)

# def index(request):
#     if request.method == 'POST':
#         age = int(request.POST.get('age'))
#         retirement_age = int(request.POST.get('retirement_age'))
#         monthly_expense = float(request.POST.get('monthly_expense'))
#         current_corpus = float(request.POST.get('current_corpus'))
#         expected_retirement_amount = float(request.POST.get('expected_retirement_amount'))
#         inflation_rate = float(request.POST.get('inflation_rate')) / 100

#         years_to_invest = retirement_age - age
#         inflation_adjusted_expense = monthly_expense * ((1 + inflation_rate) ** years_to_invest)

#         # Assuming a basic compound interest calculation for monthly contributions
#         # Need an average monthly interest rate that reaches the expected retirement amount
#         # This part needs iterative solution or financial formulas to find the rate
#         # As a placeholder, we use a simple estimation method:
#         future_value = current_corpus
#         monthly_rate = 0.004  # Example rate of 0.4% per month
#         monthly_contribution = 0
#         while future_value < expected_retirement_amount:
#             monthly_contribution += 1  # Incremental approach to find required contribution
#             future_value = 0
#             for month in range(years_to_invest * 12):
#                 future_value = calculate_future_value(future_value + monthly_contribution, monthly_rate, 1)

#         required_monthly_investment = monthly_contribution

#         context = {
#             'years_to_invest': years_to_invest,
#             'required_monthly_investment': required_monthly_investment,
#             'total_invested_amount': required_monthly_investment * years_to_invest * 12,
#             'inflation_adjusted_expense': inflation_adjusted_expense
#         }

#         return render(request, 'calculate.html', context)
#     return render(request, 'calculate.html')





# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models, expected_returns
# import pandas as pd

# def calculate_optimized_portfolio(df, risk_tolerance, investment_period):
#     mu = expected_returns.mean_historical_return(df)
#     S = risk_models.sample_cov(df)
#     ef = EfficientFrontier(mu, S)

#     # Calculate minimum volatility portfolio
#     ef_min_volatility = EfficientFrontier(mu, S)
#     min_vol_weights = ef_min_volatility.min_volatility()
#     min_vol_performance = ef_min_volatility.portfolio_performance()
#     min_volatility = min_vol_performance[1]

#     # Calculate maximum Sharpe ratio portfolio
#     ef_max_sharpe = EfficientFrontier(mu, S)
#     max_sharpe_weights = ef_max_sharpe.max_sharpe()
#     max_sharpe_performance = ef_max_sharpe.portfolio_performance()
#     max_sharpe_volatility = max_sharpe_performance[1]

#     # Adjust portfolio optimization based on risk tolerance and investment period
#     try:
#         if risk_tolerance <= 3:
#             # Low risk tolerance: minimize volatility
#             raw_weights = ef.min_volatility()
#         elif risk_tolerance >= 8:
#             # High risk tolerance: maximize Sharpe ratio
#             raw_weights = ef.max_sharpe()
#         else:
#             # Intermediate risk tolerance: target volatility
#             # Linear interpolation between minimum and maximum volatility
#             target_volatility = min_volatility + (max_sharpe_volatility - min_volatility) * ((risk_tolerance - 3) / 4)

#             # Adjust target volatility based on investment period
#             adjustment_factor = max(0.5, investment_period / 30)  # Scale down volatility for shorter investment periods
#             target_volatility *= adjustment_factor

#             raw_weights = ef.efficient_risk(target_volatility)
#     except ValueError as e:
#         # In case of error, fallback to minimum volatility portfolio
#         print(f"Error optimizing portfolio: {e}")
#         raw_weights = min_vol_weights

#     cleaned_weights = ef.clean_weights()

#     # Ensure the weights sum to 1 (or 100%)
#     weight_sum = sum(cleaned_weights.values())
#     if weight_sum != 1:
#         cleaned_weights = {k: v / weight_sum for k, v in cleaned_weights.items()}

#     expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
#     optimized_weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])

#     return optimized_weights_df


# def portfolio_optimizer(request):
#     # Replace these with your actual data retrieval methods
#     df = ...  # DataFrame containing historical stock data
#     risk_tolerance = ...  # User's risk tolerance
#     investment_period = ...  # User's investment period

#     # Call the utility function to calculate optimized portfolio
#     optimized_portfolio = calculate_optimized_portfolio(df, risk_tolerance, investment_period)

#     return render(request, 'optimizer/portfolio_optimizer.html', {
#         'optimized_portfolio': optimized_portfolio
#     })








import requests
from django.http import JsonResponse
from django.shortcuts import render
from .models import InvestmentOptimization
import json

def optimize_investment1(request):
    if request.method == 'POST':
        print(request.POST)  # Debug: print received data
        lifestyle_risk_map = {'low': 0, 'mid': 1, 'high': 2}
        lifestyle_risk = lifestyle_risk_map.get(request.POST.get('lifestyle_risk', 'high'), 2)
        expected_annual_roi_str = request.POST.get('expected_annual_roi', '0')
        expected_annual_roi = int(expected_annual_roi_str) if expected_annual_roi_str.isdigit() else 0
        monthly_investment_str = request.POST.get('monthly_investment', '0')
        monthly_investment = float(monthly_investment_str) if monthly_investment_str.replace('.', '', 1).isdigit() else 0
        current_age = int(request.POST.get('current_age', 0))
        retirement_age = int(request.POST.get('retirement_age', 0))
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        annual_salary = float(request.POST.get('annual_salary', 0))
        monthly_expenses = float(request.POST.get('monthly_expenses', 0))
        savings = float(request.POST.get('savings', 0))

        # Calculate the principal amount
        investment_period = retirement_age - current_age
        principal_amount = monthly_investment * 12 * investment_period
        
        # Save data to database
        investment_optimization = InvestmentOptimization(
            name=name,
            email=email,
            lifestyle_risk=lifestyle_risk,
            expected_annual_roi=expected_annual_roi,
            monthly_investment=monthly_investment,
            current_age=current_age,
            retirement_age=retirement_age,
            annual_salary=annual_salary,
            monthly_expenses=monthly_expenses,
            savings=savings,
            principal_amount=principal_amount,
            investment_period=investment_period
        )
        investment_optimization.save()
        print("Data saved to the database")

        data = {
            "current_age": current_age,
            "lifestyle_risk": lifestyle_risk,
            "expected_annual_roi": expected_annual_roi,
            "principal_amount": principal_amount,
            "risk": 10  # This may need to be adjusted based on actual risk calculations
        }


        # API endpoints
        url = "https://assetallocate.onrender.com/weights"
        cluster_url = "https://assetallocate.onrender.com/optimize"

        try:
            # API call to get cluster weights
            response = requests.post(cluster_url, json=data)
            response.raise_for_status()
            result = response.json()

            # API call for stock-wise allocations
            cluster_response = requests.post(url, json=data)
            cluster_response.raise_for_status()
            cluster_result = cluster_response.json()

            print(result)

            # Processing the data from the API response
            stock_allocations = []
            if "results" in result:
                for item in result["results"]:
                    weights = item.get("Weights", {}).get("Weights", {})
                    stock_allocations.append(weights)
            

            risk_wise = cluster_result['clusters']

            # Prepare data for AI model
            prompt_template = f"""
            This portfolio belongs to an investor with a lifestyle risk score of {data['lifestyle_risk']} and an expected annual ROI of {data['expected_annual_roi']}%. The principal amount invested is Rs. {data['principal_amount']}. The portfolio is divided into three clusters, potentially reflecting conservative, high risk, and mid risk stocks or cryptocurrencies.
            
            Risk wise Distribution: {json.dumps(risk_wise, indent=2)} 
            Stock Clusters: {json.dumps(stock_allocations, indent=2)}

            Please analyze this portfolio allocation and explain why it might be suitable for this aggressive investor seeking a high ROI. Consider these aspects:
            
            - Justification for allocation percentages in each cluster.
            - How the risk-reward profile aligns with the aggressive investment strategy.
            - Benefits of diversification across multiple asset classes.

            **Note:** 
            - Avoid mentioning specific URLs or citing the source of the data.
            - Focus on the analysis and justification for the aggressive portfolio allocation.
            """

            # Send prompt to AI model
            ai_summary = ai_summary_portfolio(prompt_template)
            print(ai_summary)

            # Preparing context data for the template
            context = {
                'cluster_allocation': cluster_result['clusters'],
                'stock_allocations': stock_allocations,
                'expected_annual_roi': expected_annual_roi,
                'investment_period': investment_period,
                            "principal_amount": principal_amount,
                            'monthly_investment' : monthly_investment,
                            'ai_summary':ai_summary,

            }

            # print(cluster_result['clusters'])
                # print(f"hi{stock_allocations}")
                        # Define the AI prompt template

                
            return render(request, 'pages/test1.html', context)
        except requests.exceptions.RequestException as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return render(request, 'wizard-book-room1.html')