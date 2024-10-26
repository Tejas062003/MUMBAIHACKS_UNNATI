from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_returns(data):
    data['% Change'] = data['Adj Close'].pct_change()
    data.dropna(inplace=True)
    annual_return = data['% Change'].mean() * 252 * 100
    stdev = data['% Change'].std() * np.sqrt(252) * 100
    risk_adj_return = annual_return / stdev
    return annual_return, stdev, risk_adj_return

def fetch_financial_data(ticker):
    tck = yf.Ticker(ticker)
    data = {}

    try:
        balance_sheet = tck.balance_sheet
        data['balance_sheet'] = balance_sheet.T if balance_sheet is not None and not balance_sheet.empty else None
    except Exception as e:
        data['balance_sheet'] = None

    try:
        income_statement = tck.financials
        data['income_statement'] = income_statement.T if income_statement is not None and not income_statement.empty else None
    except Exception as e:
        data['income_statement'] = None

    try:
        cash_flow = tck.cashflow
        data['cash_flow'] = cash_flow.T if cash_flow is not None and not cash_flow.empty else None
    except Exception as e:
        data['cash_flow'] = None

    return data

def fetch_news(ticker):
    tck = yf.Ticker(ticker)
    try:
        news = tck.news
    except Exception as e:
        news = []
    return news
from datetime import datetime, timedelta

def stock_analysis(request):
    context = {'has_data': False}
    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'MSFT')
        today = datetime.today().strftime('%Y-%m-%d')
        last_year = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        price = fetch_stock_data(ticker, start_date=last_year, end_date=today)
        data = yf.Ticker(ticker)
        stock_data = data.history(start=last_year, end=today)

        # Example DataFrame manipulation and analysis
        annual_return = stock_data['Close'].pct_change().mean() * 250
        stdev = stock_data['Close'].pct_change().std() * (250 ** 0.5)

        # Calculating Risk Adjusted Return assuming a risk-free rate of 0.01
        risk_adj_return = (annual_return - 0.01) / stdev

        # Check for financial and news data
        financial_data = data.financials
        news_data = data.news
        for news_item in news_data:
            if 'thumbnail' in news_item:
                thumbnail_url = news_item['thumbnail']['resolutions'][0]['url']
                news_item['thumbnail_url'] = thumbnail_url
            else:
                news_item['thumbnail_url'] = None  # Or any default value if thumbnail is not available

        context.update({
            'price': price,
            'ticker': ticker,
            'annual_return': annual_return,
            'stdev': stdev,
            'risk_adj_return': risk_adj_return,
            'financial_data': {'Financials': {'data': financial_data, 'is_empty': financial_data.empty}},
            'news_data': [
                {
                    'title': n['title'],
                    'link': n.get('link'),
                    'publisher': n.get('publisher'),
                    'thumbnail_url': n.get('thumbnail_url')  # Adding thumbnail URL to context
                } for n in news_data
            ],      
            'has_data': True
        })
        
        aiprompt = f"Summarize the latest trend, price, fundamentals, and news sentiments of this stock. Also, include the top 3 news articles and the top 3 reasons to buy or sell the stock. {context}"
        print(context)

        ai_summary = ai_summary_portfolio(aiprompt)
        context['ai_summary'] = ai_summary
    return render(request, 'aianalysis/stock_analysis.html', context)


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



#Baskets 

from django.shortcuts import render
from .models import InvestmentBasket

def fetch_investment_baskets(request):
    if request.method == 'GET':
        baskets = InvestmentBasket.objects.all()
        basket_list = []

        # Assuming you have a Basket model with a 'volatility' field
        low_risk = baskets.filter(volatility__range=(19, 23))
        mid_risk = baskets.filter(volatility__range=(24, 26))
        high_risk = baskets.filter(volatility__range=(27, 31))

        print(low_risk)
        for basket in baskets:
            basket_data = {
                                'id': basket.id,  # Include the id for URL generation
                'image' : basket.image,
                'name': basket.name,
                'cagr': basket.cagr,
                'volatility': basket.volatility,
                'manager_description': basket.manager_description,
                'minimum_investment_amount': basket.minimum_investment_amount,
          
            }
            basket_list.append(basket_data)
        context = {
            'baskets': basket_list,
                       'low_risk': low_risk,
            'mid_risk': mid_risk,
            'high_risk': high_risk,
        }
        print(context)
        return render(request, 'aianalysis/investment_baskets.html', context)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
import json
def investment_basket_detail(request, basket_id):
    # Retrieve the investment basket object
    basket = InvestmentBasket.objects.get(pk=basket_id)

    # Retrieve all holdings related to the investment basket
    holdings = basket.holdings.all()
    holdings = list(basket.holdings.values('asset_name', 'distribution_percentage'))

    # Render the template with the investment basket details
    return render(request, 'aianalysis/basket_details.html', {'basket': basket, 'holdings': holdings,'holdings_json': json.dumps(holdings)  # Pass the holdings as a JSON object
})




from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

def get_data(assets, start_date, end_date):
    df = pd.DataFrame()
    for stock in assets:
        df[stock] = yf.download(stock, start=start_date, end=end_date)['Adj Close']
    return df

def monte_carlo(df, assets, num_of_portfolios):
    log_returns = np.log(1 + df.pct_change())
    all_weights = np.zeros((num_of_portfolios, len(assets)))
    ret_arr = np.zeros(num_of_portfolios)
    vol_arr = np.zeros(num_of_portfolios)
    sharpe_arr = np.zeros(num_of_portfolios)
    
    for i in range(num_of_portfolios):
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        ret_arr[i] = np.sum(log_returns.mean() * weights) * 252
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    
    return pd.DataFrame({'Returns': ret_arr, 'Volatility': vol_arr, 'Sharpe Ratio': sharpe_arr, 'Weights': list(all_weights)})

def playground(request):
    return render(request, 'aianalysis/playground.html')

def optimize_portfolio(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        age = data['age']
        risk_tolerance = int(data['risk_tolerance'])
        annual_salary = float(data['salary'])
        savings = float(data['savings'])
        desired_amount = float(data['desired_amount'])
        assets = data['assets']

        start_date = datetime(2014, 1, 1)
        end_date = datetime.today()

        df = get_data(assets, start_date, end_date)

        weights = np.array([1/len(assets)] * len(assets))
        returns = df.pct_change()
        annual_return = np.sum(returns.mean() * weights) * 252
        cov_matrix_annual = returns.cov() * 252
        port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        port_volatility = np.sqrt(port_variance)

        num_of_portfolios = 5000
        sim_df = monte_carlo(df, assets, num_of_portfolios)
        max_sharpe_ratio = sim_df.loc[sim_df['Returns'].idxmax()]
        min_volatility = sim_df.loc[sim_df['Volatility'].idxmin()]

        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)

        ef_min_volatility = EfficientFrontier(mu, S)
        min_vol_weights = ef_min_volatility.min_volatility()
        min_vol_performance = ef_min_volatility.portfolio_performance()
        min_volatility = min_vol_performance[1]

        ef_max_sharpe = EfficientFrontier(mu, S)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        max_sharpe_performance = ef_max_sharpe.portfolio_performance()
        max_sharpe_volatility = max_sharpe_performance[1]

        investment_period = 60 - int(age)
        try:
            if risk_tolerance <= 3:
                raw_weights = ef.min_volatility()
            elif risk_tolerance >= 8:
                raw_weights = ef.max_sharpe()
            else:
                target_volatility = min_volatility + (max_sharpe_volatility - min_volatility) * ((risk_tolerance - 3) / 4)
                adjustment_factor = max(0.5, investment_period / 30)
                target_volatility *= adjustment_factor
                raw_weights = ef.efficient_risk(target_volatility)
        except ValueError as e:
            raw_weights = min_vol_weights

        cleaned_weights = ef.clean_weights()
        weight_sum = sum(cleaned_weights.values())
        if weight_sum != 1:
            cleaned_weights = {k: v / weight_sum for k, v in cleaned_weights.items()}

        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        optimized_weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
        optimized_weights_dict = optimized_weights_df['Weight'].to_dict()
        print(optimized_weights_dict)
        return JsonResponse({
            'expected_annual_return': expected_annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimized_weights': optimized_weights_dict
        })
        # Prepare the query parameters
        query_params = urllib.parse.urlencode({
            'expected_annual_return': expected_annual_return,
            'annual_volatility': annual_volatility,
            'optimized_weights': json.dumps(optimized_weights_dict)
        })

        # Redirect to better_basket view with the optimized portfolio data as parameters
        better_basket_url = reverse('better_basket') + '?' + query_params
        return redirect(better_basket_url)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def better_basket(request):
    playground_expected_annual_return = request.GET.get('expected_annual_return')
    playground_annual_volatility = request.GET.get('annual_volatility')
    playground_optimized_weights = request.GET.get('optimized_weights')

    playground_optimized_weights = json.loads(playground_optimized_weights)



    # Fetch the basket_id from the request
    basket_id = request.GET.get('basket_id', 2)  # Default to 1 if basket_id is not provided

    try:
        # Try to fetch the InvestmentBasket object based on the provided basket_id
        basket = InvestmentBasket.objects.get(id=basket_id)
    except InvestmentBasket.DoesNotExist:
        # Handle the case where the InvestmentBasket with the provided basket_id does not exist
        return JsonResponse({'error': 'InvestmentBasket with the provided ID does not exist'}, status=404)

    ai_prompt = f"""
    Compare the following two investment portfolios:

    1. Playground Portfolio:
    - Expected Annual Return: {float(playground_expected_annual_return):.2f}%
    - Annual Volatility: {float(playground_annual_volatility):.2f}%
    - Stocks: {playground_optimized_weights}

    2. Basket Portfolio:
    - Name: {basket.name}
    - Expected Annual REturns: {basket.cagr:.2f}%
    - Annual Volatility Volatility: {basket.volatility}
    - Stocks: {basket.basket_description}

    Only explain the why basket portfolio is the best
    Return the Facts that baskets are better because it invests across multiple asset classes and a wide range of assets also 
    sell the product by concise bullet points 
    Compulsory Requirement:
    *do not say negative information about the asset*
    """
    ai_summary = ai_bot(ai_prompt)
    context = {
        'playground_expected_annual_return': playground_expected_annual_return,
        'playground_annual_volatility': playground_annual_volatility,
        'playground_optimized_weights': playground_optimized_weights,
        'basket_annual_return': f'{basket.cagr:.2f}%',  # Fixed formatting issue here
        'basket_volatility': basket.volatility,
    'basket_description': basket.basket_description.split(),  # Split the string into a list
        'ai_prompt': ai_prompt,
        'ai_summary':ai_summary,
    }

    return render(request, 'aianalysis/better_basket.html', context)



def ai_bot(question):
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
        model_name="gemini-1.5-pro",  # Ensure this is a valid model name
        safety_settings=safety_settings,
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])
    aiprompt = f"{question}"
    # Send a message to the model and get a response
    response = chat_session.send_message(aiprompt)

    # Return the response text
    return response.text

def ai_bot_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question')

        # Assuming you have the ai_bot function defined elsewhere
        response_text = ai_bot(question)
        
        return JsonResponse({'response': response_text})

    return render(request, 'chatbot.html')




#risk slider playground
from django.shortcuts import render
from django.http import HttpResponse
import requests
def home(request):
    return render(request, 'aianalysis/simulate.html')
from django.shortcuts import render
from django.http import HttpResponse
import requests


from django.http import HttpResponse
from django.shortcuts import render
import requests

def simulate(request):
    if request.method == 'GET':
        symbols = request.GET.get('symbols')
        sim_no = request.GET.get('sim_no', 1000)  # Default to 1000 simulations if not specified

        if symbols:
            url = 'https://montecarloapi.onrender.com/portfolio'
            payload = {
                "Symbols": symbols,
                "sim_no": int(sim_no)
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                first_allocation = data.get('array_of_allocation', [None])[0]
                return render(request, 'aianalysis/simulate.html', {
                    'data': data,
                    'first_allocation': first_allocation,
                    'json_data': data
                })
            else:
                return HttpResponse(f"Error: {response.status_code} - {response.text}", status=response.status_code)
        else:
            return HttpResponse("Error: Missing symbols or invalid simulation number", status=400)
    elif request.method == 'POST':
        symbols = request.POST.get('symbols')
        sim_no = request.POST.get('sim_no', 1000)  # Default to 1000 simulations if not specified

        if symbols:
            url = 'https://montecarloapi.onrender.com/portfolio'
            payload = {
                "Symbols": symbols,
                "sim_no": int(sim_no)
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                first_allocation = data.get('array_of_allocation', [None])[0]
                return render(request, 'aianalysis/simulate.html', {
                    'data': data,
                    'first_allocation': first_allocation,
                    'json_data': data
                })
            else:
                return HttpResponse(f"Error: {response.status_code} - {response.text}", status=response.status_code)
        else:
            return HttpResponse("Error: Missing symbols or invalid simulation number", status=400)
    else:
        return HttpResponse("Method not allowed", status=405)



def financial_mentor(request):
    return render(request, 'aianalysis/mentor.html')


