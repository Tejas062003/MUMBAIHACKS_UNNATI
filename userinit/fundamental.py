import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# Set the title of the app
st.title('Stock Dashboard')

# Sidebar inputs for user to select ticker, start date, and end date
st.sidebar.header('User Input Features')
ticker = st.sidebar.text_input('Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

@st.cache_data
def load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

data = load_data(ticker, start_date, end_date)

# Create tabs for different sections of the dashboard
pricing_data, fundamental_data, news_tab = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    
    # Plotting the closing price
    fig = px.line(data, x=data.index, y='Adj Close', title=f'{ticker} Adjusted Close Price')
    st.plotly_chart(fig)

    # Calculating and displaying annual return, standard deviation, and risk-adjusted return
    data['% Change'] = data['Adj Close'].pct_change()
    data.dropna(inplace=True)
    annual_return = data['% Change'].mean() * 252 * 100
    stdev = data['% Change'].std() * np.sqrt(252) * 100
    risk_adj_return = annual_return / stdev

    st.markdown(f"**Annual Return:** {annual_return:.2f}%")
    st.markdown(f"**Standard Deviation:** {stdev:.2f}%")
    st.markdown(f"**Risk Adjusted Return:** {risk_adj_return:.2f}")

def fetch_financial_data(ticker):
    tck = yf.Ticker(ticker)
    data = {}
    try:
        balance_sheet = tck.balance_sheet
        data['balance_sheet'] = balance_sheet.T if balance_sheet is not None and not balance_sheet.empty else None
    except Exception as e:
        data['balance_sheet'] = None
        st.error(f"Error fetching balance sheet: {e}")

    try:
        income_statement = tck.financials
        data['income_statement'] = income_statement.T if income_statement is not None and not income_statement.empty else None
    except Exception as e:
        data['income_statement'] = None
        st.error(f"Error fetching income statement: {e}")

    try:
        cash_flow = tck.cashflow
        data['cash_flow'] = cash_flow.T if cash_flow is not None and not cash_flow.empty else None
    except Exception as e:
        data['cash_flow'] = None
        st.error(f"Error fetching cash flow statement: {e}")

    return data

with fundamental_data:
    st.header('Fundamental Data')
    financial_data = fetch_financial_data(ticker)

    for key, value in financial_data.items():
        if value is not None:
            st.subheader(f'{key.replace("_", " ").title()}')
            st.dataframe(value)
        else:
            st.error(f"No {key.replace('_', ' ')} data available.")

import google.generativeai as genai
def stock_news(ticker):
        
    # Configure the API key for the generative AI
    genai.configure(api_key="AIzaSyDntr0t9CC0OMlY91LFn2nJjnMlJn1dAZ8")

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
    context = (f"Give 3 reasons to buy and sell {ticker} stock, give short summary of stock trend, and one word sentiment of stock and future scope")
    
    # Send a message to the model and get a response
    response = chat_session.send_message(context)

    # Print the response text

    return response.text
   




with news_tab:
    st.header(f'News for {ticker}')
    
    answer = stock_news(ticker)
    print(f"hi {answer}")
    st.write(answer)

 


    st.write("**Disclaimer:** This information is for educational purposes only and should not be considered financial advice. Conduct thorough research and consult with a financial advisor before making investment decisions.")
# from django.test import TestCase

# # Create your tests here.
# import streamlit as st, pandas as pd, numpy as np, yfinance as yf
# from alpha_vantage.fundamentaldata import FundamentalData
# import plotly.express as px
# from stocknews import StockNews

# st.title('Stock Dashboard')
# ticker = st.sidebar.text_input('Ticker')
# start_date = st.sidebar.date_input('Start Date')
# end_data = st.sidebar.date_input('End Date')

# data = yf.download(ticker, start=start_date, end=end_data)
# fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
# st.plotly_chart(fig)


# pricing_data, fundamental_data, news = st.tabs(["Pricing Data","Fundamengtal Data","Top 10 News"])
# with pricing_data:
#     st.header('Price Movements')
#     data2 = data
#     data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
#     data2.dropna(inplace=True)
#     annual_return = data2['% Change'].mean()*252*100
#     st.write("Annual Return is ",annual_return, '%')
#     st.write(data2)

#     #Standard Deviation Annually (sqrt of number of days exlcuding the weekends)
#     stdev = np.std(data2['% Change'])*np.sqrt(252)
#     st.write('Standard Deviation is', stdev*100, '%')
#     st.write('Risk Adj Return is',annual_return/(stdev*100))

# # from alpha_vantage.fundamentaldata import FundamentalData
# # with fundamental_data:
# #     api_key = 'ZGCEKWTB9DAT550O'
# #     fd = FundamentalData(api_key, output_format='pandas')
# #     st.subheader('Balance Sheet')
# #     balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
# #     bs = balance_sheet.T[2:]
# #     bs.columns = list(balance_sheet.T.iloc[0])
# #     st.write(bs)
# #     st.subheader('Income Statement')
# #     income_statement = fd.get_income_statement_annual(ticker)[0]
# #     is1 = income_statement.T[2:]
# #     is1.columns = list(income_statement.T.iloc[0])
# #     st.write(is1)
# #     st.subheader('Cash Flow Statements')
# #     cash_flow = fd.get_cash_flow_annual(ticker)[0]
# #     cf = cash_flow.T[2:]
# #     cf.columns = list(cash_flow.T.iloc[0])
# #     st.write(cf)
# #     st.write("Fundamentals")
#     # Function to fetch and process financial data using yfinance
# def fetch_financial_data(ticker):
#     tck = yf.Ticker(ticker)
#     data = {}

#     try:
#         # Balance Sheet
#         balance_sheet = tck.balance_sheet
#         if balance_sheet is not None and not balance_sheet.empty:
#             data['balance_sheet'] = balance_sheet.T
#         else:
#             data['balance_sheet'] = None
#     except Exception as e:
#         st.error(f"Error fetching balance sheet: {e}")
#         data['balance_sheet'] = None

#     try:
#         # Income Statement
#         income_statement = tck.financials
#         if income_statement is not None and not income_statement.empty:
#             data['income_statement'] = income_statement.T
#         else:
#             data['income_statement'] = None
#     except Exception as e:
#         st.error(f"Error fetching income statement: {e}")
#         data['income_statement'] = None

#     try:
#         # Cash Flow Statement
#         cash_flow = tck.cashflow
#         if cash_flow is not None and not cash_flow.empty:
#             data['cash_flow'] = cash_flow.T
#         else:
#             data['cash_flow'] = None
#     except Exception as e:
#         st.error(f"Error fetching cash flow statement: {e}")
#         data['cash_flow'] = None

#     return data

# import yahoo_fin.stock_info as si

# with fundamental_data:
#     tck = yf.Ticker(ticker)
#     fundamentatls = tck.get_financials()
    
#     financial_data = fetch_financial_data(ticker)

#     # Display Balance Sheet
#     st.subheader('Balance Sheet')
#     if financial_data['balance_sheet'] is not None:
#         st.write(financial_data['balance_sheet'])
#     else:
#         st.error("No balance sheet data available.")

#     # Display Income Statement
#     st.subheader('Income Statement')
#     if financial_data['income_statement'] is not None:
#         st.write(financial_data['income_statement'])
#     else:
#         st.error("No income statement data available.")

#     # Display Cash Flow Statement
#     st.subheader('Cash Flow Statement')
#     if financial_data['cash_flow'] is not None:
#         st.write(financial_data['cash_flow'])
#     else:
#         st.error("No cash flow statement data available.")

#     st.write("Fundamentals")



# from stocknews import StockNews
# with news:
#     st.header(f'News of {ticker}')
    
#     # Fetch news data using StockNews
#     try:
#         sn = StockNews(ticker, save_news=False)
#         df_news = sn.read_rss()
        
#         # Display the latest 10 news articles
#         for i in range(10):
#             if i < len(df_news):
#                 st.subheader(f'News {i+1}')
#                 st.write(df_news['published'][i])
#                 st.write(df_news['title'][i])
#                 st.write(df_news['summary'][i])
#                 title_sentiment = df_news['sentiment_title'][i]
#                 st.write(f'Title sentiment: {title_sentiment}')
#                 news_sentiment = df_news['sentiment_summary'][i]
#                 st.write(f'News sentiment: {news_sentiment}')
#             else:
#                 st.write("No more news available.")
#     except Exception as e:
#         st.error(f"Error fetching news: {e}")






#calculator
#import streamlit as st
# import pandas as pd

# # Function to calculate future value of a series of cash flows
# def future_value_annual_investment(P, r, t):
#     FV = 0
#     for i in range(1, t + 1):
#         FV += P * ((1 + r) ** i)
#     return FV

# # Asset allocation based on risk profiles
# risk_profiles = {
#     "Very Conservative": {"Stocks": 0.20, "Bonds": 0.50, "Cash": 0.30},
#     "Conservative": {"Stocks": 0.45, "Bonds": 0.40, "Cash": 0.15},
#     "Moderate": {"Stocks": 0.65, "Bonds": 0.30, "Cash": 0.05},
#     "Aggressive": {"Stocks": 0.80, "Bonds": 0.15, "Cash": 0.05},
#     "Very Aggressive": {"Stocks": 0.90, "Bonds": 0.05, "Cash": 0.05}
# }

# # Annual return assumptions
# annual_returns = {"Stocks": 0.07, "Bonds": 0.03, "Cash": 0.01}

# # Streamlit UI components
# st.title("Investment Portfolio Growth Calculator")
# risk_level = st.selectbox("Select your risk tolerance", list(risk_profiles.keys()))
# current_age = st.number_input("Current Age", min_value=18, max_value=100, value=30)
# retirement_age = st.number_input("Retirement Age", min_value=current_age + 1, max_value=100, value=65)
# monthly_investment = st.number_input("Monthly Investment Amount (in ₹)", min_value=0, value=5000)

# # Calculate the investment period and monthly investment to annual
# years_to_invest = retirement_age - current_age
# annual_investment = monthly_investment * 12

# if st.button("Calculate Portfolio Growth"):
#     profile = risk_profiles[risk_level]
#     total_value = 0
#     yearly_values = []

#     # Compute the future value of the portfolio for each year and the total portfolio at retirement
#     for year in range(1, years_to_invest + 1):
#         yearly_total = 0
#         for asset, allocation in profile.items():
#             FV = future_value_annual_investment(annual_investment * allocation, annual_returns[asset], year)
#             yearly_total += FV
#         yearly_values.append((year + current_age, yearly_total))
#         total_value = yearly_total  # The value at the last iteration will be at retirement

#     # Create a DataFrame for plotting
#     df = pd.DataFrame(yearly_values, columns=['Age', 'Portfolio Value (₹)'])

#     # Plot the results
#     st.line_chart(df.set_index('Age'))

#     # Output total at retirement
#     st.write(f"Total Expected Portfolio Worth at Retirement (Age {retirement_age}): ₹{total_value:,.2f}")
