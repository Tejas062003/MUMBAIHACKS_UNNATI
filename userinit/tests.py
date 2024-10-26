import streamlit as st 
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import concurrent.futures

# Setting the page configuration
st.set_page_config(
    page_title="Retirement Portfolio Optimizer",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to fetch historical stock data
@st.cache_resource
def get_data(assets, start_date, end_date):
    df = pd.DataFrame()
    yf.pdr_override()
    for stock in assets:
        df[stock] = web.get_data_yahoo(stock, start=start_date, end=end_date)['Adj Close']
    return df

# Function to perform Monte Carlo simulation
@st.cache_resource
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

# Sidebar inputs
st.sidebar.subheader("Enter Your Details")
salary = st.sidebar.number_input("Annual Salary ($)", value=50000, step=1000)
age = st.sidebar.number_input("Current Age", value=30, step=1)
savings = st.sidebar.number_input("Current Savings ($)", value=10000, step=500)
desired_amount = st.sidebar.number_input("Desired Amount at Retirement ($)", value=1000000, step=5000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (1 = Low, 5 = Medium, 10 = High)", 1, 10, 5)
start_date = st.sidebar.date_input("Start Date", datetime(2014, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Calculate the investment period
investment_period = 60 - age
if investment_period <= 0:
    st.sidebar.error("Age should be less than 60 to calculate the investment period.")
else:
    # Asset class selection
    asset_classes = {
        "Technology": {
            "High Cap": ["INFY.NS", "TCS.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS"],
            "Mid Cap": ["LTIM.NS", "KPITTECH.NS", "MPHASIS.NS", "LTI.NS", "COFORGE.NS"],
            "Low Cap": ["TVSELECT.NS", "VAKRANGEE.NS", "MASTEK.NS", "GTLINFRA.NS", "FSL.NS"]
        },
        "Healthcare": {
            "High Cap": ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "METROPOLIS.NS"],
            "Mid Cap": ["AUROPHARMA.NS", "ALKEM.NS", "BIOCON.NS", "TORNTPHARM.NS", "IPCALAB.NS"],
            "Low Cap": ["BLISSGVS.NS", "MARKSANS.NS", "KMCSHIL.NS", "SMSLIFE.NS", "INDOCO.NS"]
        },
        "Finance": {
            "High Cap": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS"],
            "Mid Cap": ["BAJFINANCE.NS", "BANDHANBNK.NS", "CHOLAFIN.NS", "L&TFH.NS", "M&MFIN.NS"],
            "Low Cap": ["AUBANK.NS", "ABFRL.NS", "BATAINDIA.NS", "BHARTIARTL.NS", "CIPLA.NS"]
        },
        "Consumer Goods": {
            "High Cap": ["HINDUNILVR.NS", "NESTLEIND.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS"],
            "Mid Cap": ["JUBLFOOD.NS", "UBL.NS", "PIDILITIND.NS", "BRITANNIA.NS", "COLPAL.NS"],
            "Low Cap": ["VENKEYS.NS", "VADILALIND.NS", "ZENSARTECH.NS", "VSTIND.NS", "EMAMILTD.NS"]
        },
        "Energy": {
            "High Cap": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"],
            "Mid Cap": ["IGL.NS", "GUJGASLTD.NS", "MGL.NS", "PETRONET.NS", "COALINDIA.NS"],
            "Low Cap": ["MRPL.NS", "IOB.NS", "IWEL.NS", "NFL.NS", "HINDPETRO.NS"]
        },
        "Industrial": {
            "High Cap": ["LT.NS", "BAJAJ-AUTO.NS", "TITAN.NS", "TATASTEEL.NS", "JSWSTEEL.NS"],
            "Mid Cap": ["SAIL.NS", "VEDL.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "JINDALSTEL.NS"],
            "Low Cap": ["AIAENG.NS", "ATUL.NS", "KSB.NS", "APLAPOLLO.NS", "CROMPTON.NS"]
        }
    }

    selected_assets = []
    for asset_class, caps in asset_classes.items():
        for cap, assets in caps.items():
            with st.sidebar.expander(f"{asset_class} ({cap})", expanded=True):
                for asset in assets:
                    selected = st.checkbox(asset, value=False)
                    if selected:
                        selected_assets.append(asset)

    if len(selected_assets) < 2:
        st.warning(" Please select two or more stocks to proceed ")
    else:
        # Fetch data
        df = get_data(selected_assets, start_date=start_date, end_date=end_date)
        weights = np.array([1/len(selected_assets)] * len(selected_assets))

        # Display stock prices
        st.subheader("Stock Price Chart")
        price_chart = px.line(df, title="Stock Prices Over Time")
        st.plotly_chart(price_chart)

        st.write("Double click on the legend to isolate the stock")

        # Display data
        st.subheader("Stock Prices Data")
        st.dataframe(df)

        # Portfolio statistics
        returns = df.pct_change()
        annual_return = np.sum(returns.mean() * weights) * 252
        cov_matrix_annual = returns.cov() * 252
        port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        port_volatility = np.sqrt(port_variance)

        st.subheader("Portfolio Statistics")
        st.write(f"Annual Return: {annual_return:.2%}")
        st.write(f"Annual Volatility: {port_volatility:.2%}")
        st.write(f"Portfolio Variance: {port_variance:.2%}")

        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation")
        num_of_portfolios = 5000
        sim_df = monte_carlo(df, selected_assets, num_of_portfolios)
        max_sharpe_ratio = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]
        min_volatility = sim_df.loc[sim_df['Volatility'].idxmin()]

        monte_fig = px.scatter(
            sim_df, x="Volatility", y="Returns", color="Sharpe Ratio",
            title="Monte Carlo Simulation", labels={"Returns": "Expected Annual Return", "Volatility": "Annual Volatility"}
        )
        monte_fig.add_trace(go.Scatter(x=[max_sharpe_ratio['Volatility']], y=[max_sharpe_ratio['Returns']], mode='markers', marker_symbol='star', marker_size=15,line=dict(width=2,
                                            color='Yellow')))
        st.plotly_chart(monte_fig)

        # Optimized Portfolio
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)

        # Calculate minimum volatility portfolio
        ef_min_volatility = EfficientFrontier(mu, S)
        min_vol_weights = ef_min_volatility.min_volatility()
        min_vol_performance = ef_min_volatility.portfolio_performance()
        min_volatility = min_vol_performance[1]

        # Calculate maximum Sharpe ratio portfolio
        ef_max_sharpe = EfficientFrontier(mu, S)
        max_sharpe_weights = ef_max_sharpe.max_sharpe()
        max_sharpe_performance = ef_max_sharpe.portfolio_performance()
        max_sharpe_volatility = max_sharpe_performance[1]

        # Adjust portfolio optimization based on risk tolerance and investment period
        try:
            if risk_tolerance <= 3:
                # Low risk tolerance: minnuimize volatility
                raw_weights = ef.min_volatility()
            elif risk_tolerance >= 8:
                # High risk tolerance: maximize Sharpe ratio
                raw_weights = ef.max_sharpe()
            else:
                # Intermediate risk tolerance: target volatility
                # Linear interpolation between minimum and maximum volatility
                target_volatility = min_volatility + (max_sharpe_volatility - min_volatility) * ((risk_tolerance - 3) / 4)

                # Adjust target volatility based on investment period
                adjustment_factor = max(0.5, investment_period / 30)  # Scale down volatility for shorter investment periods
                target_volatility *= adjustment_factor

                raw_weights = ef.efficient_risk(target_volatility)
        except ValueError as e:
            # In case of error, fallback to minimum volatility portfolio
            st.error(f"Error optimizing portfolio: {e}")
            raw_weights = min_vol_weights

        cleaned_weights = ef.clean_weights()

        # Ensure the weights sum to 1 (or 100%)
        weight_sum = sum(cleaned_weights.values())
        if weight_sum != 1:
            cleaned_weights = {k: v / weight_sum for k, v in cleaned_weights.items()}

        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        optimized_weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])

        st.subheader("Optimized Portfolio Weights")
        st.dataframe(optimized_weights_df)

        # Plot pie chart of the optimized portfolio allocation
        st.subheader("Optimized Portfolio Allocation")
        fig_pie = px.pie(optimized_weights_df, values='Weight', names=optimized_weights_df.index, title='Optimized Portfolio Allocation')
        fig_pie.update_traces(hole=.4, hoverinfo="label+percent+name")  # Convert to donut chart
        st.plotly_chart(fig_pie)

        # Hide Streamlit style
        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)