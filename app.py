import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indian Stock Predictor", layout="wide")
st.title("🇮🇳 Indian Stock Price Predictor")
st.write("Predict next-day stock price using Linear Regression (Safe Version)")

# Top 50 NSE companies with symbols
top_nse_companies = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFC.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "ITC Ltd": "ITC.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Nestle India": "NESTLEIND.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Tech Mahindra": "TECHM.NS",
    "Oil & Natural Gas": "ONGC.NS",
    "Grasim Industries": "GRASIM.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Wipro": "WIPRO.NS",
    "Coal India": "COALINDIA.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Dr Reddy's Labs": "DRREDDY.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Shree Cement": "SHREECEM.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "GAIL India": "GAIL.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "Cipla": "CIPLA.NS",
    "SBI Life": "SBILIFE.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Balkrishna Industries": "BALKRISIND.NS"
}

# Select company from dropdown
company_name = st.selectbox("Select a Company", list(top_nse_companies.keys()))
symbol = top_nse_companies[company_name]

if st.button("Predict"):
    try:
        # Fetch historical data
        df = yf.download(symbol, start="2020-01-01")
        if df.empty:
            st.error("No data found for this company. Try another one.")
            st.stop()
        
        # Show historical chart
        st.subheader(f"Historical Data for {company_name} ({symbol})")
        st.line_chart(df['Close'])

        # Clean data and prepare for Linear Regression
        df = df.reset_index()
        df = df[['Date', 'Close']].dropna()
        df['DateOrdinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)

        X = df['DateOrdinal'].values.reshape(-1,1)
        y = df['Close'].values

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next day safely
        last_date = df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        next_date_ord = np.array([[next_date.toordinal()]])
        next_price_array = model.predict(next_date_ord)

        # Convert to float safely
        next_price = float(np.squeeze(next_price_array))
        if np.isnan(next_price):
            st.error("Prediction resulted in NaN. Check the stock symbol or historical data.")
        else:
            st.success(f"Next day predicted price: ₹{next_price:.2f}")

        # Plot regression trend line
        plt.figure(figsize=(10,5))
        plt.scatter(df['Date'], y, label="Actual Price")
        plt.plot(df['Date'], model.predict(X), color='red', label="Trend Line")
        plt.title(f"{company_name} Price Trend")
        plt.xlabel("Date")
        plt.ylabel("Close Price (₹)")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error fetching data: {e}")