import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import plotly.graph_objects as go

# --- 1. 網頁設定 ---
st.set_page_config(page_title="台積電 AI 預測儀表板", layout="wide")
st.title("📈 台積電 (2330.TW) 股價趨勢預測")
st.markdown("### 真實數據版")
st.markdown("此應用程式使用 **yfinance** 下載即時數據，並使用 **Random Forest** 演算法進行隔日漲跌預測。")

# --- 2. 側邊欄設定 (參數調整) ---
st.sidebar.header("⚙️ 模型參數設定")
period_days = st.sidebar.slider("歷史數據長度 (天)", 200, 2000, 500)
ma_short = st.sidebar.slider("短期均線 (MA)", 3, 20, 5)
ma_long = st.sidebar.slider("長期均線 (MA)", 10, 60, 20)

# --- 3. 數據獲取與處理 ---
@st.cache_data(ttl=3600) # 快取數據 1 小時
def get_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{int(period*1.5)}d") 
    return df

with st.spinner('正在從 Yahoo Finance 下載台積電最新數據...'):
    try:
        df = get_data("2330.TW", period_days)
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        st.stop()

if len(df) == 0:
    st.error("無法下載數據，請檢查網路連線。")
    st.stop()

# --- 4. 特徵工程 ---
def prepare_data(data, short_window, long_window):
    df = data.copy()
    
    # 計算技術指標
    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_window).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    
    # 目標: 明日收盤 > 今日收盤 (1=漲, 0=跌)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    return df

df_processed = prepare_data(df, ma_short, ma_long)

# --- 5. 模型訓練 ---
features = ['Close', 'Volume', 'MA_Short', 'MA_Long', 'Return', 'Volatility']
X = df_processed[features]
y = df_processed['Target']

# 切分數據 (保留最後 100 天做驗證)
split = len(df_processed) - 100
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# 測試集評估
preds = model.predict(X_test)
accuracy = precision_score(y_test, preds, zero_division=0)

# --- 6. 未來預測 ---
last_row = X.iloc[[-1]]
next_day_prediction = model.predict(last_row)[0]
next_day_proba = model.predict_proba(last_row)[0]

# --- 7. 視覺化顯示 ---
col1, col2, col3, col4 = st.columns(4)
latest_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100

col1.metric("最新收盤價", f"{latest_price:.1f}", f"{price_change:.1f} ({pct_change:.2f}%)")
col2.metric("模型準確率 (Precision)", f"{accuracy:.2%}", "近100天回測")

pred_label = "📈 看漲 (Up)" if next_day_prediction == 1 else "📉 看跌 (Down)"
confidence = next_day_proba[next_day_prediction]

with col3:
    st.metric("AI 預測下個交易日", pred_label)
with col4:
    st.metric("信心指數", f"{confidence:.2%}")

st.divider()

# 繪圖
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='收盤價', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=df.index, y=df_processed['MA_Short'], mode='lines', name=f'MA {ma_short}', line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df_processed['MA_Long'], mode='lines', name=f'MA {ma_long}', line=dict(color='green', width=1)))

fig.update_layout(title="台積電歷史股價與均線", xaxis_title="日期", yaxis_title="價格", height=500, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

