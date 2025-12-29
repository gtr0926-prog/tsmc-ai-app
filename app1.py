import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 初始化設定 ---
st.set_page_config(page_title="2330 AI 指揮所", layout="wide")

# 深色模式美化
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    .stMetric { background-color: #1e201f; padding: 10px; border-radius: 10px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. 側邊欄設定 ---
st.sidebar.title("💰 帳戶設定")
total_capital = st.sidebar.number_input("操作資金", value=1000000)
current_shares = st.sidebar.number_input("目前股數", value=0, step=1000)
avg_cost = st.sidebar.number_input("買進成本", value=1000.0)
take_profit_pct = st.sidebar.slider("停利門檻 (%)", 1.0, 10.0, 5.0)

# --- 3. 數據抓取 (增加防錯機制) ---
def get_safe_data():
    # 嘗試抓取即時價格
    rt_price = None
    try:
        # twstock 有時在非交易時間會噴錯，增加捕捉
        rt = twstock.realtime.get('2330')
        if rt['success']:
            p = rt['realtime']['latest_trade_price']
            if p != '-': rt_price = float(p)
    except:
        pass
    
    # 抓取歷史數據
    try:
        # 增加緩存以利載入速度
        df = yf.download("2330.TW TSM ^SOX", period="2y", interval="1d", progress=False)
        # 整理 Multi-index
        tw_close = df['Close']['2330.TW'].dropna()
        us_close = df['Close']['TSM'].dropna()
        sox_close = df['Close']['^SOX'].dropna()
        
        main_df = pd.DataFrame({
            'TW_Close': tw_close,
            'US_Close': us_close,
            'SOX_Close': sox_close
        }).dropna()
        return main_df, rt_price
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None

data, current_p = get_safe_data()

if data is not None:
    # 如果沒抓到即時價格，就用最後一筆收盤價
    display_p = current_p if current_p else data['TW_Close'].iloc[-1]
    
    # --- 4. 簡單特徵工程與模型 ---
    data['MA5'] = data['TW_Close'].rolling(5).mean()
    data['ADR_P'] = (data['US_Close'] * 31 / 5) - data['TW_Close']
    data['Target'] = (data['TW_Close'].shift(-1) > data['TW_Close']).astype(int)
    train = data.dropna()
    
    X = train[['TW_Close', 'MA5', 'ADR_P']]
    y = train['Target']
    
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    
    # 預測
    last_feat = X.iloc[[-1]].copy()
    last_feat['TW_Close'] = display_p # 置換成即時價格做判斷
    pred = model.predict(last_row=last_feat)[0]
    
    # --- 5. 決策邏輯 ---
    profit = (display_p - avg_cost) / avg_cost * 100
    
    st.title("🚀 TSMC AI 即時戰略")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("即時股價", f"{display_p:.1f}", f"{display_p - data['TW_Close'].iloc[-1]:.1f}")
    c2.metric("目前損益", f"{profit:.2f}%")
    
    if pred == 1:
        signal = "買進 / 持有"
        color = "#00c853"
        shares = int((total_capital * 0.3) // display_p)
    else:
        if profit >= take_profit_pct and current_shares > 0:
            signal = f"達標調節 (>{take_profit_pct}%)"
            color = "#ff9100"
            shares = int(current_shares * 0.5)
        else:
            signal = "續抱觀望"
            color = "#2962ff"
            shares = 0

    st.markdown(f"""
    <div style="padding:20px; border-radius:15px; background-color:{color}; color:white; text-align:center;">
        <h1 style="color:white !important;">建議操作：{signal}</h1>
        <h2>建議股數：{shares:,} 股</h2>
    </div>
    """, unsafe_allow_html=True)

