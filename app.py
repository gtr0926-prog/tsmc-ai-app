import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, accuracy_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import time

# --- 1. 設定與標題 ---
st.set_page_config(page_title="台積電 AI 戰略指揮所", layout="wide")

# 強制設定深色背景 CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    h1, h2, h3, h4, h5, h6, p, span, div, label, li { color: #fafafa !important; }
    [data-testid="stMetricValue"] { color: #fafafa !important; }
    div.stButton > button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #ff4b4b; color: white; 
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: white;'>🚀 台積電 (2330.TW) AI 戰略指揮所</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color:#1e201f;padding:15px;border-radius:10px;margin-bottom:20px;border-left: 5px solid #ff9800; color: white;">
    <strong>🤖 AI 戰略升級 (v3.3 - 獲利調節優化)：</strong> <br>
    1. <b>5% 獲利調節</b>：設定獲利達 5% 以上且 AI 看空時才進行調節，減少交易成本。<br>
    2. <b>防頻繁交易</b>：過濾微幅震盪訊號，避免成本加劇。<br>
    3. <b>即時連動</b>：使用 TWSE 即時報價計算獲利趴數。
</div>
""", unsafe_allow_html=True)

# --- 2. 側邊欄參數 ---
st.sidebar.markdown("<h2 style='color: white;'>🔄 數據更新</h2>", unsafe_allow_html=True)
if st.sidebar.button('🔄 立即刷新即時數據'):
    st.cache_data.clear()

st.sidebar.markdown("<h2 style='color: white;'>💰 資金與持倉設定</h2>", unsafe_allow_html=True)
total_capital = st.sidebar.number_input("總操作資金 (TWD)", min_value=10000, value=1000000, step=10000)
current_shares = st.sidebar.number_input("目前持有股數", min_value=0, value=1000, step=1000)
avg_cost = st.sidebar.number_input("平均買進成本 (TWD)", min_value=1.0, value=1000.0, step=0.5)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color: white;'>🛡️ 調節策略設定</h2>", unsafe_allow_html=True)
take_profit_percent = st.sidebar.slider("目標獲利調節門檻 (%)", 1.0, 20.0, 5.0, help="股價超過成本此比例時才會考慮調節")
adjust_ratio = st.sidebar.slider("單次調節賣出比例 (%)", 10, 100, 30, help="符合條件時建議賣出的庫存比例")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color: white;'>⚙️ 模型參數</h2>", unsafe_allow_html=True)
period_days = st.sidebar.slider("訓練數據長度 (天)", 500, 3000, 1000)
threshold = st.sidebar.slider("上漲判定門檻 (%)", 0.0, 2.0, 0.0, step=0.1)

# --- 3. 數據抓取函數 ---

def get_twse_realtime():
    try:
        stock = twstock.realtime.get('2330')
        if stock['success']:
            price = stock['realtime']['latest_trade_price']
            if price == '-' or price is None:
                return None, None
            return float(price), stock['info']['time']
        else:
            return None, None
    except:
        return None, None

@st.cache_data(ttl=300)
def get_historical_data(period_days):
    tw = yf.Ticker("2330.TW").history(period=f"{period_days+150}d", auto_adjust=False)
    if len(tw) == 0: return None
    tw = tw[['Close', 'Open', 'High', 'Low', 'Volume']]
    tw.columns = [f"TW_{col}" for col in tw.columns]
    
    us = yf.Ticker("TSM").history(period=f"{period_days+150}d")
    us = us[['Close', 'Volume']]
    us.columns = [f"US_{col}" for col in us.columns]
    
    sox = yf.Ticker("^SOX").history(period=f"{period_days+150}d")
    sox = sox[['Close']]
    sox.columns = [f"SOX_{col}" for col in sox.columns]
    
    tw.index = tw.index.tz_localize(None)
    us.index = us.index.tz_localize(None)
    sox.index = sox.index.tz_localize(None)
    
    df = pd.concat([tw, us, sox], axis=1).dropna()
    return df

# --- 4. 數據準備 ---
with st.spinner('同步 AI 模型數據與 TWSE 即時報價...'):
    raw_df = get_historical_data(period_days)
    realtime_price, realtime_time = get_twse_realtime()

if raw_df is None:
    st.error("無法下載數據")
    st.stop()

latest_price = realtime_price if realtime_price else raw_df['TW_Close'].iloc[-1]
price_source = f"TWSE 即時 ({realtime_time})" if realtime_price else "Yahoo 延遲"

# --- 5. 特徵工程與訓練 ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def engineer_features(df):
    data = df.copy()
    data['ADR_Premium'] = (data['US_Close'] * 31) / 5 - data['TW_Close'] 
    data['US_Return'] = data['US_Close'].pct_change()
    data['SOX_Return'] = data['SOX_Close'].pct_change()
    data['RSI'] = calculate_rsi(data['TW_Close'])
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['TW_Close'])
    data['MA_5'] = data['TW_Close'].rolling(window=5).mean()
    data['MA_20'] = data['TW_Close'].rolling(window=20).mean()
    next_return = data['TW_Close'].shift(-1) / data['TW_Close'] - 1
    data['Target_Class'] = (next_return > (threshold/100)).astype(int)
    data['Target_Price'] = data['TW_Close'].shift(-1)
    return data.dropna()

df_ready = engineer_features(raw_df)
features = ['TW_Close', 'TW_Volume', 'US_Close', 'US_Return', 'SOX_Return', 'RSI', 'MACD', 'MACD_Signal', 'MA_5', 'MA_20', 'ADR_Premium']
X = df_ready[features]

clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df_ready['Target_Class'])
reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, df_ready['Target_Price'])

# --- 6. 決策邏輯 (包含 5% 獲利檢查) ---
last_row = X.iloc[[-1]].copy()
last_row['TW_Close'] = latest_price 
next_class = clf.predict(last_row)[0] 
next_price_pred = reg.predict(last_row)[0]

# 計算當前獲利趴數
current_profit_pct = ((latest_price - avg_cost) / avg_cost) * 100
target_price_for_tp = avg_cost * (1 + take_profit_percent / 100)

suggested_action = "觀望 / 續抱"
suggested_shares = 0
action_color = "#aaaaaa"
reason = "AI 訊號中性或獲利未達門檻"

if next_class == 1:
    suggested_action = "加碼 / 買進"
    # 簡單預算邏輯：使用 30% 可用資金
    suggested_shares = int((total_capital * 0.3) // latest_price)
    action_color = "#00c853"
    reason = "AI 模型看好明日走勢"
else:
    # 只有在獲利 > 門檻時，才進行調節
    if current_shares > 0 and current_profit_pct >= take_profit_percent:
        suggested_action = "分批調節"
        suggested_shares = int(current_shares * (adjust_ratio / 100))
        action_color = "#ff9100"
        reason = f"已達 {take_profit_percent}% 獲利目標且 AI 看空，執行減碼以節省成本"
    elif current_shares > 0 and next_class == 0:
        suggested_action = "續抱 (未達停利)"
        suggested_shares = 0
        action_color = "#2962ff"
        reason = f"雖然 AI 看空，但目前獲利僅 {current_profit_pct:.1f}%，未達調節門檻，避免頻繁交易"

# --- 7. 儀表板顯示 ---
st.markdown(f"<div style='text-align: right; color: #888;'>數據來源: {price_source}</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("即時股價", f"{latest_price:.1f}")
m2.metric("平均成本", f"{avg_cost:.1f}")
profit_display_color = "normal" if current_profit_pct >= 0 else "inverse"
m3.metric("目前獲利", f"{current_profit_pct:.2f}%", delta=f"目標 {take_profit_percent}%")
m4.metric("AI 目標價", f"{next_price_pred:.1f}")

st.markdown(f"""
<div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 10px solid {action_color};">
    <h2 style="margin: 0; color: {action_color} !important;">決策建議：{suggested_action}</h2>
    <p style="font-size: 1.2em; margin: 10px 0;">建議操作股數：<strong style="font-size: 1.5em; color: white;">{suggested_shares:,}</strong> 股</p>
    <div style="background: #1a1b21; padding: 10px; border-radius: 5px; color: #bbb;">
        <strong>💡 策略分析：</strong> {reason} <br>
        <strong>🎯 調節目標：</strong> 股價需達 <span style="color: #ff9100;">{target_price_for_tp:.1f}</span> 以上才進行分批調節。
    </div>
</div>
""", unsafe_allow_html=True)

# --- 8. 走勢圖 ---
plot_df = df_ready.iloc[-150:]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

# 股價與成本線
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['TW_Close'], name='歷史股價', line=dict(color='#2962ff')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=[avg_cost]*len(plot_df), name='我的成本', line=dict(color='white', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=[target_price_for_tp]*len(plot_df), name='調節門檻 (5%)', line=dict(color='#ff9100', dash='dot')), row=1, col=1)

# 成交量
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['TW_Volume'], name='成交量', marker_color='#444'), row=2, col=1)

fig.update_layout(height=500, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

