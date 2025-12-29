import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import twstock # 新增：引入台灣股市套件
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
    <strong>🤖 AI 戰略升級 (v3.1 - TWSE 即時版)：</strong> <br>
    1. <b>混合數據源</b>：歷史數據使用 Yahoo (完整美股資料)，即時報價使用 <b>TWSE 證交所 (毫秒級更新)</b>。<br>
    2. <b>精準決策</b>：利用即時股價進行 AI 運算，消除延遲誤差。<br>
    3. <b>資金控管</b>：自動計算建議進出張數。
</div>
""", unsafe_allow_html=True)

# --- 2. 側邊欄參數 ---
st.sidebar.markdown("<h2 style='color: white;'>🔄 數據更新</h2>", unsafe_allow_html=True)
if st.sidebar.button('🔄 立即刷新 TWSE 報價'):
    st.cache_data.clear()

st.sidebar.markdown("<h2 style='color: white;'>💰 資金與部位設定</h2>", unsafe_allow_html=True)
total_capital = st.sidebar.number_input("總操作資金 (TWD)", min_value=10000, value=1000000, step=10000)
current_shares = st.sidebar.number_input("目前持有股數 (Shares)", min_value=0, value=0, step=1000, help="1張 = 1000股")
risk_per_trade = st.sidebar.slider("單筆投入資金比例 (%)", 10, 100, 30)
adjust_ratio = st.sidebar.slider("調節賣出比例 (%)", 10, 100, 50)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color: white;'>⚙️ 模型參數</h2>", unsafe_allow_html=True)
period_days = st.sidebar.slider("訓練數據長度 (天)", 500, 3000, 1000)
threshold = st.sidebar.slider("上漲判定門檻 (%)", 0.0, 2.0, 0.0, step=0.1)

# --- 3. 數據抓取函數 (混合模式) ---

# A. 即時報價 (來源：TWSE / twstock)
def get_twse_realtime():
    try:
        # 抓取 2330 即時資料
        stock = twstock.realtime.get('2330')
        if stock['success']:
            # 確保抓到的是有效數字 (有時候開盤前會是 -)
            price = stock['realtime']['latest_trade_price']
            if price == '-' or price is None:
                return None, None
            return float(price), stock['info']['time']
        else:
            return None, None
    except Exception as e:
        return None, None

# B. 歷史與關聯數據 (來源：Yahoo Finance)
@st.cache_data(ttl=300) # 歷史數據不需要太常更新，5分鐘一次即可
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

# --- 4. 數據準備流程 ---
status_placeholder = st.empty()
status_placeholder.info("⏳ 正在連線 Yahoo Finance 下載 AI 訓練數據...")

# 1. 獲取歷史數據 (用於訓練 AI)
try:
    raw_df = get_historical_data(period_days)
    if raw_df is None:
        st.error("無法下載歷史數據。")
        st.stop()
except Exception as e:
    st.error(f"數據下載錯誤: {e}")
    st.stop()

# 2. 獲取 TWSE 即時報價 (用於決策)
status_placeholder.info("⚡ 正在連線 TWSE 台灣證交所獲取即時報價...")
realtime_price, realtime_time = get_twse_realtime()

# 決定使用哪個價格作為「最新價格」
if realtime_price is not None:
    latest_price = realtime_price
    price_source = f"TWSE 證交所即時盤 (更新: {realtime_time})"
    source_color = "#00c853" # Green
else:
    latest_price = raw_df['TW_Close'].iloc[-1]
    price_source = f"Yahoo Finance (延遲報價)"
    source_color = "#ff9100" # Orange

status_placeholder.empty() # 清除讀取訊息

# --- 5. 特徵工程與模型訓練 ---
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
    data['Return'] = data['TW_Close'].pct_change()
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
y_class = df_ready['Target_Class']
y_price = df_ready['Target_Price']

# 訓練模型
clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42)
clf.fit(X, y_class) # 使用全部數據訓練以求最新預測

reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
reg.fit(X, y_price)

# --- 6. 整合即時數據進行預測 ---
# 我們需要構建一個包含「最新即時股價」的特徵列
last_row = X.iloc[[-1]].copy()

# 【關鍵步驟】將 TWSE 的即時股價填入特徵中
# 注意：這裡我們假設其他指標(如美股)暫時不變，只更新台積電價格來觀察 AI 反應
last_row['TW_Close'] = latest_price 
# 重新計算均線 (約略估計)
last_row['MA_5'] = (last_row['MA_5'] * 4 + latest_price) / 5

# AI 進行預測
next_class = clf.predict(last_row)[0] 
next_price_pred = reg.predict(last_row)[0]

# 計算決策
suggested_action = ""
suggested_shares = 0
estimated_amount = 0

if next_class == 1: 
    suggested_action = "加碼 / 買進"
    budget = total_capital * (risk_per_trade / 100)
    suggested_shares = int(budget // latest_price)
    estimated_amount = suggested_shares * latest_price
    action_color = "#00c853" 
else: 
    suggested_action = "調節 / 賣出"
    if current_shares > 0:
        suggested_shares = int(current_shares * (adjust_ratio / 100))
        estimated_amount = suggested_shares * latest_price
    else:
        suggested_shares = 0
        estimated_amount = 0
    action_color = "#ff9100" 

# --- 7. 視覺化儀表板 ---
st.markdown(f"<div style='text-align: right; color: {source_color}; font-size: 0.9em; font-weight: bold;'>● 資料來源: {price_source}</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("最新股價", f"{latest_price:.1f}")
col2.metric("AI 目標價", f"{next_price_pred:.1f}", f"{next_price_pred - latest_price:.1f}")
col3.metric("建議操作", suggested_action)
col4.metric("預估股數", f"{suggested_shares:,} 股")

st.markdown(f"""
<div style="background-color: #262730; padding: 20px; border-radius: 10px; margin-top: 10px; border: 1px solid #444;">
    <h3 style="margin-top: 0; color: {action_color} !important;">📝 AI 交易決策書 (TWSE 即時版)</h3>
    <div style="display: flex; justify-content: space-between; font-size: 1.1em;">
        <span>策略訊號：<strong style="color: {action_color};">{suggested_action}</strong></span>
        <span>當前持倉：<strong>{current_shares:,} 股</strong></span>
    </div>
    <hr style="border-color: #555;">
    <div style="display: flex; justify-content: space-between; font-size: 1.1em;">
        <span>建議交易股數：</span>
        <strong style="font-size: 1.3em; color: white;">{suggested_shares:,} 股</strong>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 1.1em; margin-top: 5px;">
        <span>預估交易金額：</span>
        <span style="color: #ccc;">$ {estimated_amount:,.0f} TWD</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- 繪圖核心 ---
plot_df = df_ready.iloc[-200:]
split_idx = len(plot_df) - 50 

# 生成歷史訊號 (僅供回測參考)
subset_pred = clf.predict(X.iloc[-200:])
subset_series = pd.Series(subset_pred, index=plot_df.index)
subset_signals = subset_series.diff()

p_entry_pts = subset_series[subset_signals == 1].index
p_adjust_pts = subset_series[subset_signals == -1].index
p_entry_prices = plot_df.loc[p_entry_pts]['TW_Close']
p_adjust_prices = plot_df.loc[p_adjust_pts]['TW_Close']

st.subheader("📊 近期走勢與資金動能")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
    subplot_titles=("股價趨勢與 AI 訊號", "成交量")
)

fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['TW_Close'], mode='lines', name='真實股價', 
    line=dict(color='#2962ff', width=2)
), row=1, col=1)

full_reg_pred = reg.predict(X.iloc[-200:])
fig.add_trace(go.Scatter(
    x=plot_df.index, y=full_reg_pred, mode='lines', name='AI 趨勢線', 
    line=dict(color='#ff6d00', width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=p_entry_pts, y=p_entry_prices * 0.98, mode='markers', name='買進 (Entry)', 
    marker=dict(color='#00c853', size=12, symbol='triangle-up')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=p_adjust_pts, y=p_adjust_prices * 1.02, mode='markers', name='調節 (Adjust)', 
    marker=dict(color='#ff9100', size=12, symbol='circle-dot')
), row=1, col=1)

colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(plot_df['TW_Close'], plot_df['TW_Open'])]
fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['TW_Volume'], name='成交量', marker_color=colors
), row=2, col=1)

fig.update_layout(
    height=600, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.02, x=1)
)

st.plotly_chart(fig, use_container_width=True)


