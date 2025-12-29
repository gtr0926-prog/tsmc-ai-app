import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, accuracy_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --- 1. 設定與標題 ---
st.set_page_config(page_title="台積電 AI 戰略指揮所", layout="wide")

# 新增：強制設定深色背景 CSS
st.markdown("""
<style>
    /* 強制主畫面深色背景 */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* 強制側邊欄深色背景 */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    /* 強制文字顏色為白色 */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li {
        color: #fafafa !important;
    }
    /* 修正 Metrics 數值顏色 */
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: white;'>🚀 台積電 (2330.TW) AI 戰略指揮所</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color:#1e201f;padding:15px;border-radius:10px;margin-bottom:20px;border-left: 5px solid #ff9800; color: white;">
    <strong>🤖 AI 戰略升級 (v2.0)：</strong> <br>
    1. <b>調節指標 (Adjust)</b>：訊號由「賣出」改為「調節」，建議逢高減碼或獲利了結，保留核心持股。<br>
    2. <b>資金動能指標</b>：新增成交量柱狀圖，以此觀察外資與主力進出的熱度。<br>
    3. <b>AI 趨勢線</b>：持續追蹤合理股價乖離率。
</div>
""", unsafe_allow_html=True)

# --- 2. 側邊欄參數 ---
st.sidebar.markdown("<h2 style='color: white;'>⚙️ 參數微調</h2>", unsafe_allow_html=True)

period_days = st.sidebar.slider("訓練數據長度 (天)", 500, 3000, 1000)
threshold = st.sidebar.slider("上漲判定門檻 (%)", 0.0, 2.0, 0.0, step=0.1)

# --- 3. 數據處理核心函數 ---
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

@st.cache_data(ttl=3600)
def get_advanced_data(period_days):
    tw = yf.Ticker("2330.TW").history(period=f"{period_days+150}d")
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

with st.spinner('正在部署 AI 模型與下載數據...'):
    try:
        raw_df = get_advanced_data(period_days)
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        st.stop()

# --- 4. 特徵工程 ---
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
    
    # Target 1: 分類 (漲跌)
    next_return = data['TW_Close'].shift(-1) / data['TW_Close'] - 1
    data['Target_Class'] = (next_return > (threshold/100)).astype(int)
    
    # Target 2: 回歸 (股價)
    data['Target_Price'] = data['TW_Close'].shift(-1)
    
    return data.dropna()

df_ready = engineer_features(raw_df)

# --- 5. 雙模型訓練 ---
features = ['TW_Close', 'TW_Volume', 'US_Close', 'US_Return', 'SOX_Return', 'RSI', 'MACD', 'MACD_Signal', 'MA_5', 'MA_20', 'ADR_Premium']

X = df_ready[features]
y_class = df_ready['Target_Class']
y_price = df_ready['Target_Price']

split = len(df_ready) - 150
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_class_train, y_class_test = y_class.iloc[:split], y_class.iloc[split:]
y_price_train, y_price_test = y_price.iloc[:split], y_price.iloc[split:]

clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42)
clf.fit(X_train, y_class_train)

reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
reg.fit(X_train, y_price_train)

# --- 6. 預測與訊號生成 ---
pred_classes = clf.predict(X_test)
pred_prices = reg.predict(X_test)

series_pred = pd.Series(pred_classes, index=X_test.index)
signals = series_pred.diff()

# 定義訊號
entry_points = series_pred[signals == 1].index
adjust_points = series_pred[signals == -1].index 

entry_prices = df_ready.loc[entry_points]['TW_Close']
adjust_prices = df_ready.loc[adjust_points]['TW_Close']

# --- 7. 未來預測 ---
last_known_data = X.iloc[[-1]].copy()
future_prices = []
future_dates = []
current_date = last_known_data.index[0]

temp_X = last_known_data.copy()
for i in range(1, 4):
    next_price = reg.predict(temp_X)[0]
    future_prices.append(next_price)
    next_date = current_date + timedelta(days=i)
    if next_date.weekday() >= 5: next_date += timedelta(days=2)
    future_dates.append(next_date)
    current_date = next_date
    temp_X['TW_Close'] = next_price 
    temp_X['MA_5'] = (temp_X['MA_5'] * 4 + next_price) / 5

# --- 8. 視覺化儀表板 ---
accuracy = accuracy_score(y_class_test, pred_classes)
rmse = np.sqrt(mean_squared_error(y_price_test, pred_prices))

col1, col2, col3, col4 = st.columns(4)
col1.metric("目前股價", f"{raw_df['TW_Close'].iloc[-1]:.0f}")
col2.metric("趨勢預測準度", f"{accuracy:.1%}")
col3.metric("AI 目標價誤差", f"±{rmse:.1f} 元")

next_class = clf.predict(X.iloc[[-1]])[0]
next_price_pred = reg.predict(X.iloc[[-1]])[0]

status = "🚀 進場/續抱" if next_class == 1 else "⚠️ 建議調節"
color = "red" if next_class == 1 else "orange"

with col4:
    st.markdown(f"### 明日策略: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.caption(f"AI 目標價: {next_price_pred:.1f}")

st.divider()

# --- 繪圖核心 (使用 plotly_dark 模板) ---
st.subheader("📊 AI 戰略地圖：調節訊號與資金動能")

# 建立雙軸圖表 (Row 1: 股價, Row 2: 成交量)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3],
    subplot_titles=("股價趨勢與 AI 訊號", "成交量 (資金動能)")
)

# 1. 真實股價 (上圖)
fig.add_trace(go.Scatter(
    x=df_ready.index[split:], 
    y=df_ready['TW_Close'].iloc[split:], 
    mode='lines', 
    name='真實股價', 
    line=dict(color='#2962ff', width=2),
    opacity=0.8
), row=1, col=1)

# 2. AI 趨勢線 (上圖)
fig.add_trace(go.Scatter(
    x=df_ready.index[split:], 
    y=pred_prices, 
    mode='lines', 
    name='AI 合理趨勢線', 
    line=dict(color='#ff6d00', width=2)
), row=1, col=1)

# 3. 買進訊號 (上圖)
fig.add_trace(go.Scatter(
    x=entry_points, 
    y=entry_prices * 0.98, 
    mode='markers', 
    name='AI 買進訊號 (Entry)', 
    marker=dict(color='#00c853', size=12, symbol='triangle-up'),
    text='買進',
    hoverinfo='x+y+name'
), row=1, col=1)

# 4. 調節訊號 (Adjust) (上圖)
fig.add_trace(go.Scatter(
    x=adjust_points, 
    y=adjust_prices * 1.02, 
    mode='markers', 
    name='AI 調節訊號 (Adjust)', 
    marker=dict(color='#ff9100', size=12, symbol='circle-dot'), 
    text='調節 (減碼)',
    hoverinfo='x+y+name'
), row=1, col=1)

# 5. 未來預測 (上圖)
fig.add_trace(go.Scatter(
    x=[df_ready.index[-1]] + future_dates,
    y=[df_ready['TW_Close'].iloc[-1]] + future_prices,
    mode='lines+markers',
    name='未來 3 日預測',
    line=dict(color='#d500f9', width=3, dash='dot'),
    marker=dict(size=8)
), row=1, col=1)

# 6. 成交量柱狀圖 (下圖)
volume_colors = ['#ef5350' if c >= o else '#26a69a' for c, o in zip(df_ready['TW_Close'].iloc[split:], df_ready['TW_Open'].iloc[split:])]

fig.add_trace(go.Bar(
    x=df_ready.index[split:],
    y=df_ready['TW_Volume'].iloc[split:],
    name='成交量',
    marker_color=volume_colors,
    opacity=0.8
), row=2, col=1)

# 修改圖表模板為 dark
fig.update_layout(
    height=700,
    hovermode="x unified",
    template="plotly_dark", # 關鍵修改：使用深色圖表背景
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=50, t=50, b=50),
    plot_bgcolor='rgba(0,0,0,0)', # 讓圖表背景透明以顯示網頁背景
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

st.info(f"""
    **📊 訊號統計：**
    - **買進 (Entry)**: {len(entry_points)} 次 - 趨勢翻多，建議進場。
    - **調節 (Adjust)**: {len(adjust_points)} 次 - 趨勢轉弱，建議**部分獲利了結**或減碼，保留核心部位。
    - **成交量指標**: 下方柱狀圖紅色代表買盤力道強，綠色代表賣盤力道強，可視為法人資金動向參考。
""")


