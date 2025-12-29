import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ComposedChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Target, ArrowUpCircle, AlertCircle, PieChart, DollarSign } from 'lucide-react';

// 模擬數據生成 (含訊號與外資量)
const generateMockData = (days) => {
  const data = [];
  let price = 980; 
  let aiPrice = 980;
  const now = new Date();
  let holding = false; 

  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - (days - i));
    
    const change = (Math.random() - 0.48) * 25; 
    price += change;
    if (price < 0) price = 10;
    
    aiPrice = aiPrice * 0.9 + price * 0.1 + (Math.random() - 0.5) * 5;

    let foreignVol = (Math.random() * 50000) - 20000; 
    if (change > 0) foreignVol += 15000; 
    if (change < 0) foreignVol -= 15000;

    let action = null;
    let signalMarker = null;

    const ma5 = price * 0.98 + (Math.random() * 10);
    const ma20 = price * 0.95 + (Math.random() * 15);

    if (ma5 > ma20 && !holding) {
        action = 'Buy';
        holding = true;
        signalMarker = price * 0.98; 
    } else if (ma5 < ma20 && holding) {
        action = 'Adjust';
        holding = false;
        signalMarker = price * 1.02; 
    }

    data.push({
      date: date.toISOString().split('T')[0],
      Close: price,
      AIPrediction: aiPrice,
      Action: action,
      SignalPrice: signalMarker,
      ForeignVol: foreignVol
    });
  }
  return data;
};

const CustomizedDot = (props) => {
  const { cx, cy, payload } = props;
  
  if (payload.Action === 'Buy') {
    return (
      <svg x={cx - 10} y={cy + 10} width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="#4ade80" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="19" x2="12" y2="5"></line>
        <polyline points="5 12 12 5 19 12"></polyline>
      </svg>
    );
  }
  
  if (payload.Action === 'Adjust') {
    return (
      <svg x={cx - 10} y={cy - 25} width={20} height={20} viewBox="0 0 24 24" fill="#fb923c" stroke="#fb923c" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="6"></circle>
      </svg>
    );
  }
  return null;
};

export default function StockDashboard() {
  const [periodDays, setPeriodDays] = useState(150);
  const [chartData, setChartData] = useState([]);
  const [lastSignal, setLastSignal] = useState(null);

  useEffect(() => {
    const data = generateMockData(periodDays);
    setChartData(data);
    const signals = data.filter(d => d.Action);
    if(signals.length > 0) setLastSignal(signals[signals.length - 1]);
  }, [periodDays]);

  const currentPrice = chartData.length > 0 ? chartData[chartData.length - 1].Close : 0;
  const aiTarget = chartData.length > 0 ? chartData[chartData.length - 1].AIPrediction : 0;

  return (
    // 修改：背景改為深色 bg-slate-900，文字改為淺色 text-slate-100
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-4 md:p-8">
      <header className="mb-8 border-b border-slate-700 pb-4">
        {/* 修改：標題文字白色 */}
        <h1 className="text-3xl font-bold text-white flex items-center gap-2">
          <Target className="text-purple-400" />
          台積電 AI 戰略指揮所 (v2.0)
        </h1>
        <p className="text-slate-400 mt-1">AI 趨勢線、調節訊號與外資動能分析</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
        {/* 修改：卡片背景改為深灰色 bg-slate-800，邊框 border-slate-700 */}
        <div className="bg-slate-800 p-5 rounded-xl shadow-sm border border-slate-700">
            <div className="text-sm text-slate-400 mb-1">最新收盤價</div>
            <div className="text-3xl font-bold text-white">{currentPrice.toFixed(0)}</div>
        </div>
        <div className="bg-slate-800 p-5 rounded-xl shadow-sm border border-slate-700">
            <div className="text-sm text-slate-400 mb-1">AI 趨勢線目標</div>
            <div className="text-3xl font-bold text-orange-400">{aiTarget.toFixed(0)}</div>
            <div className="text-xs text-slate-500">合理價格區間</div>
        </div>
        <div className="bg-slate-800 p-5 rounded-xl shadow-sm border border-slate-700">
            <div className="text-sm text-slate-400 mb-1">最新 AI 建議</div>
            <div className="flex items-center gap-2">
                {lastSignal?.Action === 'Buy' ? <ArrowUpCircle className="text-green-400"/> : <AlertCircle className="text-orange-400"/>}
                <div className={`text-2xl font-bold ${lastSignal?.Action === 'Buy' ? 'text-green-400' : 'text-orange-400'}`}>
                    {lastSignal?.Action === 'Buy' ? '買進 (Buy)' : '調節 (Adjust)'}
                </div>
            </div>
            <div className="text-xs text-slate-500">{lastSignal?.date} 觸發</div>
        </div>
        
        <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col justify-center">
            {/* 修改：文字改為淺色 */}
            <label className="text-sm font-medium text-slate-300 mb-2">觀察區間: {periodDays} 天</label>
            <input 
                type="range" min="50" max="300" step="10" 
                value={periodDays}
                onChange={(e) => setPeriodDays(Number(e.target.value))}
                className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
        </div>
      </div>

      {/* 主圖表區 - 背景深色 */}
      <div className="space-y-4">
        <div className="bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-700 h-[450px]">
          <h3 className="text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
              <Activity size={20}/> 股價趨勢與調節點
          </h3>
          <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  {/* 修改：網格線改暗 stroke="#334155" */}
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                  <XAxis dataKey="date" tick={{fontSize: 12, fill: '#94a3b8'}} tickFormatter={(str) => str.slice(5)} minTickGap={30}/>
                  <YAxis domain={['auto', 'auto']} orientation="right" tick={{fontSize: 12, fill: '#94a3b8'}}/>
                  {/* 修改：Tooltip 背景深色 */}
                  <Tooltip contentStyle={{borderRadius: '8px', backgroundColor: '#1e293b', border: '1px solid #475569', color: '#f8fafc'}} />
                  <Legend wrapperStyle={{ color: '#cbd5e1' }}/>
                  
                  <Area type="monotone" dataKey="Close" stroke="#3b82f6" fill="url(#colorClose)" fillOpacity={0.1} strokeWidth={2} name="真實股價" />
                  <defs>
                      <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                  </defs>
                  <Line type="monotone" dataKey="AIPrediction" stroke="#f97316" strokeWidth={2} name="AI 趨勢線" dot={false} />
                  <Line type="monotone" dataKey="Close" stroke="none" dot={<CustomizedDot />} activeDot={false} legendType="none" />
              </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-700 h-[250px]">
           <h3 className="text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
              <DollarSign size={20}/> 外資進出動能 (模擬)
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                <XAxis dataKey="date" tick={false} />
                <YAxis orientation="right" fontSize={12} tick={{fill: '#94a3b8'}}/>
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{borderRadius: '8px', backgroundColor: '#1e293b', border: '1px solid #475569', color: '#f8fafc'}} />
                <ReferenceLine y={0} stroke="#64748b" />
                <Bar dataKey="ForeignVol" name="外資買賣超">
                  {chartData.map((entry, index) => (
                    <cell key={`cell-${index}`} fill={entry.ForeignVol > 0 ? '#ef4444' : '#22c55e'} />
                  ))}
                </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-6 flex gap-6 justify-center text-sm text-slate-400">
        <div className="flex items-center gap-2">
            <ArrowUpCircle size={16} className="text-green-400" /> 綠色箭頭：AI 建議進場 (Buy)
        </div>
        <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-orange-400"></div> 橘色圓點：AI 建議調節 (Adjust/Reduce)
        </div>
        <div className="flex items-center gap-2">
             <div className="w-3 h-3 bg-red-500 rounded-sm"></div> 紅柱：外資買超
        </div>
      </div>
    </div>
  );
}


