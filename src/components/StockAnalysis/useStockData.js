import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useState } from 'react';
import { generateAdvancedSummary, generateProfessionalReport, generateTradingStrategies } from './analysis/summary';
import { calculateBollingerBands, calculateKD, calculateMACD, calculateRSI } from './analysis/utils/indicators';
import { analyzeVolumeTrend } from './analysis/utils/volumeAnalysis';

const I18N = {
  'zh-hant': {
    httpError: (status) => `HTTP錯誤：${status}`,
    noDataOrError: '查無資料或 API 返回格式有誤',
    insufficientData: '分析資料不足',
    networkError: '網路或系統錯誤，請稍後再試'
  },
  'en': {
    httpError: (status) => `HTTP Error: ${status}`,
    noDataOrError: 'No data found or API response format is incorrect',
    insufficientData: 'Insufficient data for analysis',
    networkError: 'Network or system error, please try again later'
  },
  'ja': {
    httpError: (status) => `HTTPエラー：${status}`,
    noDataOrError: 'データが見つからない、またはAPI応答形式が不正です',
    insufficientData: '分析に十分なデータがありません',
    networkError: 'ネットワークまたはシステムエラー、後でもう一度お試しください'
  }
};

export function useStockData() {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  const [stockId, setStockId] = useState('2330');
  const [ohlcData, setOhlcData] = useState([]);
  const [kd, setKD] = useState(null);
  const [rsi, setRSI] = useState(null);
  const [bollinger, setBollinger] = useState(null);
  const [macd, setMACD] = useState(null);
  const [advancedAnalysis, setAdvancedAnalysis] = useState(null);
  const [professionalReport, setProfessionalReport] = useState('');
  const [numDays, setNumDays] = useState(0);
  const [marketView, setMarketView] = useState('');
  const [strategies, setStrategies] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rawData, setRawData] = useState([]);

  async function fetchData() {
    setLoading(true);
    setError(null);
    resetResults();

    try {
      const url = `https://api.docsaid.org/stocks/prices/?stock_id=${stockId}`;
      const res = await fetch(url);

      if (!res.ok) {
        setError(t.httpError(res.status));
        setLoading(false);
        return;
      }

      const data = await res.json();
      if (!data || Number(data.status) !== 200 || !Array.isArray(data.data) || data.data.length === 0) {
        setError(t.noDataOrError);
        setLoading(false);
        return;
      }

      const raw = data.data.sort((a,b) => new Date(a.date) - new Date(b.date));
      setRawData(raw);
      setNumDays(raw.length);

      const ohlc = raw.map(item => ({
        x: item.date,
        y: [item.open, item.max, item.min, item.close]
      }));
      setOhlcData(ohlc);

      // 計算各技術指標
      const kdResult = calculateKD(raw, { period: 9 });
      const rsiResult = calculateRSI(raw, { period: 14 });
      const bollResult = calculateBollingerBands(raw, { period: 20 });
      const macdResult = calculateMACD(raw, { short:12, long:26, signal:9 });

      setKD(kdResult);
      setRSI(rsiResult);
      setBollinger(bollResult);
      setMACD(macdResult);

      const analysis = generateAdvancedSummary(raw);
      if (!analysis) {
        setError(t.insufficientData);
        setLoading(false);
        return;
      }
      setAdvancedAnalysis(analysis);

      const volumeTrend = analyzeVolumeTrend(raw, currentLocale);
      const { marketView, strategies } = generateTradingStrategies(analysis, volumeTrend, currentLocale);
      setMarketView(marketView);
      setStrategies(strategies);

      const report = generateProfessionalReport(stockId, analysis, volumeTrend, marketView, strategies, currentLocale);
      setProfessionalReport(report);

      setError(null);

    } catch (e) {
      console.error(e);
      setError(t.networkError);
    } finally {
      setLoading(false);
    }
  }

  function resetResults() {
    setAdvancedAnalysis(null);
    setProfessionalReport('');
    setMarketView('');
    setStrategies([]);
    setOhlcData([]);
    setKD(null);
    setRSI(null);
    setBollinger(null);
    setMACD(null);
    setRawData([]);
  }

  return {
    stockId, setStockId,
    ohlcData,
    kd,
    rsi,
    bollinger,
    macd,
    advancedAnalysis,
    professionalReport,
    numDays,
    marketView,
    strategies,
    loading,
    error,
    fetchData,
    rawData
  }
}
