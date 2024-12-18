import { mean, std } from 'mathjs';

function ema(values, period) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const k = 2/(period+1);
  const arr = [values[0]];
  for (let i=1; i<values.length; i++){
    arr.push(values[i]*k + arr[i-1]*(1-k));
  }
  return arr;
}

export function calculateKD(data, {period=9}={}) {
  // KD 計算邏輯
  const K = [];
  const D = [];
  if (!Array.isArray(data) || data.length === 0) return { K:[], D:[] };

  data.forEach((d, i) => {
    const range = d.max - d.min;
    const kValue = (range === 0) ? (i>0?K[i-1]:50) : ((d.close - d.min)/range)*100;
    const prevK = (i>0 && K[i-1]!=null)?K[i-1]:50;
    const newK = (2/3)*prevK + (1/3)*kValue;
    K.push(newK);
  });
  K.forEach((k, i) => {
    const prevD = (i>0 && D[i-1]!=null)?D[i-1]:50;
    const newD = (2/3)*prevD + (1/3)*k;
    D.push(newD);
  });
  return { K, D };
}

export function calculateRSI(data, {period=14}={}) {
  // RSI 計算
  const rsi = [];
  if (!Array.isArray(data) || data.length === 0) return rsi;

  let gains = 0;
  let losses = 0;
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i-1].close;
    gains += change > 0 ? change : 0;
    losses += change < 0 ? Math.abs(change) : 0;
    if (i >= period) {
      const avgGain = gains / period;
      const avgLoss = losses / period;
      const rs = avgLoss === 0 ? 100 : (avgGain / avgLoss);
      const currentRSI = 100 - (100 / (1 + rs));
      rsi.push(currentRSI);
      const oldChange = data[i - period + 1].close - data[i - period].close;
      if (oldChange > 0) gains -= oldChange; else losses -= Math.abs(oldChange);
    } else {
      rsi.push(null);
    }
  }
  return rsi;
}

export function calculateMACD(data, {short=12, long=26, signal=9}={}) {
  // MACD 計算
  if (!Array.isArray(data)) return null;
  const closes = data.map(d=>d.close);
  if (closes.length < long) return null;

  const emaShort = ema(closes, short);
  const emaLong = ema(closes, long);
  const DIF = emaShort.map((v,i) => i<(long-1)?null:v - emaLong[i]);
  const difFiltered = DIF.filter(v=>v!==null);
  if (difFiltered.length === 0) return null;

  const DEA = ema(difFiltered, signal);

  // 對齊長度
  while (DEA.length < difFiltered.length) {
    DEA.unshift(null);
  }
  while (DEA.length < DIF.length) {
    DEA.unshift(null);
  }

  const MACD = DIF.map((d,i)=>(d!==null && DEA[i]!==null)?(d-DEA[i])*2:null);
  return { DIF, DEA, MACD };
}

export function calculateBollingerBands(data, {period=20}={}) {
  // 布林通道計算
  if (!Array.isArray(data) || data.length === 0) {
    return { upperBand:[], middleBand:[], lowerBand:[] };
  }
  const closes = data.map(d => d.close);
  const middleBand = [];
  const upperBand = [];
  const lowerBand = [];

  for (let i = 0; i < closes.length; i++) {
    if (i < period-1) {
      middleBand.push(null);
      upperBand.push(null);
      lowerBand.push(null);
      continue;
    }
    const slice = closes.slice(i-period+1, i+1);
    const m = mean(slice);
    const s = std(slice);
    middleBand.push(m);
    upperBand.push(m + 2*s);
    lowerBand.push(m - 2*s);
  }
  return { upperBand, middleBand, lowerBand };
}
