import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React from 'react';
import Chart from 'react-apexcharts';
import { formatPrice } from './analysis/utils/dataHelpers';

const I18N = {
  'zh-hant': {
    noData: "尚無主圖資料",
    chartTitle: (numDays) => `近 ${numDays} 交易日 K 線圖`,
    support: "支撐",
    resistance: "壓力",
    candles: "K線",
    upperBand: "上軌",
    middleBand: "中軌",
    lowerBand: "下軌",
    ma5: "5MA",
    ma10: "10MA",
    ma20: "20MA"
  },
  'en': {
    noData: "No main chart data",
    chartTitle: (numDays) => `${numDays}-Day K Line Chart`,
    support: "Support",
    resistance: "Resistance",
    candles: "Candles",
    upperBand: "Upper Band",
    middleBand: "Middle Band",
    lowerBand: "Lower Band",
    ma5: "5MA",
    ma10: "10MA",
    ma20: "20MA"
  },
  'ja': {
    noData: "メインチャートデータなし",
    chartTitle: (numDays) => `直近${numDays}営業日のKラインチャート`,
    support: "サポート",
    resistance: "レジスタンス",
    candles: "ローソク足",
    upperBand: "上限バンド",
    middleBand: "中心バンド",
    lowerBand: "下限バンド",
    ma5: "5MA",
    ma10: "10MA",
    ma20: "20MA"
  }
};

export default function MainChartView({ ohlcData, bollinger, advancedAnalysis, numDays }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (ohlcData.length === 0) return <p>{t.noData}</p>;

  const closes = ohlcData.map(d => d.y[3]);
  const ma5 = calcMovingAverage(closes, 5);
  const ma10 = calcMovingAverage(closes, 10);
  const ma20 = calcMovingAverage(closes, 20);
  const categories = ohlcData.map(d => d.x);

  const formatBollingerSeries = (bollinger, ohlcData) => {
    const categories = ohlcData.map(d => d.x);
    const formatBand = (band) =>
      band.map((v, i) => (v !== null ? { x: categories[i], y: v } : null)).filter(Boolean);

    return [
      { name: t.upperBand, type: 'line', data: formatBand(bollinger.upperBand) },
      { name: t.middleBand, type: 'line', data: formatBand(bollinger.middleBand) },
      { name: t.lowerBand, type: 'line', data: formatBand(bollinger.lowerBand) },
    ];
  };

  const yFormatter = (val) => formatPrice(val);

  const candleOptions = {
    chart: { type: 'candlestick', height: 450, toolbar: { show: false } },
    title: { text: t.chartTitle(numDays), align: 'left' },
    plotOptions: {
      candlestick: {
        colors: { upward: '#EF403C', downward: '#00B746' },
      },
    },
    xaxis: {
      type: 'datetime',
      labels: {
        rotate: -45,
        datetimeUTC: false,
        format: 'yyyy-MM-dd',
      },
      tickAmount: Math.min(10, ohlcData.length),
    },
    yaxis: {
      labels: { formatter: yFormatter },
      tooltip: { enabled: true },
    },
    annotations: {
      yaxis: advancedAnalysis
        ? [
            {
              y: advancedAnalysis.support,
              borderColor: '#00E396',
              label: { text: `${t.support} ${formatPrice(advancedAnalysis.support)}`, position: 'left'},
            },
            {
              y: advancedAnalysis.resistance,
              borderColor: '#FF4560',
              label: { text: `${t.resistance} ${formatPrice(advancedAnalysis.resistance)}`, position: 'left' },
            },
          ]
        : [],
    },
    grid: {
      padding: { left: 10, right: 10, top: 20, bottom: 20 }
    },
    colors: [
      '#A9A9A9', // Candles
      '#FF6F61', // Upper Band
      '#1E90FF', // Middle Band
      '#9C88FF', // Lower Band
      '#FFB74D', // 5MA
      '#0078D7', // 10MA
      '#32CD32'  // 20MA
    ],
    stroke: {
      width: [1, 2, 2, 2, 2, 2, 2],
      curve: 'smooth'
    },
    legend: {
      position: 'top',
      horizontalAlign: 'left',
      offsetX: 0
    },
    tooltip: {
      x: { format: 'yyyy-MM-dd' }
    }
  };

  const series = [{ name: t.candles, type: 'candlestick', data: ohlcData }];

  if (bollinger) {
    series.push(...formatBollingerSeries(bollinger, ohlcData));
  }

  const ma5Series = ma5.map((v, i) => v !== null ? { x: categories[i], y: v } : { x: categories[i], y: null });
  const ma10Series = ma10.map((v, i) => v !== null ? { x: categories[i], y: v } : { x: categories[i], y: null });
  const ma20Series = ma20.map((v, i) => v !== null ? { x: categories[i], y: v } : { x: categories[i], y: null });

  series.push(
    { name: t.ma5, type: 'line', data: ma5Series },
    { name: t.ma10, type: 'line', data: ma10Series },
    { name: t.ma20, type: 'line', data: ma20Series }
  );

  return (
    <div style={{ marginTop: 20 }}>
      <Chart options={candleOptions} series={series} height={450} />
    </div>
  );
}

// 計算簡單移動平均線的函式
function calcMovingAverage(data, period) {
  if (!Array.isArray(data) || data.length === 0) return [];
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const slice = data.slice(i - period + 1, i + 1);
      const avg = slice.reduce((sum, val) => sum + val, 0) / period;
      result.push(avg);
    }
  }
  return result;
}
