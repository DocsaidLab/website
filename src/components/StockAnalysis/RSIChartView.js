import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card } from 'antd';
import React from 'react';
import Chart from 'react-apexcharts';

const I18N = {
  'zh-hant': {
    noData: "尚無 RSI 資料或資料不足",
    title: "RSI 指標",
    overbought: "超買(70)",
    oversold: "超賣(30)"
  },
  'en': {
    noData: "No RSI data or insufficient data",
    title: "RSI Indicator",
    overbought: "Overbought(70)",
    oversold: "Oversold(30)"
  },
  'ja': {
    noData: "RSIデータなし、または不十分です",
    title: "RSI 指標",
    overbought: "買われ過ぎ(70)",
    oversold: "売られ過ぎ(30)"
  }
};

export default function RSIChartView({ rsi, ohlcData }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (!rsi || !ohlcData || ohlcData.length === 0) return <p>{t.noData}</p>;

  const minLen = Math.min(rsi.length, ohlcData.length);
  const truncatedRSI = rsi.slice(0, minLen).map(v => (v !== null ? v : null));
  const categories = ohlcData.slice(0, minLen).map(d => d.x);

  const hasValidRSI = truncatedRSI.some(v => v !== null);
  if (!hasValidRSI) return <p>{t.noData}</p>;

  const options = {
    chart: { height: 300, type: 'line', toolbar: { show: false } },
    theme: {
      mode: 'light',
      palette: 'palette2'
    },
    xaxis: {
      type: 'datetime',
      categories,
      labels: {
        rotate: -45,
        datetimeUTC: false,
        format: 'yyyy-MM-dd',
      },
      tickAmount: Math.min(10, categories.length),
    },
    yaxis: {
      min: 0,
      max: 100,
      labels: {
        formatter: (val) => (val != null ? val.toFixed(2) : '')
      }
    },
    grid: {
      padding: { left: 10, right: 10, top: 20, bottom: 20 }
    },
    stroke: {
      curve: 'smooth',
      width: 2
    },
    colors: ['#1f77b4'],
    annotations: {
      yaxis: [
        {
          y: 70,
          borderColor: '#FF4560',
          label: {
            text: t.overbought,
            style: { background: '#f5f5f5', color: '#FF4560' },
            position: 'left'
          },
          strokeDashArray: 4
        },
        {
          y: 30,
          borderColor: '#00E396',
          label: {
            text: t.oversold,
            style: { background: '#f5f5f5', color: '#00E396' },
            position: 'left'
          },
          strokeDashArray: 4
        }
      ]
    },
    tooltip: {
      x: {
        format: 'yyyy-MM-dd'
      }
    },
    legend: {
      position: 'top'
    }
  };

  const series = [{ name: 'RSI', data: truncatedRSI }];

  return (
    <Card style={{ marginTop: 20 }} title={t.title}>
      <Chart options={options} series={series} height={300} type="line" />
    </Card>
  );
}
