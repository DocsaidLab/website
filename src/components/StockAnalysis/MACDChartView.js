import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card } from 'antd';
import React from 'react';
import Chart from 'react-apexcharts';

const I18N = {
  'zh-hant': {
    noMACDData: "資料不足，無法計算 MACD",
    noMACDChart: "資料不足，無法繪製 MACD 圖表",
    title: "MACD 指標",
    zeroAxis: "0 軸",
    dif: "DIF",
    dea: "DEA",
    macd: "MACD"
  },
  'en': {
    noMACDData: "Insufficient data, unable to calculate MACD",
    noMACDChart: "Insufficient data, unable to plot MACD chart",
    title: "MACD Indicator",
    zeroAxis: "Zero Axis",
    dif: "DIF",
    dea: "DEA",
    macd: "MACD"
  },
  'ja': {
    noMACDData: "データ不足のため、MACDを計算できません",
    noMACDChart: "データ不足のため、MACDチャートを描画できません",
    title: "MACD 指標",
    zeroAxis: "0軸",
    dif: "DIF",
    dea: "DEA",
    macd: "MACD"
  }
};

export default function MACDChartView({ macd, ohlcData }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  const hasValidMACD = macd && macd.DIF && macd.DEA && macd.MACD;
  if (!hasValidMACD) return <p>{t.noMACDData}</p>;

  const categories = ohlcData.map(d => d.x);

  const minLen = Math.min(categories.length, macd.DIF.length, macd.DEA.length, macd.MACD.length);
  const DIFData = macd.DIF.slice(0, minLen).map(v => v ?? null);
  const DEAData = macd.DEA.slice(0, minLen).map(v => v ?? null);
  const MACDData = macd.MACD.slice(0, minLen).map(v => v ?? 0);
  const alignedCategories = categories.slice(0, minLen);

  const options = {
    chart: { height: 300, toolbar: { show: false } },
    theme: {
      mode: 'light',
      palette: 'palette2'
    },
    xaxis: {
      type: 'datetime',
      categories: alignedCategories,
      labels: {
        rotate: -45,
        datetimeUTC: false,
        format: 'yyyy-MM-dd',
      },
      tickAmount: Math.min(10, alignedCategories.length),
    },
    yaxis: {
      labels: {
        formatter: (val) => (val != null ? val.toFixed(2) : '')
      }
    },
    plotOptions: {
      bar: {
        columnWidth: '40%',
        colors: {
          ranges: [
            { from: -1000, to: 0, color: '#00B746' },
            { from: 0, to: 1000, color: '#EF403C' }
          ]
        }
      }
    },
    annotations: {
      yaxis: [
        {
          y: 0,
          strokeDashArray: 4,
          borderColor: '#999',
          label: {
            text: t.zeroAxis,
            style: {
              color: '#333',
              background: '#eaeaea'
            },
            position: 'left'
          }
        }
      ]
    },
    grid: {
      padding: { left: 10, right: 10, top: 20, bottom: 20 }
    },
    stroke: {
      curve: 'smooth',
      width: 2
    },
    colors: ['#1f77b4', '#d62728', '#7f8fa6'],
    legend: {
      position: 'top'
    }
  };

  const series = [
    { name: t.dif, type: 'line', data: DIFData },
    { name: t.dea, type: 'line', data: DEAData },
    { name: t.macd, type: 'column', data: MACDData }
  ];

  const hasData = series.some(s => s.data.some(val => val !== null && val !== undefined));
  if (!hasData) return <p>{t.noMACDChart}</p>;

  return (
    <Card style={{ marginTop: 20 }} title={t.title}>
      <Chart options={options} series={series} height={300} type="line" />
    </Card>
  );
}
