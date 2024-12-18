import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card } from 'antd';
import React from 'react';
import Chart from 'react-apexcharts';
import { formatNumber } from './analysis/utils/dataHelpers';

const I18N = {
  'zh-hant': {
    noData: "尚無成交量資料",
    title: "成交量",
    volume: "成交量",
    ma5: "5日均量",
    ma10: "10日均量",
    ma20: "20日均量",
    unit: " 張"
  },
  'en': {
    noData: "No volume data available",
    title: "Volume",
    volume: "Volume",
    ma5: "5-day MA Volume",
    ma10: "10-day MA Volume",
    ma20: "20-day MA Volume",
    unit: " lots"
  },
  'ja': {
    noData: "出来高データなし",
    title: "出来高",
    volume: "出来高",
    ma5: "5日平均出来高",
    ma10: "10日平均出来高",
    ma20: "20日平均出来高",
    unit: " 枚"
  }
};

export default function VolumeChartView({ rawData }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (!rawData || rawData.length === 0) return <p>{t.noData}</p>;

  const volumeData = rawData.map(d => ({
    x: d.date,
    y: d.Trading_Volume / 1000
  }));

  const categories = rawData.map(d => d.date);

  const ma5 = calcMovingAverage(volumeData.map(d=>d.y), 5);
  const ma10 = calcMovingAverage(volumeData.map(d=>d.y), 10);
  const ma20 = calcMovingAverage(volumeData.map(d=>d.y), 20);

  const ma5Series = ma5.map((val, i) => val !== null ? { x: categories[i], y: val } : { x: categories[i], y: null });
  const ma10Series = ma10.map((val, i) => val !== null ? { x: categories[i], y: val } : { x: categories[i], y: null });
  const ma20Series = ma20.map((val, i) => val !== null ? { x: categories[i], y: val } : { x: categories[i], y: null });

  const options = {
    chart: {
      type: 'bar',
      height: 300,
      toolbar: { show: false }
    },
    xaxis: {
      type: 'datetime',
      categories: categories,
      labels: {
        datetimeUTC: false,
        format: 'yyyy-MM-dd'
      },
      tickAmount: Math.min(10, categories.length),
    },
    yaxis: {
      labels: {
        formatter: (val) => (val != null ? `${formatNumber(val)}${t.unit}` : '')
      }
    },
    grid: {
      padding: { left: 10, right: 10, top: 20, bottom: 20 }
    },
    tooltip: {
      x: { format: 'yyyy-MM-dd' }
    },
    theme: {
      mode: 'light',
      palette: 'palette1'
    },
    plotOptions: {
      bar: {
        columnWidth: '50%',
        borderRadius: 2
      }
    },
    legend: {
      position: 'top',
      horizontalAlign: 'left',
      offsetX: 0
    },
    stroke: {
      curve: 'smooth',
      width: [0, 2, 2, 2]
    },
    colors: [
      '#7f8fa6',
      '#FF4560',
      '#00E396',
      '#775DD0'
    ]
  };

  const series = [
    {
      name: t.volume,
      type: 'bar',
      data: volumeData
    },
    {
      name: t.ma5,
      type: 'line',
      data: ma5Series
    },
    {
      name: t.ma10,
      type: 'line',
      data: ma10Series
    },
    {
      name: t.ma20,
      type: 'line',
      data: ma20Series
    }
  ];

  return (
    <Card style={{ marginTop: 20 }} title={t.title}>
      <Chart options={options} series={series} height={300} />
    </Card>
  );
}

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
