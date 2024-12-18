import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card } from 'antd';
import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

const I18N = {
  'zh-hant': {
    noData: "尚無 KD 資料或資料不足",
    title: "KD 指標",
    overbought: "超買(80)",
    oversold: "超賣(20)",
    kLine: "K",
    dLine: "D"
  },
  'en': {
    noData: "No KD data or insufficient data",
    title: "KD Indicator",
    overbought: "Overbought(80)",
    oversold: "Oversold(20)",
    kLine: "K",
    dLine: "D"
  },
  'ja': {
    noData: "KDデータなし、または不十分です",
    title: "KD 指標",
    overbought: "買われ過ぎ(80)",
    oversold: "売られ過ぎ(20)",
    kLine: "K",
    dLine: "D"
  }
};

export default function KDChartView({ kd, ohlcData }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (!kd || !kd.K || !kd.D || !Array.isArray(ohlcData) || ohlcData.length === 0) {
    return <p>{t.noData}</p>;
  }

  const KValues = kd.K;
  const DValues = kd.D;
  const categories = ohlcData.map(d => d.x);

  const minLen = Math.min(KValues.length, DValues.length, categories.length);
  const KData = KValues.slice(0, minLen).map(v => v ?? null);
  const DData = DValues.slice(0, minLen).map(v => v ?? null);
  const alignedCategories = categories.slice(0, minLen);

  // 將 Chart 的部分放入 BrowserOnly 中
  return (
    <Card style={{ marginTop: 20 }} title={t.title}>
      <BrowserOnly>
        {() => {
          const Chart = require('react-apexcharts').default; // 動態載入

          const options = {
            chart: { height: 300, type: 'line', toolbar: { show: false } },
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
              min: 0,
              max: 100,
              labels: {
                formatter: (val) => (val != null ? val.toFixed(2) : '')
              }
            },
            grid: {
              padding: { left: 10, right: 10, top: 20, bottom: 20 }
            },
            tooltip: {
              x: {
                format: 'yyyy-MM-dd'
              }
            },
            colors: ['#1f77b4', '#d62728'],
            stroke: {
              curve: 'smooth',
              width: 2
            },
            annotations: {
              yaxis: [
                {
                  y: 80,
                  borderColor: '#FF4560',
                  label: {
                    text: t.overbought,
                    style: { background: '#f5f5f5', color: '#FF4560' },
                    position: 'left'
                  },
                  strokeDashArray: 5
                },
                {
                  y: 20,
                  borderColor: '#00E396',
                  label: {
                    text: t.oversold,
                    style: { background: '#f5f5f5', color: '#00E396' },
                    position: 'left'
                  },
                  strokeDashArray: 5
                }
              ]
            }
          };

          const series = [
            { name: t.kLine, data: KData },
            { name: t.dLine, data: DData }
          ];

          const hasValidKD = series.some(s => s.data.some(val => val !== null));
          if (!hasValidKD) return <p>{t.noData}</p>;

          return <Chart options={options} series={series} height={300} type="line" />;
        }}
      </BrowserOnly>
    </Card>
  );
}
