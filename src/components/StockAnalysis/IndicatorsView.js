import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card, Descriptions, Tag } from 'antd';
import React from 'react';

const I18N = {
  'zh-hant': {
    noData: "尚無指標資料",
    title: "技術指標",
    intro: "下列為常見技術指標最新值。K、D、RSI 等指標可輔助判斷超買/超賣狀況，布林通道則可觀察價格在平均值上下的分布範圍。",
    kdLabel: "KD 指標 (短期動能)",
    rsiLabel: "RSI 指標 (相對強弱)",
    bollingerLabel: "布林通道 (價格波動範圍)",
    upper: "上軌",
    middle: "中軌",
    lower: "下軌",
    overbought: "超買",
    oversold: "超賣"
  },
  'en': {
    noData: "No indicator data",
    title: "Technical Indicators",
    intro: "Below are the latest values of common technical indicators. K, D, RSI can help judge overbought/oversold conditions, and Bollinger Bands show price distribution around the average.",
    kdLabel: "KD Indicator (Short-term momentum)",
    rsiLabel: "RSI Indicator (Relative Strength)",
    bollingerLabel: "Bollinger Bands (Price Volatility Range)",
    upper: "Upper Band",
    middle: "Middle Band",
    lower: "Lower Band",
    overbought: "Overbought",
    oversold: "Oversold"
  },
  'ja': {
    noData: "指標データなし",
    title: "テクニカル指標",
    intro: "以下は一般的なテクニカル指標の最新値です。K、D、RSIは買われ過ぎ/売られ過ぎの判断に役立ち、ボリンジャーバンドは平均値周辺での価格分布を観察できます。",
    kdLabel: "KD 指標（短期モメンタム）",
    rsiLabel: "RSI 指標（相対的強弱）",
    bollingerLabel: "ボリンジャーバンド（価格変動範囲）",
    upper: "上限バンド",
    middle: "中心バンド",
    lower: "下限バンド",
    overbought: "買われ過ぎ",
    oversold: "売られ過ぎ"
  }
};

const getLatestValue = (data, defaultValue = "N/A") => {
  if (!data || data.length === 0) return defaultValue;
  const value = data[data.length - 1];
  return value != null ? value : defaultValue;
};

const formatNumber = (val, decimals = 2) => {
  if (val === "N/A" || val == null || isNaN(val)) return "N/A";
  return Number(val).toFixed(decimals);
};

const getKDStatus = (k, d, t) => {
  if (k === "N/A" || d === "N/A") return null;
  const kVal = parseFloat(k);
  const dVal = parseFloat(d);
  if (kVal > 80 || dVal > 80) return { status: t.overbought, color: 'red' };
  if (kVal < 20 || dVal < 20) return { status: t.oversold, color: 'green' };
  return null;
};

const getRSIStatus = (rsi, t) => {
  if (rsi === "N/A") return null;
  const val = parseFloat(rsi);
  if (val > 70) return { status: t.overbought, color: 'red' };
  if (val < 30) return { status: t.oversold, color: 'green' };
  return null;
};

export default function IndicatorsView({ kd, rsi, bollinger }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  const hasValidData = kd && kd.K && kd.D && rsi && bollinger && bollinger.upperBand;
  if (!hasValidData) {
    return <p>{t.noData}</p>;
  }

  const latestK = formatNumber(getLatestValue(kd.K));
  const latestD = formatNumber(getLatestValue(kd.D));
  const latestRSIVal = formatNumber(getLatestValue(rsi));
  const fUB = formatNumber(getLatestValue(bollinger.upperBand));
  const fMB = formatNumber(getLatestValue(bollinger.middleBand));
  const fLB = formatNumber(getLatestValue(bollinger.lowerBand));

  const kdStatus = getKDStatus(latestK, latestD, t);
  const rsiStatus = getRSIStatus(latestRSIVal, t);

  return (
    <Card style={{ marginTop: 20 }} title={t.title}>
      <p style={{ marginBottom: 10 }}>
        {t.intro}
      </p>
      <Descriptions bordered size="small" column={1}>
        <Descriptions.Item labelStyle={{ fontWeight: 'bold' }} label={t.kdLabel}>
          K: {latestK}　D: {latestD}
          {kdStatus && <Tag color={kdStatus.color} style={{ marginLeft: 10 }}>{kdStatus.status}</Tag>}
        </Descriptions.Item>

        <Descriptions.Item labelStyle={{ fontWeight: 'bold' }} label={t.rsiLabel}>
          RSI: {latestRSIVal}
          {rsiStatus && <Tag color={rsiStatus.color} style={{ marginLeft: 10 }}>{rsiStatus.status}</Tag>}
        </Descriptions.Item>

        <Descriptions.Item labelStyle={{ fontWeight: 'bold' }} label={t.bollingerLabel}>
          {t.upper}: {fUB}　{t.middle}: {fMB}　{t.lower}: {fLB}
        </Descriptions.Item>
      </Descriptions>
    </Card>
  );
}
