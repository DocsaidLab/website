import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Card, Descriptions, Table, Tag, Tooltip } from 'antd';
import React from 'react';

const I18N = {
  'zh-hant': {
    noAnalysis: "尚無進階分析結果",
    marketTitle: "市場觀點與價位",
    marketView: "市場觀點",
    noMarketView: "尚無市場觀點",
    supportResistance: "支撐位 / 壓力位",
    srTooltip: "支撐與壓力為參考價，用於判斷可能的買賣區間",
    overallTrend: "總體趨勢變動",
    rise: "上漲 {val} 點",
    fall: "下跌 {val} 點",
    flat: "持平",
    marketStats: "市場統計",
    avgVolume: "平均成交量(股)",
    avgSpread: "平均Spread(點)",
    upDownFlatDays: "上漲 / 下跌 / 平盤天數",
    extremesTitle: "極值與紀錄",
    maxGain: "最大單日漲幅",
    maxLoss: "最大單日跌幅",
    highestVolumeDay: "最高成交量日",
    minMaxCloseDay: "最低 / 最高收盤價日",
    strategyTitle: "策略建議",
    noStrategy: "尚無策略建議",
    strategyItem: "策略項目",
    condition: "條件判斷",
    logic: "操作邏輯",
    risk: "風險控管",
    unitShares: "股",
    lowest: "最低",
    highest: "最高",
    close: "元"
  },
  'en': {
    noAnalysis: "No advanced analysis available",
    marketTitle: "Market View & Key Levels",
    marketView: "Market View",
    noMarketView: "No market view",
    supportResistance: "Support / Resistance",
    srTooltip: "Support and resistance are reference prices for potential buy/sell zones",
    overallTrend: "Overall Trend Change",
    rise: "Up {val} pts",
    fall: "Down {val} pts",
    flat: "Flat",
    marketStats: "Market Statistics",
    avgVolume: "Average Volume (shares)",
    avgSpread: "Average Spread (points)",
    upDownFlatDays: "Up / Down / Flat Days",
    extremesTitle: "Extremes & Records",
    maxGain: "Max Single-Day Gain",
    maxLoss: "Max Single-Day Loss",
    highestVolumeDay: "Highest Volume Day",
    minMaxCloseDay: "Lowest / Highest Close Day",
    strategyTitle: "Strategy Suggestions",
    noStrategy: "No strategy suggestions",
    strategyItem: "Strategy",
    condition: "Condition",
    logic: "Logic",
    risk: "Risk Control",
    unitShares: "shares",
    lowest: "Lowest",
    highest: "Highest",
    close: ""
  },
  'ja': {
    noAnalysis: "高度な分析結果はありません",
    marketTitle: "市場観点と重要レベル",
    marketView: "市場観点",
    noMarketView: "市場観点なし",
    supportResistance: "サポート / レジスタンス",
    srTooltip: "サポートとレジスタンスは潜在的な売買ゾーンの参考価格です",
    overallTrend: "総合トレンド変化",
    rise: "上昇 {val} ポイント",
    fall: "下落 {val} ポイント",
    flat: "横ばい",
    marketStats: "市場統計",
    avgVolume: "平均出来高(株)",
    avgSpread: "平均スプレッド(ポイント)",
    upDownFlatDays: "上昇 / 下落 / 横ばい日数",
    extremesTitle: "極値と記録",
    maxGain: "最大単日上昇幅",
    maxLoss: "最大単日下落幅",
    highestVolumeDay: "最高出来高日",
    minMaxCloseDay: "最安 / 最高終値日",
    strategyTitle: "戦略提案",
    noStrategy: "戦略提案なし",
    strategyItem: "戦略項目",
    condition: "条件判断",
    logic: "操作ロジック",
    risk: "リスク管理",
    unitShares: "株",
    lowest: "最安",
    highest: "最高",
    close: ""
  }
};

const formatNumber = (value, defaultValue = "N/A") => {
  if (value === null || value === undefined || isNaN(value)) return defaultValue;
  return Number(value).toFixed(2);
};

const getTrendDescription = (finalTrend, t) => {
  if (finalTrend > 0) return { text: t.rise.replace('{val}', formatNumber(finalTrend)), color: 'red' };
  if (finalTrend < 0) return { text: t.fall.replace('{val}', formatNumber(Math.abs(finalTrend))), color: 'green' };
  return { text: t.flat, color: 'default' };
};

export default function KeyMetricsView({ advancedAnalysis, marketView, strategies }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  const hasValidAnalysis = advancedAnalysis && advancedAnalysis.support !== undefined && advancedAnalysis.resistance !== undefined;
  if (!hasValidAnalysis) {
    return <p>{t.noAnalysis}</p>;
  }

  const {
    avgVolume, avgSpread, positive_days, negative_days, neutral_days,
    maxGain, maxGainDay, maxLoss, maxLossDay,
    maxGainClose, maxLossClose,
    maxVolume, maxVolumeDay,
    minClose, minCloseDay,
    maxClose, maxCloseDay,
    finalTrend, support, resistance
  } = advancedAnalysis;

  const trendDesc = getTrendDescription(finalTrend, t);
  const hasValidStrategies = Array.isArray(strategies) && strategies.length > 0;

  const columns = [
    { title: t.strategyItem, dataIndex: 'strategy', key: 'strategy', width: '20%' },
    { title: t.condition, dataIndex: 'condition', key: 'condition', width: '30%' },
    { title: t.logic, dataIndex: 'logic', key: 'logic', width: '30%' },
    { title: t.risk, dataIndex: 'risk', key: 'risk', width: '20%' }
  ];

  return (
    <div style={{ marginTop: 20 }}>
      <Card title={t.marketTitle} style={{ marginBottom: 20 }}>
        <Descriptions bordered column={1} size="small">
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.marketView}>
            {marketView || t.noMarketView}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.supportResistance}>
            <Tooltip title={t.srTooltip}>
              {formatNumber(support)} / {formatNumber(resistance)}
            </Tooltip>
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.overallTrend}>
            <Tag color={trendDesc.color}>{trendDesc.text}</Tag>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title={t.marketStats} style={{ marginBottom: 20 }}>
        <Descriptions bordered column={1} size="small">
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.avgVolume}>
            {avgVolume && !isNaN(avgVolume) ? Math.round(avgVolume).toLocaleString() + " " + t.unitShares : "N/A"}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.avgSpread}>
            {formatNumber(avgSpread)}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.upDownFlatDays}>
            {(positive_days || 0)} / {(negative_days || 0)} / {(neutral_days || 0)}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title={t.extremesTitle} style={{ marginBottom: 20 }}>
        <Descriptions bordered column={1} size="small">
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.maxGain}>
            {maxGain && !isNaN(maxGain)
              ? `${formatNumber(maxGain)} (${maxGainDay || "N/A"}) ${t.highest}: ${formatNumber(maxGainClose)}${t.close}`
              : "N/A"}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.maxLoss}>
            {maxLoss && !isNaN(maxLoss)
              ? `${formatNumber(maxLoss)} (${maxLossDay || "N/A"}) ${t.close}: ${formatNumber(maxLossClose)}${t.close}`
              : "N/A"}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.highestVolumeDay}>
            {(maxVolumeDay && maxVolume && !isNaN(maxVolume))
              ? `${maxVolumeDay}，${Number(maxVolume).toLocaleString()} ${t.unitShares}`
              : "N/A"}
          </Descriptions.Item>
          <Descriptions.Item labelStyle={{fontWeight:'bold'}} label={t.minMaxCloseDay}>
            {(minCloseDay && maxCloseDay && !isNaN(minClose) && !isNaN(maxClose))
              ? `${t.lowest}: ${formatNumber(minClose)}${t.close} (${minCloseDay}) / ${t.highest}: ${formatNumber(maxClose)}${t.close} (${maxCloseDay})`
              : "N/A"}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title={t.strategyTitle}>
        {hasValidStrategies ? (
          <Table
            columns={columns}
            dataSource={strategies}
            pagination={false}
            bordered
            size="small"
            style={{ marginTop: 10 }}
            tableLayout="fixed"
            rowKey={(record) => record.key || record.strategy}
          />
        ) : (
          <p style={{ marginTop: 10 }}>{t.noStrategy}</p>
        )}
      </Card>
    </div>
  );
}
