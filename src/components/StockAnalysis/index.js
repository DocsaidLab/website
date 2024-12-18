import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Alert, Button, Col, Input, Layout, Row, Spin } from 'antd';
import React, { useMemo } from 'react';
import IndicatorsView from './IndicatorsView';
import KDChartView from './KDChartView';
import KeyMetricsView from './KeyMetricsView';
import MACDChartView from './MACDChartView';
import MainChartView from './MainChartView';
import RSIChartView from './RSIChartView';
import VolumeChartView from './VolumeChartView';
import { useStockData } from './useStockData';

const { Content } = Layout;

const I18N = {
  'zh-hant': {
    enterStockId: "請輸入股票代號 (例如：2330)",
    placeholderId: "輸入股票代號",
    getData: "取得資料",
    loadingData: "資料載入中...",
    retry: "重新嘗試"
  },
  'en': {
    enterStockId: "Please enter a stock ID (e.g., 2330)",
    placeholderId: "Enter stock ID",
    getData: "Get Data",
    loadingData: "Loading data...",
    retry: "Retry"
  },
  'ja': {
    enterStockId: "株式IDを入力してください（例：2330）",
    placeholderId: "株式IDを入力",
    getData: "データ取得",
    loadingData: "データ読み込み中...",
    retry: "再試行"
  }
};

export default function OptimizedStockAnalysisPage() {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const t = I18N[currentLocale] || I18N['zh-hant'];

  const {
    stockId, setStockId,
    ohlcData,
    kd,
    rsi,
    bollinger,
    macd,
    advancedAnalysis,
    professionalReport,
    loading,
    error,
    fetchData,
    numDays,
    rawData,
    marketView,
    strategies
  } = useStockData();

  const chartProps = useMemo(() => ({
    ohlcData,
    kd,
    rsi,
    macd,
    bollinger,
    advancedAnalysis,
    numDays,
    rawData
  }), [ohlcData, kd, rsi, macd, bollinger, advancedAnalysis, numDays, rawData]);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ padding: '20px' }}>
        <p>{t.enterStockId}</p>
        <Row gutter={16} align="middle">
          <Col span={6}>
            <Input
              value={stockId}
              onChange={e => setStockId(e.target.value)}
              placeholder={t.placeholderId}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4}>
            <Button type="primary" onClick={fetchData} block>
              {t.getData}
            </Button>
          </Col>
        </Row>

        {loading && <Spin style={{ marginTop: 20 }} tip={t.loadingData} />}
        {error && (
          <Alert
            style={{ marginTop: 20 }}
            message={error}
            type="error"
            action={
              <Button size="small" onClick={fetchData}>
                {t.retry}
              </Button>
            }
          />
        )}

        {ohlcData.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <MainChartView {...chartProps} />
            <VolumeChartView {...chartProps} />
            <RSIChartView {...chartProps} />
            <KDChartView {...chartProps} />
            <MACDChartView {...chartProps} />
            <IndicatorsView {...chartProps} />
            <KeyMetricsView
              advancedAnalysis={advancedAnalysis}
              marketView={marketView}
              strategies={strategies}
            />
          </div>
        )}
      </Content>
    </Layout>
  );
}
