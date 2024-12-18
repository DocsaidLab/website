import { mean, std } from 'mathjs';
import { findMax, findMin, formatNumber } from './utils/dataHelpers';

const I18N = {
  'zh-hant': {
    unableToJudge: "無法判斷",
    consolidation: "盤整",
    bullish: "多方趨勢",
    bearish: "空方趨勢",
    watchStrategy: "觀望策略",
    noDataForReport: "尚無足夠資料進行分析報告。",
    volumeTrendHint: "{volumeTrend} 顯示資金流動性變化",
    professionalReportIntro: "以下為股票代號 {stock_id} 的專業分析報告：",
    basicStatsTitle: "### 基礎數據統計與趨勢概覽",
    avgVolume: "平均交易量：約 {avgVolume} 股/日，{volumeTrend}",
    avgSpread: "收盤價平均變動（spread）：{avgSpread} 點/日",
    riseDays: "上漲天數：{positive_days} 天",
    fallDays: "下跌天數：{negative_days} 天",
    flatDays: "平盤天數：{neutral_days} 天",
    overallTrend: "總體價格趨勢：從首日到末日{trendDescription}",
    extremesTitle: "### 極值表現與關鍵價位",
    maxGain: "最大單日漲幅：{maxGain} 點 ({maxGainDay})，收盤價 {maxGainClose} 元",
    maxLoss: "最大單日跌幅：{maxLoss} 點 ({maxLossDay})，收盤價 {maxLossClose} 元",
    highestVolumeDay: "最高成交量日：{maxVolumeDay}，達 {maxVolume} 股",
    minMaxClose: "最低收盤價：{minClose} 元 ({minCloseDay})，最高收盤價：{maxClose} 元 ({maxCloseDay})",
    supportResistance: "支撐位：{support} 元，壓力位：{resistance} 元",
    marketViewTitle: "### 市場觀點與量價結構",
    marketTone: "市場基調：{marketView}",
    patternSignalsTitle: "### 型態訊號",
    strategiesTitle: "### 策略建議與風險控管",
    summaryTitle: "### 總結",
    summaryContent: "在當前環境下，適度觀察量價結構與技術指標變化，依據支撐、壓力位調整操作策略，並嚴守風險控管。",
    trendUp: "上漲 {val} 點",
    trendDown: "下跌 {val} 點",
    trendFlat: "幾乎持平",

    strategyShortTermBull: "短線多方布局",
    strategyShortTermBear: "短線空方避險",
    strategyRangeTrade: "區間操作",
    strategyTrendBreak: "多空轉折確認",
    noCondition: "無特定條件",

    conditionStableIncrease: "站穩 {price}元並量增價漲 ({volumeTrend})",
    conditionBreakBelow: "跌破 {price}元 且放量",
    conditionRange: "在 {lowerBound}元 - {upperBound}元 區間",
    conditionBreakOut: "有效突破 {price}元",

    logicBuyOnDips: "逢低買進，上看 {resistance}元",
    logicExitOnRebound: "反彈出場，空倉觀望",
    logicRangeTrade: "區間高拋低吸",
    logicFollowTrend: "加碼多單，順勢操作",

    riskStopLoss: "止損設在 {support}元",
    riskAvoidBreakdown: "避免下行破位",
    riskDiscipline: "嚴守紀律，勿追高",
    riskPullback: "留意短期回檔風險",

    conditionLabel: "條件",
    logicLabel: "操作邏輯",
    riskLabel: "風險控管"
  },
  'en': {
    unableToJudge: "Unable to determine",
    consolidation: "Consolidation",
    bullish: "Bullish trend",
    bearish: "Bearish trend",
    watchStrategy: "Watch and Wait",
    noDataForReport: "Not enough data to generate analysis report.",
    volumeTrendHint: "{volumeTrend} indicates changes in liquidity",
    professionalReportIntro: "Below is the professional analysis report for stock ID {stock_id}:",
    basicStatsTitle: "### Basic Statistics & Trend Overview",
    avgVolume: "Average volume: about {avgVolume} shares/day, {volumeTrend}",
    avgSpread: "Average close spread: {avgSpread} points/day",
    riseDays: "Days Up: {positive_days}",
    fallDays: "Days Down: {negative_days}",
    flatDays: "Flat Days: {neutral_days}",
    overallTrend: "Overall price trend from first to last day: {trendDescription}",
    extremesTitle: "### Extreme Performances & Key Levels",
    maxGain: "Max single-day gain: {maxGain} pts ({maxGainDay}), Close: {maxGainClose}",
    maxLoss: "Max single-day loss: {maxLoss} pts ({maxLossDay}), Close: {maxLossClose}",
    highestVolumeDay: "Highest volume day: {maxVolumeDay}, {maxVolume} shares",
    minMaxClose: "Lowest close: {minClose} ({minCloseDay}), Highest close: {maxClose} ({maxCloseDay})",
    supportResistance: "Support: {support}, Resistance: {resistance}",
    marketViewTitle: "### Market View & Volume/Price Structure",
    marketTone: "Market tone: {marketView}",
    patternSignalsTitle: "### Pattern Signals",
    strategiesTitle: "### Strategy Suggestions & Risk Control",
    summaryTitle: "### Summary",
    summaryContent: "Given the current environment, observe volume-price structures and technical indicators, adjust strategies based on support/resistance levels, and maintain strict risk control.",
    trendUp: "up {val} pts",
    trendDown: "down {val} pts",
    trendFlat: "nearly flat",

    strategyShortTermBull: "Short-term Bullish Strategy",
    strategyShortTermBear: "Short-term Bearish Hedge",
    strategyRangeTrade: "Range Trading",
    strategyTrendBreak: "Trend Break Confirmation",
    noCondition: "No specific conditions",

    conditionStableIncrease: "Stay above {price} and price-volume increasing ({volumeTrend})",
    conditionBreakBelow: "Break below {price} with volume",
    conditionRange: "In {lowerBound} - {upperBound} range",
    conditionBreakOut: "Effective breakout above {price}",

    logicBuyOnDips: "Buy on dips, target {resistance}",
    logicExitOnRebound: "Exit on rebound, stay sideline",
    logicRangeTrade: "Range trading, buy low sell high",
    logicFollowTrend: "Add long positions, follow the trend",

    riskStopLoss: "Stop loss at {support}",
    riskAvoidBreakdown: "Avoid further breakdown",
    riskDiscipline: "Discipline, don't chase highs",
    riskPullback: "Watch for short-term pullbacks",

    conditionLabel: "Condition",
    logicLabel: "Logic",
    riskLabel: "Risk Control"
  },
  'ja': {
    unableToJudge: "判断不可",
    consolidation: "もみ合い",
    bullish: "強気トレンド",
    bearish: "弱気トレンド",
    watchStrategy: "様子見戦略",
    noDataForReport: "分析レポートを生成するための十分なデータがありません。",
    volumeTrendHint: "{volumeTrend} は流動性の変化を示す",
    professionalReportIntro: "以下は株式コード {stock_id} の専門的分析レポートです:",
    basicStatsTitle: "### 基本統計とトレンド概観",
    avgVolume: "平均出来高: 約 {avgVolume} 株/日，{volumeTrend}",
    avgSpread: "終値平均変動（スプレッド）：{avgSpread} ポイント/日",
    riseDays: "上昇日数：{positive_days}日",
    fallDays: "下落日数：{negative_days}日",
    flatDays: "横ばい日数：{neutral_days}日",
    overallTrend: "初日から最終日までの総合的な価格トレンド：{trendDescription}",
    extremesTitle: "### 極値および重要価格",
    maxGain: "最大単日上昇幅：{maxGain} ポイント ({maxGainDay})，終値 {maxGainClose}",
    maxLoss: "最大単日下落幅：{maxLoss} ポイント ({maxLossDay})，終値 {maxLossClose}",
    highestVolumeDay: "最高出来高日：{maxVolumeDay}，{maxVolume} 株",
    minMaxClose: "最安終値：{minClose} ({minCloseDay})，最高終値：{maxClose} ({maxCloseDay})",
    supportResistance: "サポート：{support}，レジスタンス：{resistance}",
    marketViewTitle: "### 市場観点と出来高・価格構造",
    marketTone: "市場基調：{marketView}",
    patternSignalsTitle: "### パターンサイン",
    strategiesTitle: "### 戦略とリスク管理",
    summaryTitle: "### まとめ",
    summaryContent: "現状では、出来高・価格構造やテクニカル指標を観察し、サポート・レジスタンスに基づいて戦略を調整し、厳格なリスク管理を維持してください。",
    trendUp: "上昇 {val} ポイント",
    trendDown: "下落 {val} ポイント",
    trendFlat: "ほぼ変わらず",

    strategyShortTermBull: "短期強気戦略",
    strategyShortTermBear: "短期弱気ヘッジ",
    strategyRangeTrade: "レンジ取引",
    strategyTrendBreak: "トレンド転換確認",
    noCondition: "特定条件なし",

    conditionStableIncrease: "{price}以上を維持し、出来高増加({volumeTrend})で価格上昇",
    conditionBreakBelow: "{price}を割れ、かつ出来高増加",
    conditionRange: "{lowerBound} - {upperBound} のレンジ内",
    conditionBreakOut: "{price}を有効突破",

    logicBuyOnDips: "安値で買い、高値目標 {resistance}",
    logicExitOnRebound: "反発で退出、ノーポジション維持",
    logicRangeTrade: "レンジ内で安値買い・高値売り",
    logicFollowTrend: "多ポジション追加、トレンドフォロー",

    riskStopLoss: "{support}でストップロス設定",
    riskAvoidBreakdown: "さらなる下落を回避",
    riskDiscipline: "規律重視、高値追い禁止",
    riskPullback: "短期的な押し戻りに注意",

    conditionLabel: "条件",
    logicLabel: "操作ロジック",
    riskLabel: "リスク管理"
  }
};

/**
 * 產生進階分析摘要資訊
 * @param {Array} data
 * @returns {Object|null}
 */
export function generateAdvancedSummary(data) {
  if (!Array.isArray(data) || data.length === 0) return null;

  const volumes = data.map(d => d.Trading_Volume);
  const closes = data.map(d => d.close);
  const spreads = data.map(d => d.spread);

  if (volumes.length === 0 || closes.length === 0 || spreads.length === 0) return null;

  const avgVolume = mean(volumes);
  const avgSpread = mean(spreads);
  const positive_days = spreads.filter(s => s > 0).length;
  const negative_days = spreads.filter(s => s < 0).length;
  const neutral_days = spreads.filter(s => s === 0).length;

  const maxGainDay = findMax(data, 'spread');
  const maxLossDay = findMin(data, 'spread');
  const maxVolumeDay = findMax(data, 'Trading_Volume');
  const minCloseDay = findMin(data, 'close');
  const maxCloseDay = findMax(data, 'close');

  if (!maxGainDay || !maxLossDay || !maxVolumeDay || !minCloseDay || !maxCloseDay) return null;

  const finalTrend = closes[closes.length - 1] - closes[0];
  const stdClose = std(closes);
  const support = parseFloat(formatNumber(mean(closes) - stdClose));
  const resistance = parseFloat(formatNumber(mean(closes) + stdClose));

  const patterns = []; // patterns由外部決定是否傳入語系呼叫 detectPatterns(data,lang)

  return {
    avgVolume,
    avgSpread,
    positive_days,
    negative_days,
    neutral_days,
    maxGain: maxGainDay.spread,
    maxGainDay: maxGainDay.date,
    maxGainClose: maxGainDay.close,
    maxLoss: maxLossDay.spread,
    maxLossDay: maxLossDay.date,
    maxLossClose: maxLossDay.close,
    maxVolume: maxVolumeDay.Trading_Volume,
    maxVolumeDay: maxVolumeDay.date,
    minClose: minCloseDay.close,
    minCloseDay: minCloseDay.date,
    maxClose: maxCloseDay.close,
    maxCloseDay: maxCloseDay.date,
    finalTrend,
    support,
    resistance,
    patterns
  };
}

/**
 * 根據分析結果與量能趨勢產生交易策略
 * @param {Object} analysis
 * @param {String} volumeTrend
 * @param {String} lang 'zh-hant'|'en'|'ja'
 * @returns {Object} {marketView, strategies}
 */
export function generateTradingStrategies(analysis, volumeTrend, lang='zh-hant') {
  const t = I18N[lang] || I18N['zh-hant'];

  if (!analysis) {
    return { marketView: t.unableToJudge, strategies: [] };
  }

  const { support, resistance, finalTrend } = analysis;

  let marketView = t.consolidation;
  if (finalTrend > 20) {
    marketView = t.bullish;
  } else if (finalTrend < -20) {
    marketView = t.bearish;
  }

  const midPoint = ((support + resistance) / 2).toFixed(2);

  const strategies = [
    {
      key: '1',
      strategy: t.strategyShortTermBull,
      condition: t.conditionStableIncrease
        .replace('{price}', (support+5).toFixed(2))
        .replace('{volumeTrend}', volumeTrend),
      logic: t.logicBuyOnDips.replace('{resistance}', resistance.toFixed(2)),
      risk: t.riskStopLoss.replace('{support}', support.toFixed(2))
    },
    {
      key: '2',
      strategy: t.strategyShortTermBear,
      condition: t.conditionBreakBelow.replace('{price}', support.toFixed(2)),
      logic: t.logicExitOnRebound,
      risk: t.riskAvoidBreakdown
    },
    {
      key: '3',
      strategy: t.strategyRangeTrade,
      condition: t.conditionRange
        .replace('{lowerBound}', (parseFloat(midPoint)-10).toFixed(2))
        .replace('{upperBound}', (parseFloat(midPoint)+10).toFixed(2)),
      logic: t.logicRangeTrade,
      risk: t.riskDiscipline
    },
    {
      key: '4',
      strategy: t.strategyTrendBreak,
      condition: t.conditionBreakOut.replace('{price}', resistance.toFixed(2)),
      logic: t.logicFollowTrend,
      risk: t.riskPullback
    }
  ];

  if (!strategies || strategies.length === 0) {
    return {
      marketView,
      strategies: [{
        key: 'default',
        strategy: t.watchStrategy,
        condition: t.noCondition,
        logic: t.noCondition,
        risk: t.riskDiscipline
      }]
    };
  }

  return { marketView, strategies };
}

/**
 * 產生專業分析報告
 * @param {String} stock_id
 * @param {Object} advancedAnalysis
 * @param {String} volumeTrend
 * @param {String} marketView
 * @param {Array} strategies
 * @param {String} lang 'zh-hant'|'en'|'ja'
 * @param {Array} patterns 已由 detectPatterns(data, lang) 得到之結果
 * @returns {String}
 */
export function generateProfessionalReport(stock_id, advancedAnalysis, volumeTrend, marketView, strategies, lang='zh-hant', patterns=[]) {
  const t = I18N[lang] || I18N['zh-hant'];

  if (!advancedAnalysis) return t.noDataForReport;

  const {
    avgVolume, avgSpread, positive_days, negative_days, neutral_days,
    maxGain, maxGainDay, maxGainClose, maxLoss, maxLossDay, maxLossClose,
    maxVolume, maxVolumeDay, minClose, minCloseDay, maxClose, maxCloseDay,
    finalTrend, support, resistance
  } = advancedAnalysis;

  let trendDescription = t.trendFlat;
  if (finalTrend > 0) {
    trendDescription = t.trendUp.replace('{val}', Math.abs(finalTrend).toFixed(2));
  } else if (finalTrend < 0) {
    trendDescription = t.trendDown.replace('{val}', Math.abs(finalTrend).toFixed(2));
  }

  const patternsText = (patterns && patterns.length > 0)
    ? `\n${t.patternSignalsTitle}\n` + patterns.map(p =>
      `- ${p.date} ${p.pattern} (${p.note})`
    ).join('\n')
    : '';

  const strategiesText = strategies.map(s =>
    `- **${s.strategy}**：\n  ${t.conditionLabel}：${s.condition}\n  ${t.logicLabel}：${s.logic}\n  ${t.riskLabel}：${s.risk}`
  ).join('\n\n');

  const report = `
${t.professionalReportIntro.replace('{stock_id}', stock_id)}

${t.basicStatsTitle}
- ${t.avgVolume.replace('{avgVolume}', Math.round(avgVolume)).replace('{volumeTrend}', volumeTrend)}
- ${t.avgSpread.replace('{avgSpread}', avgSpread.toFixed(2))}
- ${t.riseDays.replace('{positive_days}', positive_days)}，${t.fallDays.replace('{negative_days}', negative_days)}，${t.flatDays.replace('{neutral_days}', neutral_days)}
- ${t.overallTrend.replace('{trendDescription}', trendDescription)}

${t.extremesTitle}
- ${t.maxGain.replace('{maxGain}', maxGain.toFixed(2)).replace('{maxGainDay}', maxGainDay).replace('{maxGainClose}', maxGainClose.toFixed(2))}
- ${t.maxLoss.replace('{maxLoss}', maxLoss.toFixed(2)).replace('{maxLossDay}', maxLossDay).replace('{maxLossClose}', maxLossClose.toFixed(2))}
- ${t.highestVolumeDay.replace('{maxVolumeDay}', maxVolumeDay).replace('{maxVolume}', maxVolume.toFixed(0))}
- ${t.minMaxClose.replace('{minClose}', minClose.toFixed(2)).replace('{minCloseDay}', minCloseDay).replace('{maxClose}', maxClose.toFixed(2)).replace('{maxCloseDay}', maxCloseDay)}
- ${t.supportResistance.replace('{support}', support.toFixed(2)).replace('{resistance}', resistance.toFixed(2))}

${t.marketViewTitle}
- ${t.marketTone.replace('{marketView}', marketView)}
- ${t.volumeTrendHint.replace('{volumeTrend}', volumeTrend)}

${patternsText}

${t.strategiesTitle}
${strategiesText}

${t.summaryTitle}
${t.summaryContent}
`.trim();

  return report;
}
