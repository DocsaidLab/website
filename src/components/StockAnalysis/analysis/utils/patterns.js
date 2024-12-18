const I18N = {
  'zh-hant': {
    doji: 'Doji',
    dojiNote: '不確定性加大，可能變盤訊號'
  },
  'en': {
    doji: 'Doji',
    dojiNote: 'Increased uncertainty, possible trend reversal signal'
  },
  'ja': {
    doji: 'ドージ',
    dojiNote: '不確実性が高まり、トレンド転換の可能性'
  }
};

/**
 * 偵測 K 線圖中的特定型態（例如 Doji）
 * @param {Array} data 原始股票資料
 * @param {String} currentLocale 語系，可為 'zh-hant'|'en'|'ja'，預設 'zh-hant'
 * @returns {Array} 包含型態資訊的陣列
 */
export function detectPatterns(data, currentLocale = 'zh-hant') {
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (!Array.isArray(data) || data.length < 3) return [];

  const patterns = [];
  data.forEach(d => {
    const body = Math.abs(d.close - d.open);
    const range = d.max - d.min;

    // 簡單的 Doji 判斷：若實體(body)小於全幅(range)的5%，可視為 Doji
    if (range !== 0 && body < range * 0.05) {
      patterns.push({
        date: d.date,
        pattern: t.doji,
        note: t.dojiNote
      });
    }
  });

  return patterns;
}
