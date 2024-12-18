import { mean } from 'mathjs';

const I18N = {
  'zh-hant': {
    noData: "無資料",
    stable: "量能穩定",
    increase: "近期量能顯著放大",
    decrease: "近期量能明顯萎縮"
  },
  'en': {
    noData: "No data",
    stable: "Stable volume",
    increase: "Significant recent increase in volume",
    decrease: "Significant recent decrease in volume"
  },
  'ja': {
    noData: "データなし",
    stable: "出来高は安定",
    increase: "直近の出来高が顕著に拡大",
    decrease: "直近の出来高が明らかに減少"
  }
};

export function analyzeVolumeTrend(data, currentLocale = 'zh-hant') {
  const t = I18N[currentLocale] || I18N['zh-hant'];

  if (!Array.isArray(data) || data.length === 0) return t.noData;

  const volumes = data.map(d => d.Trading_Volume);
  const avgVol = mean(volumes);
  const recentCount = Math.min(data.length, 5);
  const recentVol = volumes.slice(-recentCount);
  const recentAvg = mean(recentVol);

  let volumeComment = t.stable;
  if (recentAvg > avgVol * 1.3) {
    volumeComment = t.increase;
  } else if (recentAvg < avgVol * 0.7) {
    volumeComment = t.decrease;
  }

  return volumeComment;
}
