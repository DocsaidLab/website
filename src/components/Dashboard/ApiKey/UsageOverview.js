import { ReloadOutlined } from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Card, Divider, Progress } from "antd";
import PropTypes from "prop-types";
import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import styles from "./UsageOverview.module.css";

const apiKeyLocale = {
  "zh-hant": {
    loadingUsage: "載入使用情況...",
    usageOverviewTitle: "用量概覽",
    usageHistoryTitle: "歷史使用趨勢",
    usageThisHourLabel: "本小時用量：",
    remainingLabel: "剩餘：",
    refreshButton: "更新用量",
  },
  en: {
    loadingUsage: "Loading usage...",
    usageOverviewTitle: "Usage Overview",
    usageHistoryTitle: "Historical Usage Trends",
    usageThisHourLabel: "Usage this hour:",
    remainingLabel: "Remaining:",
    refreshButton: "Refresh",
  },
  ja: {
    loadingUsage: "使用状況を読み込み中...",
    usageOverviewTitle: "使用状況概観",
    usageHistoryTitle: "過去の使用傾向",
    usageThisHourLabel: "今時間の使用量：",
    remainingLabel: "残り：",
    refreshButton: "リフレッシュ",
  },
};

export default function UsageOverview({ userUsage, usageHistory, onRefresh }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  // 若後端尚未回傳 userUsage，顯示 loading 狀態
  if (!userUsage) {
    return (
      <Card className={styles.usageCard}>
        <p>{text.loadingUsage}</p>
      </Card>
    );
  }

  const { billing_type, used_this_hour, limit_per_hour, remaining } = userUsage;

  // 計算當前用量的百分比，避免 division by zero
  const percent =
    billing_type === "rate_limit" && limit_per_hour > 0
      ? Math.min(100, (used_this_hour / limit_per_hour) * 100)
      : 0;

  // 點擊刷新按鈕時，若沒有傳入 onRefresh 回呼，預設使用 console.log
  const handleRefreshClick = () => {
    if (onRefresh) {
      onRefresh();
    } else {
      console.log("No onRefresh callback provided.");
    }
  };

  // 你也可以在父元件就把時間轉好，再給 usageHistory
  // 如果此處仍是 UTC，可用 tickFormatter 轉為當地時間
  const formatLocalTime = (val) => {
    // 若父層已是 local time string，可直接回傳 val
    // 若 val 仍是 UTC 格式，如 "2025-03-14T12:00:00Z"，則可自行轉換
    const dt = new Date(val);
    if (Number.isNaN(dt.getTime())) {
      return val; // 無法轉換就直接回傳原字串
    }
    // 回傳當地時區的 HH:MM (或更完整格式)
    return dt.toLocaleString();
  };

  return (
    <Card className={styles.usageCard}>
      {/*
        1) 刷新按鈕放在最上方，使用 primary 樣式使其更顯眼
      */}
      <div className={styles.topBar}>
        <h3 className={styles.mainTitle}>{text.usageOverviewTitle}</h3>
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          onClick={handleRefreshClick}
          className={styles.refreshButton}
        >
          {text.refreshButton}
        </Button>
      </div>

      {/* 2) 用量進度顯示區 */}
      {billing_type === "rate_limit" ? (
        <div className={styles.usageHeader}>
          <Progress
            type="circle"
            percent={percent}
            status={used_this_hour >= limit_per_hour ? "exception" : "normal"}
            size={80}
            format={() => `${used_this_hour}/${limit_per_hour}`}
            className={styles.progressCircle}
          />
          <div className={styles.usageDetails}>
            <p className={styles.usageText}>
              {text.usageThisHourLabel} {used_this_hour}/{limit_per_hour}
              <span className={styles.divider}>•</span>
              {text.remainingLabel} {remaining}
            </p>
          </div>
        </div>
      ) : billing_type === "pay_per_use" ? (
        <div className={styles.usageHeader}>
          <Progress
            type="circle"
            percent={0}
            size={80}
            format={() => `${used_this_hour}`}
            className={styles.progressCircle}
          />
          <div className={styles.usageDetails}>
            <p className={styles.usageText}>
              {text.usageThisHourLabel} {used_this_hour} (Pay-As-You-Go)
            </p>
          </div>
        </div>
      ) : (
        <p className={styles.errorText}>Unknown billing type</p>
      )}

      {/* 3) 歷史用量折線圖 */}
      {usageHistory && usageHistory.length > 0 && (
        <>
          <Divider className={styles.dividerStyle} />
          <h4 className={styles.historyTitle}>{text.usageHistoryTitle}</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={usageHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 12, fill: "#666" }}
                tickFormatter={formatLocalTime}
              />
              <YAxis tick={{ fontSize: 12, fill: "#666" }} />
              <Tooltip
                labelFormatter={(label) => {
                  // ToolTip 也做本地時間顯示
                  return formatLocalTime(label);
                }}
              />
              <Line
                type="monotone"
                dataKey="used"
                stroke="#1890ff"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </Card>
  );
}

UsageOverview.propTypes = {
  userUsage: PropTypes.shape({
    billing_type: PropTypes.string.isRequired,
    used_this_hour: PropTypes.number.isRequired,
    limit_per_hour: PropTypes.number,
    remaining: PropTypes.number,
  }),
  usageHistory: PropTypes.arrayOf(
    PropTypes.shape({
      time: PropTypes.string.isRequired, // 從父層拿到的時間字串
      used: PropTypes.number.isRequired,
    })
  ),
  onRefresh: PropTypes.func,
};

UsageOverview.defaultProps = {
  userUsage: null,
  usageHistory: [],
  onRefresh: null,
};
