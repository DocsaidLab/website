// src/components/Dashboard/ApiKey/UsageOverview.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Card, Progress } from "antd";
import PropTypes from "prop-types";
import React from "react";

const apiKeyLocale = {
  "zh-hant": {
    loadingUsage: "載入使用情況...",
    usageOverviewTitle: "用量概覽",
    usageThisHourLabel: "本小時用量：",
    remainingLabel: "剩餘：",
  },
  en: {
    loadingUsage: "Loading usage...",
    usageOverviewTitle: "Usage Overview",
    usageThisHourLabel: "Usage this hour:",
    remainingLabel: "Remaining:",
  },
  ja: {
    loadingUsage: "使用状況を読み込み中...",
    usageOverviewTitle: "使用状況概観",
    usageThisHourLabel: "今時間の使用量：",
    remainingLabel: "残り：",
  },
};

const cardStyle = { marginBottom: 24, borderRadius: 8 };

export default function UsageOverview({ userUsage }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  if (!userUsage) {
    // 可依需求改為 skeleton 或 loading
    return (
      <Card style={cardStyle}>
        <p>{text.loadingUsage}</p>
      </Card>
    );
  }

  const { billing_type, used_this_hour, limit_per_hour, remaining } = userUsage;

  // 當 limit_per_hour 為 0 時避免除以 0 錯誤
  const percent =
    billing_type === "rate_limit" && limit_per_hour > 0
      ? Math.min(100, (used_this_hour / limit_per_hour) * 100)
      : 0;

  return (
    <Card style={cardStyle}>
      {billing_type === "rate_limit" ? (
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <Progress
            type="circle"
            percent={percent}
            status={used_this_hour >= limit_per_hour ? "exception" : "normal"}
            size={80}
            format={() => `${used_this_hour}/${limit_per_hour}`}
          />
          <div>
            <h4 style={{ margin: 0 }}>{text.usageOverviewTitle}</h4>
            <p style={{ margin: 0 }}>
              {text.usageThisHourLabel}
              {used_this_hour}/{limit_per_hour}
              &emsp;•&emsp;
              {text.remainingLabel}
              {remaining}
            </p>
          </div>
        </div>
      ) : billing_type === "pay_per_use" ? (
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <Progress
            type="circle"
            percent={0}
            size={80}
            format={() => `${used_this_hour}`}
          />
          <div>
            <h4 style={{ margin: 0 }}>{text.usageOverviewTitle}</h4>
            <p style={{ margin: 0 }}>
              {text.usageThisHourLabel}
              {used_this_hour} (Pay-As-You-Go)
            </p>
          </div>
        </div>
      ) : (
        <p>Unknown billing type</p>
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
};
