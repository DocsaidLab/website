// src/components/Dashboard/ApiKey/UsageOverview.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Card, Progress } from "antd";
import React from "react";
import { apiKeyLocale } from "./locales";

export default function UsageOverview({ userUsage }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  if (!userUsage) {
    // 可依需求改為 skeleton 或 loading
    return (
      <Card style={{ marginBottom: 24, borderRadius: 8 }}>
        <p>{text.loadingUsage}</p>
      </Card>
    );
  }

  const { billing_type, used_this_hour, limit_per_hour, remaining } = userUsage;

  return (
    <Card style={{ marginBottom: 24, borderRadius: 8 }}>
      {billing_type === "rate_limit" ? (
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <Progress
            type="circle"
            percent={Math.min(100, (used_this_hour / limit_per_hour) * 100)}
            status={used_this_hour >= limit_per_hour ? "exception" : "normal"}
            width={80}
            format={() => `${used_this_hour}/${limit_per_hour}`}
          />
          <div>
            <h4 style={{ margin: 0 }}>{text.usageOverviewTitle}</h4>
            <p style={{ margin: 0 }}>
              {text.usageThisHourLabel}{used_this_hour}/{limit_per_hour}
              &emsp;•&emsp;
              {text.remainingLabel}{remaining}
            </p>
          </div>
        </div>
      ) : billing_type === "pay_per_use" ? (
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <Progress
            type="circle"
            percent={0}
            width={80}
            format={() => `${used_this_hour}`}
          />
          <div>
            <h4 style={{ margin: 0 }}>{text.usageOverviewTitle}</h4>
            <p style={{ margin: 0 }}>
              {text.usageThisHourLabel}{used_this_hour} (Pay-As-You-Go)
            </p>
          </div>
        </div>
      ) : (
        <p>Unknown billing type</p>
      )}
    </Card>
  );
}
