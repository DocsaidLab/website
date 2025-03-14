// src/components/Dashboard/ApiKey/TokenCard.jsx
import { CopyOutlined, DeleteOutlined, ExclamationCircleOutlined } from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Card, Popconfirm, Tooltip } from "antd";
import React from "react";
import styles from "./index.module.css";
import { apiKeyLocale } from "./locales";

/** 取得方案名稱的輔助函式 */
function getPlanName(id, text) {
  switch (id) {
    case 1:
      return text.planBasic;
    case 2:
      return text.planProfessional;
    case 3:
      return text.planPayAsYouGo;
    default:
      return text.planUnknown;
  }
}

/**
 * 單一 Token 卡片組件
 */
export default function TokenCard({
  item,
  onCopyToken,
  onRevokeOrDelete,
  onOpenDetail,
  maskToken,
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;
  const isActive = item.is_active;
  const planLabel = getPlanName(item.usage_plan_id, text);

  // 按鈕文字 & Popconfirm 標題
  const buttonText = isActive ? text.revokeButton : text.deleteButton;
  const popConfirmTitle = isActive
    ? text.popconfirmRevokeTitle
    : text.popconfirmDeleteTitle;

  return (
    <Card hoverable className={styles.tokenCard}>
      {/* 卡片 Header */}
      <div className={styles.tokenCardHeader}>
        <div>
          <div className={styles.tokenName}>{item.name || text.defaultTokenName}</div>
          <div className={styles.planLabel}>{planLabel}</div>
        </div>

        {/* Revoke / Delete 按鈕 */}
        <Popconfirm
          title={popConfirmTitle}
          icon={<ExclamationCircleOutlined style={{ color: "red" }} />}
          onConfirm={() => onRevokeOrDelete(item)}
        >
          <Button
            type="primary"
            danger
            size="small"
            icon={<DeleteOutlined />}
          >
            {buttonText}
          </Button>
        </Popconfirm>
      </div>

      {/* 主要資訊 Rows */}
      <div className={styles.tokenRow}>
        <span className={styles.label}>{text.tokenIdLabel}</span>
        <Tooltip title={text.tooltipCopyToken}>
          <a onClick={() => onCopyToken(item.jti)}>
            {maskToken(item.jti)}
            <CopyOutlined style={{ marginLeft: 6 }} />
          </a>
        </Tooltip>
      </div>

      <div className={styles.tokenRow}>
        <span className={styles.label}>{text.expiryLabel}</span>
        {item.expires_at || text.forever}
      </div>

      <div className={styles.tokenRow}>
        <span className={styles.label}>{text.statusLabel}</span>
        {isActive ? text.active : text.revoked}
      </div>

      {/* 額外資訊 / Usage Detail 按鈕 */}
      {isActive && onOpenDetail && (
        <div style={{ marginTop: 12, textAlign: "right" }}>
          <Button onClick={() => onOpenDetail(item)}>
            {text.detailUsageButton}
          </Button>
        </div>
      )}
    </Card>
  );
}
