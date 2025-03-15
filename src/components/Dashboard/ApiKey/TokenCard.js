import { DeleteOutlined, ExclamationCircleOutlined } from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Card, Popconfirm } from "antd";
import PropTypes from "prop-types";
import React from "react";
import styles from "./index.module.css";
import { apiKeyLocale } from "./locales";

/**
 * 單一 Token 卡片組件
 */
export default function TokenCard({
  item,
  onRevokeOrDelete,
  maskToken,
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  const {
    name,
    is_active,
    expires_local,     // 前端轉換後的當地時區時間
    jti,
    __frontend_expired // 若前端判定該 token 已自然過期
  } = item;

  // 狀態
  //  1) 已過期 => 顯示「已過期」
  //  2) 已撤銷 => 顯示「已撤銷」(revoked)
  //  3) 否則 => "有效"
  let statusText = text.active;
  if (!is_active) {
    // 若 .__frontend_expired === true => 顯示「已過期」
    // 否則 => 「已撤銷」
    statusText = __frontend_expired ? (text.expired || "已過期") : text.revoked;
  }

  // Popconfirm
  const popConfirmTitle = is_active
    ? text.popconfirmRevokeTitle
    : text.popconfirmDeleteTitle;

  const buttonText = is_active ? text.revokeButton : text.deleteButton;

  return (
    <Card hoverable className={styles.tokenCard}>
      {/* 卡片 Header */}
      <div className={styles.tokenCardHeader}>
        <div className={styles.tokenName}>{name || text.defaultTokenName}</div>
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
        {/* 已移除複製功能 */}
        <span>{maskToken(jti)}</span>
      </div>

      <div className={styles.tokenRow}>
        <span className={styles.label}>{text.expiryLabel}</span>
        {expires_local || "-"}
      </div>

      <div className={styles.tokenRow}>
        <span className={styles.label}>{text.statusLabel}</span>
        {statusText}
      </div>
    </Card>
  );
}

TokenCard.propTypes = {
  item: PropTypes.shape({
    name: PropTypes.string,
    is_active: PropTypes.bool.isRequired,
    expires_local: PropTypes.string,
    jti: PropTypes.string.isRequired,
    __frontend_expired: PropTypes.bool,
  }).isRequired,
  onRevokeOrDelete: PropTypes.func.isRequired,
  maskToken: PropTypes.func.isRequired,
};
