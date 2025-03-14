// src/components/Dashboard/ApiKey/index.js
import {
  CopyOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  InfoCircleOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  Button,
  Collapse,
  Drawer,
  List,
  message
} from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";

import styles from "./index.module.css";
import { apiKeyLocale } from "./locales";

import CreateTokenModal from "./CreateTokenModal";
import DocAlignerPanel from "./DocAlignerPanel";
import TokenCard from "./TokenCard";
import UsageOverview from "./UsageOverview";

const API_BASE_URL = "https://api.docsaid.org/public/token";

export default function DashboardApiKey() {
  const { token: userToken } = useAuth();
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  // ===== Global states =====
  const [apiKeys, setApiKeys] = useState([]);
  const [userUsage, setUserUsage] = useState(null);
  const [loading, setLoading] = useState(false);

  // 控制 Token 明碼
  const [showTokenPlain, setShowTokenPlain] = useState(false);

  // 建立 Token Modal
  const [createModalVisible, setCreateModalVisible] = useState(false);

  // Drawer: 詳細資訊
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [detailToken, setDetailToken] = useState(null);

  // ===== DocAligner Panel: 公開 Token 測試 =====
  const [publicToken, setPublicToken] = useState("");
  const [usageData, setUsageData] = useState(null);
  const [checkLoading, setCheckLoading] = useState(false);

  // ---------------------------------------
  // 1) 載入 Token 列表
  // ---------------------------------------
  const fetchTokens = useCallback(async () => {
    if (!userToken) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/my-tokens`, {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        throw new Error(`Fetch tokens failed: ${res.status}`);
      }
      const data = await res.json();
      setApiKeys(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [userToken]);

  // ---------------------------------------
  // 2) 載入使用者整體用量
  // ---------------------------------------
  const fetchUserUsage = useCallback(async () => {
    if (!userToken) return;
    try {
      const res = await fetch(`${API_BASE_URL}/user-usage`, {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Fetch user usage error: ${res.status}`);
      }
      const usage = await res.json();
      setUserUsage(usage);
    } catch (err) {
      console.error(err);
    }
  }, [userToken]);

  useEffect(() => {
    fetchTokens();
    fetchUserUsage();
  }, [fetchTokens, fetchUserUsage]);

  // ---------------------------------------
  // 3) 建立 Token
  // ---------------------------------------
  const handleCreateToken = async (formValues) => {
    const { usage_plan_id, isPermanent, expires_minutes, name } = formValues;
    const finalExpires = isPermanent ? 999999 : expires_minutes;

    if (!userToken) {
      message.error(text.notLoggedIn);
      return;
    }

    setLoading(true);
    try {
      const planParam = `usage_plan_id=${usage_plan_id}`;
      const nameParam = name ? `&name=${encodeURIComponent(name)}` : "";
      const url = `${API_BASE_URL}/?${planParam}&expires_minutes=${finalExpires}${nameParam}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Create token failed: ${res.status}`);
      }
      const data = await res.json();

      message.success(text.createTokenSuccessTitle);
      // 也可用 Modal.success()
      // -> 省略: 你可自行放 Token 的提示

      const jti = _parseJti(data.access_token) || `temp-${Date.now()}`;
      const newItem = {
        jti,
        usage_plan_id: data.usage_plan_id || usage_plan_id,
        expires_at: data.expires_at,
        is_active: true,
        name: name || "",
      };

      setApiKeys((prev) => [newItem, ...prev]);
      setCreateModalVisible(false);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  function _parseJti(jwtStr) {
    try {
      const parts = jwtStr.split(".");
      if (parts.length !== 3) return null;
      const payloadRaw = atob(parts[1]);
      const payload = JSON.parse(payloadRaw);
      return payload.jti;
    } catch {
      return null;
    }
  }

  // ---------------------------------------
  // 4) 撤銷 / 刪除 Token
  // ---------------------------------------
  const handleRevokeOrDelete = async (tokenItem) => {
    if (!userToken) return;
    setLoading(true);
    const endpoint = tokenItem.is_active ? "revoke" : "remove";
    try {
      const res = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${userToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jti: tokenItem.jti }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Operation failed: ${res.status}`);
      }
      message.success(tokenItem.is_active ? text.tokenRevoked : text.tokenDeleted);
      await fetchTokens();

      // 如果 Drawer 正在顯示這個 Token，關閉
      if (detailToken && detailToken.jti === tokenItem.jti) {
        setDrawerVisible(false);
        setDetailToken(null);
      }
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ---------------------------------------
  // 5) Drawer 打開 / 關閉
  // ---------------------------------------
  const openDrawer = (tokenItem) => {
    setDetailToken(tokenItem);
    setDrawerVisible(true);
  };
  const closeDrawer = () => {
    setDrawerVisible(false);
    setDetailToken(null);
  };

  // ---------------------------------------
  // 6) 複製 Token
  // ---------------------------------------
  const copyToken = async (tokenId) => {
    try {
      await navigator.clipboard.writeText(tokenId);
      message.success(text.copySuccess);
    } catch {
      message.error(text.copyFailure);
    }
  };

  // ---------------------------------------
  // 7) 遮罩 Token
  // ---------------------------------------
  const maskToken = (val) => {
    if (!val) return "";
    if (showTokenPlain) return val;
    return val.slice(0, 6) + "****" + val.slice(-4);
  };

  // ---------------------------------------
  // 8) DocAlignerPanel：查詢 usage
  // ---------------------------------------
  const handleCheckUsage = async () => {
    if (!publicToken) {
      message.warning("請先輸入公開 Token");
      return;
    }
    setCheckLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/usage`, {
        headers: { Authorization: `Bearer ${publicToken}` },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Failed to get usage");
      }
      const data = await res.json();
      setUsageData(data);
      message.success("Usage updated!");
    } catch (err) {
      message.error(err.message);
      setUsageData(null);
    } finally {
      setCheckLoading(false);
    }
  };

  return (
    <div className={styles.apiKeyContainer}>
      {/* 頁面標題 */}
      <header className={styles.header}>
        <h2>{text.headerTitle}</h2>
        <p>{text.headerDescription}</p>
      </header>

      {/* 用量概覽 */}
      <UsageOverview userUsage={userUsage} />

      {/* 操作按鈕區 */}
      <div className={styles.actions}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
          style={{ marginRight: 16 }}
        >
          {text.createTokenButton}
        </Button>
        <Button onClick={() => setShowTokenPlain(!showTokenPlain)}>
          {showTokenPlain ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          {showTokenPlain ? text.toggleHideTokens : text.toggleShowTokens}
        </Button>
      </div>

      {/* Collapse: Token 列表 + DocAligner */}
      <Collapse bordered={false} className={styles.collapseRoot} defaultActiveKey={["tokens"]}>
        <Collapse.Panel
          key="tokens"
          header={
            <div style={{ display: "flex", alignItems: "center" }}>
              <InfoCircleOutlined style={{ marginRight: 8 }} />
              <span>{text.collapseHeader}</span>
            </div>
          }
        >
          <List
            loading={loading}
            grid={{ gutter: 16, column: 2 }}
            dataSource={apiKeys}
            rowKey={(item) => item.jti}
            renderItem={(item) => (
              <TokenCard
                item={item}
                onCopyToken={copyToken}
                onRevokeOrDelete={handleRevokeOrDelete}
                onOpenDetail={openDrawer}
                maskToken={maskToken}
              />
            )}
          />
        </Collapse.Panel>

        <Collapse.Panel key="docaligner" header="DocAligner Example">
          <DocAlignerPanel
            publicToken={publicToken}
            setPublicToken={setPublicToken}
            usageData={usageData}
            setUsageData={setUsageData}
            checkLoading={checkLoading}
            onCheckUsage={handleCheckUsage}
          />
        </Collapse.Panel>
      </Collapse>

      {/* 建立 Token 的 Modal */}
      <CreateTokenModal
        visible={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onSubmit={handleCreateToken}
        loading={loading}
      />

      {/* Drawer - Token 詳細資訊 */}
      <Drawer
        title={
          detailToken
            ? `${detailToken.name || text.defaultTokenName} - ${text.drawerDetail}`
            : text.drawerInfo
        }
        open={drawerVisible}
        onClose={closeDrawer}
        width={420}
      >
        {detailToken && (
          <>
            <p>
              <strong>{text.drawerTokenIdLabel}</strong>{" "}
              <a onClick={() => copyToken(detailToken.jti)}>
                {maskToken(detailToken.jti)}
                <CopyOutlined style={{ marginLeft: 6 }} />
              </a>
            </p>
            <p>
              <strong>{text.drawerExpiryLabel}</strong>{" "}
              {detailToken.expires_at || text.forever}
            </p>
            <p>
              <strong>{text.drawerStatusLabel}</strong>{" "}
              {detailToken.is_active ? text.active : text.revoked}
            </p>
          </>
        )}
      </Drawer>
    </div>
  );
}
