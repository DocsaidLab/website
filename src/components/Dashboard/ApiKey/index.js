import {
  CopyOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, List, message, Modal, Spin, Tabs } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";

import ApiUsageExamples from "./ApiUsageExamples";
import CreateTokenModal from "./CreateTokenModal";
import TokenCard from "./TokenCard";
import UsageOverview from "./UsageOverview";
import styles from "./index.module.css";
import { apiKeyLocale } from "./locales";

// 後端路徑
const PROFILE_URL = "https://api.docsaid.org/auth/me";
const API_BASE_URL = "https://api.docsaid.org/public/token";

const { TabPane } = Tabs;

/** 解析 JWT 中的 jti（供新建 Token 時使用） */
function parseJti(jwtStr) {
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

/** 將 UTC 時間字串轉成本地時間 (不顯示「永久」字樣) */
function formatToLocalTime(utcString) {
  if (!utcString) return ""; // 若後端返回空 => 視為無期限，但避免誤導就留空
  const dt = new Date(utcString);
  if (Number.isNaN(dt.getTime())) return utcString; // 解析失敗就原樣
  return dt.toLocaleString(); // 可自行換成 dayjs/moment
}

export default function DashboardApiKey() {
  const { token: userToken } = useAuth();
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  // ========================
  // State
  // ========================
  const [userProfile, setUserProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(false);

  const [apiKeys, setApiKeys] = useState([]);
  const [userUsage, setUserUsage] = useState(null);

  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);

  // 顯示/隱藏「新建 Token 後」Modal，並保存本次創建的 Token 完整字串
  const [newTokenModalVisible, setNewTokenModalVisible] = useState(false);
  const [latestCreatedToken, setLatestCreatedToken] = useState("");

  // 是否顯示「明碼 jti」(可自行決定要不要保留此功能)
  const [showTokenPlain, setShowTokenPlain] = useState(false);

  // ========================
  // 抓取使用者檔案 /auth/me
  // ========================
  const fetchUserProfile = useCallback(async () => {
    if (!userToken) return;
    setLoadingProfile(true);
    try {
      const res = await fetch(PROFILE_URL, {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        throw new Error(`Fetch user profile failed: ${res.status}`);
      }
      const data = await res.json();
      setUserProfile(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoadingProfile(false);
    }
  }, [userToken]);

  // ========================
  // 抓取 token 列表 /my-tokens
  // ========================
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
      let data = await res.json();

      // 前端做「是否已自然過期」的檢查
      const now = new Date();
      data = data.map((tk) => {
        if (tk.expires_at) {
          const dt = new Date(tk.expires_at);
          if (dt <= now) {
            // 標記為過期
            return { ...tk, is_active: false, __frontend_expired: true };
          }
        }
        return tk;
      });

      setApiKeys(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [userToken]);

  // ========================
  // 抓取用量 /user-usage
  // ========================
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

  // ========================
  // 頁面初始載入
  // ========================
  useEffect(() => {
    fetchUserProfile();
    fetchTokens();
    fetchUserUsage();
  }, [fetchUserProfile, fetchTokens, fetchUserUsage]);

  // ========================
  // 申請新 Token
  // ========================
  const handleCreateToken = async (formValues) => {
    if (!userProfile) return;
    if (!userProfile.is_email_verified) {
      message.error("尚未驗證電子郵件，無法申請 API Token。");
      return;
    }
    const { isLongTerm, expires_minutes, name } = formValues;

    // 長期使用設定為一年 (525600 分鐘)
    const finalExpires = isLongTerm ? 525600 : expires_minutes;

    if (!userToken) {
      message.error(text.notLoggedIn);
      return;
    }

    setLoading(true);
    try {
      // 將 expires_minutes 與 name 以 query string 方式傳遞
      const params = new URLSearchParams({
        expires_minutes: finalExpires,
        name: name || ""
      });

      const res = await fetch(`${API_BASE_URL}/?${params.toString()}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${userToken}`,
          "Content-Type": "application/json"
        }
        // 參數已在 URL 中傳遞，故不需要傳 body 資料
      });

      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Create token failed: ${res.status}`);
      }
      const data = await res.json();

      message.success(text.createTokenSuccessTitle);

      // 後端回傳 { access_token, expires_at, token_type... }
      const newAccessToken = data.access_token;
      const jti = parseJti(newAccessToken) || `temp-${Date.now()}`;

      // 新增到 apiKeys (只存必要資訊)
      let newItem = {
        jti,
        is_active: true,
        expires_at: data.expires_at,
        name: name || ""
      };

      // 檢查是否已過期
      if (newItem.expires_at) {
        const now = new Date();
        const dt = new Date(newItem.expires_at);
        if (dt <= now) {
          newItem.is_active = false;
          newItem.__frontend_expired = true;
        }
      }

      setApiKeys((prev) => [newItem, ...prev]);

      setTimeout(async () => {
        await fetchTokens();
      }, 1000);

      // 顯示一次性的完整 Token
      setLatestCreatedToken(newAccessToken);
      setNewTokenModalVisible(true);
      setCreateModalVisible(false);
    } catch (err) {
      message.error(err.message || "Create token failed");
    } finally {
      setLoading(false);
    }
  };

  // ========================
  // Revoke / Remove
  // ========================
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
      message.success(
        tokenItem.is_active ? text.tokenRevoked : text.tokenDeleted
      );
      await fetchTokens();
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ========================
  // 複製 Token (於新建後的 Modal)
  // ========================
  const copyToken = async (tokenStr) => {
    if (!tokenStr) {
      message.error(text.copyFailure || "複製失敗");
      return;
    }
    try {
      await navigator.clipboard.writeText(tokenStr);
      // ★ 第 3 點：message.success => 提示「已複製」
      message.success("已複製");
    } catch {
      message.error(text.copyFailure || "複製失敗");
    }
  };

  // 遮罩 jti
  const maskToken = (val) => {
    if (!val) return "N/A";
    if (showTokenPlain) return val;
    if (val.length < 10) {
      return val.slice(0, 2) + "****" + val.slice(-2);
    }
    return val.slice(0, 6) + "****" + val.slice(-4);
  };

  // 顯示計費方案
  function getPlanLabel(billingType) {
    switch (billingType) {
      case "rate_limit":
        return "Basic (Free)";
      case "pay_per_use":
        return "Pay-As-You-Go";
      default:
        return "Unknown Plan";
    }
  }
  const renderPlanBox = () => {
    if (!userUsage) return null;
    const planLabel = getPlanLabel(userUsage.billing_type);
    return (
      <div className={styles.planBox}>
        <strong>{text.currentPlanLabel}: </strong> {planLabel}
      </div>
    );
  };

  // ========================
  // 主體渲染
  // ========================
  if (loadingProfile && !userProfile) {
    return (
      <div style={{ textAlign: "center", marginTop: 50 }}>
        <Spin tip="Loading Profile..." />
      </div>
    );
  }

  if (userProfile && !userProfile.is_email_verified) {
    return (
      <div className={styles.apiKeyContainer} style={{ textAlign: "center", marginTop: 50 }}>
        <h2>請先驗證電子郵件</h2>
        <p>
          您尚未完成電子郵件驗證，無法使用 API Token 功能。<br />
          請前往 <strong>MyInfo</strong> 頁面完成驗證。
        </p>
      </div>
    );
  }

  return (
    <div className={styles.apiKeyContainer}>
      {/* Header */}
      <header className={styles.header}>
        <h2>{text.headerTitle}</h2>
        <p>{text.headerDescription}</p>
      </header>

      {/* 顯示方案 & 用量 */}
      {renderPlanBox()}
      <UsageOverview userUsage={userUsage} />

      {/* 操作按鈕 */}
      <div className={styles.actions}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
          className={styles.createButton}
        >
          {text.createTokenButton}
        </Button>
        <Button onClick={() => setShowTokenPlain(!showTokenPlain)}>
          {showTokenPlain ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          {showTokenPlain ? text.toggleHideTokens : text.toggleShowTokens}
        </Button>
      </div>

      <Tabs defaultActiveKey="tokens" className={styles.mainTabs}>
        <TabPane tab={text.collapseHeader} key="tokens">
          {/* 第4點: 調整排列 / 間距 */}
          <List
            loading={loading}
            // 設定更大的 gutter，改成 1 欄 (若想 2 欄也可)
            grid={{ gutter: 24, column: 1 }}
            dataSource={apiKeys}
            rowKey={(item) => item.jti}
            renderItem={(item) => {
              // 將 expires_at (UTC) => 本地時間字串
              const localExpires = item.expires_at
                ? formatToLocalTime(item.expires_at)
                : ""; // 避免顯示 "forever" 改成空

              return (
                <List.Item style={{ marginBottom: 24 }}>
                  <TokenCard
                    item={{
                      ...item,
                      expires_local: localExpires,
                    }}
                    onRevokeOrDelete={handleRevokeOrDelete}
                    maskToken={maskToken}
                  />
                </List.Item>
              );
            }}
          />
        </TabPane>
        <TabPane tab={text.apiUsageExampleTitle} key="apiUsage">
          <ApiUsageExamples />
        </TabPane>
      </Tabs>

      {/* 建立 Token 的 Modal */}
      <CreateTokenModal
        visible={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onSubmit={handleCreateToken}
        loading={loading}
      />

      {/* 新建 Token 後一次性顯示 */}
      <Modal
        title={text.newTokenModalTitle || "以下是您的新 Token"}
        open={newTokenModalVisible}
        onCancel={() => setNewTokenModalVisible(false)}
        footer={null}
        destroyOnClose
      >
        <p style={{ marginBottom: 10 }}>
          {text.newTokenModalDesc || "請複製並保存，關閉後無法再次查看。"}
        </p>
        <div
          style={{
            wordBreak: "break-all",
            background: "#f5f5f5",
            padding: "8px 10px",
            borderRadius: 6,
            marginBottom: 16,
          }}
        >
          {latestCreatedToken}
        </div>
        <Button
          icon={<CopyOutlined />}
          onClick={() => copyToken(latestCreatedToken)}
          style={{ marginRight: 8 }}
        >
          {text.copyTokenButton || "複製 Token"}
        </Button>
        <Button type="primary" onClick={() => setNewTokenModalVisible(false)}>
          {text.closeButton || "關閉"}
        </Button>
      </Modal>
    </div>
  );
}
