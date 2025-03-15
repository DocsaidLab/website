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

const { TabPane } = Tabs;

const apiKeyLocale = {
  "zh-hant": {
    headerTitle: "我的 API Keys",
    headerDescription: "管理、檢視、撤銷與刪除你的公開 Token",
    currentPlanLabel: "目前方案",
    toggleShowTokens: "顯示全部 Token",
    toggleHideTokens: "隱藏全部 Token",
    createTokenButton: "建立新 Token",
    collapseHeader: "我的 Token 列表",
    apiUsageExampleTitle: "API 使用範例",
    newTokenModalTitle: "以下是您的新 Token",
    newTokenModalDesc: "請複製並保存，關閉後無法再次查看。",
    copyTokenButton: "複製 Token",
    closeButton: "關閉",
    notLoggedIn: "尚未登入，無法申請 Token",
    tokenRevoked: "Token 已撤銷",
    tokenDeleted: "Token 已刪除",
    createTokenSuccessTitle: "Token 建立成功",
    copyFailure: "複製失敗",
    copySuccess: "已複製",
    emailNotVerified: "尚未驗證電子郵件，無法申請 API Token。",
    loadingProfile: "載入個人資料...",
    pleaseVerifyEmailTitle: "請先驗證電子郵件",
    pleaseVerifyEmailDesc: "您尚未完成電子郵件驗證，無法使用 API Token 功能。請前往 {0} 頁面完成驗證。",
    myInfoPage: "我的資訊",
    fetchUserProfileFailed: "載入個人資料失敗: {0}",
    fetchTokensFailed: "取得 Token 列表失敗: {0}",
    fetchUserUsageFailed: "取得用量失敗: {0}",
    createTokenFailed: "建立 Token 失敗: {0}",
    operationFailed: "操作失敗: {0}",
    notAvailable: "N/A",
    basicPlan: "基本 (免費)",
    payPerUsePlan: "隨用隨付",
    unknownPlan: "未知方案",
  },
  en: {
    headerTitle: "My API Keys",
    headerDescription: "Manage, view, revoke, and delete your public tokens",
    currentPlanLabel: "Current Plan",
    toggleShowTokens: "Show All Tokens",
    toggleHideTokens: "Hide All Tokens",
    createTokenButton: "Create New Token",
    collapseHeader: "My Token List",
    apiUsageExampleTitle: "API Usage Examples",
    newTokenModalTitle: "Your New Token",
    newTokenModalDesc: "Please copy and save it. It will not be shown again after closing.",
    copyTokenButton: "Copy Token",
    closeButton: "Close",
    notLoggedIn: "Not logged in, unable to create Token",
    tokenRevoked: "Token Revoked",
    tokenDeleted: "Token Deleted",
    createTokenSuccessTitle: "Token Created Successfully",
    copyFailure: "Copy failed",
    copySuccess: "Copied",
    emailNotVerified: "Email not verified, unable to create API Token.",
    loadingProfile: "Loading Profile...",
    pleaseVerifyEmailTitle: "Please verify your email",
    pleaseVerifyEmailDesc: "You have not verified your email, and cannot use the API Token feature. Please go to the {0} page to complete verification.",
    myInfoPage: "My Info",
    fetchUserProfileFailed: "Failed to load profile: {0}",
    fetchTokensFailed: "Failed to fetch tokens: {0}",
    fetchUserUsageFailed: "Failed to fetch usage: {0}",
    createTokenFailed: "Failed to create token: {0}",
    operationFailed: "Operation failed: {0}",
    notAvailable: "N/A",
    basicPlan: "Basic (Free)",
    payPerUsePlan: "Pay-As-You-Go",
    unknownPlan: "Unknown Plan",
  },
  ja: {
    headerTitle: "マイAPIキー",
    headerDescription: "公開トークンの管理、確認、取り消し、削除",
    currentPlanLabel: "Current Plan",
    toggleShowTokens: "全トークンを表示",
    toggleHideTokens: "全トークンを隠す",
    createTokenButton: "新規トークン作成",
    collapseHeader: "マイトークン一覧",
    apiUsageExampleTitle: "API利用例",
    newTokenModalTitle: "あなたの新しいトークン",
    newTokenModalDesc: "コピーして保存してください。閉じると再表示されません。",
    copyTokenButton: "トークンをコピー",
    closeButton: "閉じる",
    notLoggedIn: "未ログインのため、Token を作成できません",
    tokenRevoked: "Token は取り消されました",
    tokenDeleted: "Token は削除されました",
    createTokenSuccessTitle: "Token 作成成功",
    copyFailure: "コピー失敗",
    copySuccess: "コピーしました",
    emailNotVerified: "メールが認証されていないため、API Token を作成できません。",
    loadingProfile: "プロファイルを読み込み中...",
    pleaseVerifyEmailTitle: "メール認証をしてください",
    pleaseVerifyEmailDesc: "メール認証が完了していないため、API Token 機能を利用できません。{0} ページに進んで認証を完了してください。",
    myInfoPage: "マイ情報",
    fetchUserProfileFailed: "プロファイルの読み込みに失敗しました: {0}",
    fetchTokensFailed: "トークン一覧の取得に失敗しました: {0}",
    fetchUserUsageFailed: "使用状況の取得に失敗しました: {0}",
    createTokenFailed: "トークンの作成に失敗しました: {0}",
    operationFailed: "操作に失敗しました: {0}",
    notAvailable: "N/A",
    basicPlan: "ベーシック (無料)",
    payPerUsePlan: "従量課金",
    unknownPlan: "不明なプラン",
  },
};

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

/** 將 UTC 時間字串轉成本地時間 */
function formatToLocalTime(utcString) {
  if (!utcString) return "";
  const dt = new Date(utcString);
  if (Number.isNaN(dt.getTime())) return utcString;
  return dt.toLocaleString();
}

export default function DashboardApiKey() {
  const { token: userToken } = useAuth();
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
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
  const [newTokenModalVisible, setNewTokenModalVisible] = useState(false);
  const [latestCreatedToken, setLatestCreatedToken] = useState("");
  const [showTokenPlain, setShowTokenPlain] = useState(false);

  // ========================
  // 抓取使用者檔案 /auth/me
  // ========================
  const fetchUserProfile = useCallback(async () => {
    if (!userToken) return;
    setLoadingProfile(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/me", {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        throw new Error(text.fetchUserProfileFailed.replace("{0}", res.status));
      }
      const data = await res.json();
      setUserProfile(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoadingProfile(false);
    }
  }, [userToken, text]);

  // ========================
  // 抓取 token 列表 /my-tokens
  // ========================
  const fetchTokens = useCallback(async () => {
    if (!userToken) return;
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/public/token/my-tokens", {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        throw new Error(text.fetchTokensFailed.replace("{0}", res.status));
      }
      let data = await res.json();
      const now = new Date();
      data = data.map((tk) => {
        if (tk.expires_at) {
          const dt = new Date(tk.expires_at);
          if (dt <= now) {
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
  }, [userToken, text]);

  // ========================
  // 抓取用量 /user-usage
  // ========================
  const fetchUserUsage = useCallback(async () => {
    if (!userToken) return;
    try {
      const res = await fetch("https://api.docsaid.org/public/token/user-usage", {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || text.fetchUserUsageFailed.replace("{0}", res.status));
      }
      const usage = await res.json();
      setUserUsage(usage);
    } catch (err) {
      console.error(err);
    }
  }, [userToken, text]);

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
      message.error(text.emailNotVerified);
      return;
    }
    const { isLongTerm, expires_minutes, name } = formValues;
    const finalExpires = isLongTerm ? 525600 : expires_minutes;

    if (!userToken) {
      message.error(text.notLoggedIn);
      return;
    }

    setLoading(true);
    try {
      const params = new URLSearchParams({
        expires_minutes: finalExpires,
        name: name || "",
      });
      const res = await fetch(
        `https://api.docsaid.org/public/token/?${params.toString()}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${userToken}`,
            "Content-Type": "application/json",
          },
        }
      );
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || text.createTokenFailed.replace("{0}", res.status));
      }
      const data = await res.json();
      message.success(text.createTokenSuccessTitle);

      const newAccessToken = data.access_token;
      const jti = parseJti(newAccessToken) || `temp-${Date.now()}`;

      let newItem = {
        jti,
        is_active: true,
        expires_at: data.expires_at,
        name: name || "",
      };

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

      setLatestCreatedToken(newAccessToken);
      setNewTokenModalVisible(true);
      setCreateModalVisible(false);
    } catch (err) {
      message.error(err.message || text.createTokenFailed.replace("{0}", ""));
    } finally {
      setLoading(false);
    }
  };

  // ========================
  // Revoke / Remove Token
  // ========================
  const handleRevokeOrDelete = async (tokenItem) => {
    if (!userToken) return;
    setLoading(true);
    const endpoint = tokenItem.is_active ? "revoke" : "remove";
    try {
      const res = await fetch(`https://api.docsaid.org/public/token/${endpoint}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${userToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jti: tokenItem.jti }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || text.operationFailed.replace("{0}", res.status));
      }
      message.success(tokenItem.is_active ? text.tokenRevoked : text.tokenDeleted);
      await fetchTokens();
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ========================
  // 複製 Token（新建後 Modal 使用）
  // ========================
  const copyToken = async (tokenStr) => {
    if (!tokenStr) {
      message.error(text.copyFailure);
      return;
    }
    try {
      await navigator.clipboard.writeText(tokenStr);
      message.success(text.copySuccess);
    } catch {
      message.error(text.copyFailure);
    }
  };

  // 遮罩 jti（若未顯示明碼）
  const maskToken = (val) => {
    if (!val) return text.notAvailable;
    if (showTokenPlain) return val;
    if (val.length < 10) {
      return val.slice(0, 2) + "****" + val.slice(-2);
    }
    return val.slice(0, 6) + "****" + val.slice(-4);
  };

  // 顯示計費方案（使用 i18n）
  function getPlanLabel(billingType) {
    switch (billingType) {
      case "rate_limit":
        return text.basicPlan;
      case "pay_per_use":
        return text.payPerUsePlan;
      default:
        return text.unknownPlan;
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
        <Spin tip={text.loadingProfile} />
      </div>
    );
  }

  if (userProfile && !userProfile.is_email_verified) {
    return (
      <div className={styles.apiKeyContainer} style={{ textAlign: "center", marginTop: 50 }}>
        <h2>{text.pleaseVerifyEmailTitle}</h2>
        <p>{text.pleaseVerifyEmailDesc.replace("{0}", text.myInfoPage)}</p>
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
          <List
            loading={loading}
            grid={{ gutter: 24, column: 1 }}
            dataSource={apiKeys}
            rowKey={(item) => item.jti}
            renderItem={(item) => {
              const localExpires = item.expires_at ? formatToLocalTime(item.expires_at) : "";
              return (
                <List.Item style={{ marginBottom: 24 }}>
                  <TokenCard
                    item={{ ...item, expires_local: localExpires }}
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
        title={text.newTokenModalTitle}
        open={newTokenModalVisible}
        onCancel={() => setNewTokenModalVisible(false)}
        footer={null}
        destroyOnClose
      >
        <p style={{ marginBottom: 10 }}>{text.newTokenModalDesc}</p>
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
          {text.copyTokenButton}
        </Button>
        <Button type="primary" onClick={() => setNewTokenModalVisible(false)}>
          {text.closeButton}
        </Button>
      </Modal>
    </div>
  );
}
