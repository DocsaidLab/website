// src/components/Dashboard/ApiDocs/index.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Spin } from "antd";
import React, { useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";
import ApiUsageExamples from "./ApiUsageExamples";
import styles from "./index.module.css";

const locales = {
  "zh-hant": {
    title: "API 使用說明",
    description: "這裡是各種 API 的技術文件與使用範例介紹頁。",
    pleaseVerifyEmailTitle: "請先驗證電子郵件",
    pleaseVerifyEmailDesc:
      "您尚未完成電子郵件驗證，無法使用 API 文件功能。請前往 我的資訊 頁面完成驗證。",
    loadingProfile: "載入個人資料...",
  },
  en: {
    title: "API Documents",
    description:
      "This page provides technical documentation and usage examples for various APIs.",
    pleaseVerifyEmailTitle: "Please verify your email",
    pleaseVerifyEmailDesc:
      "You have not verified your email, and cannot use the API Documentation feature. Please go to My Info page to complete verification.",
    loadingProfile: "Loading profile...",
  },
  ja: {
    title: "APIドキュメント",
    description: "ここでは、各種APIの技術文書と使用例を紹介しています。",
    pleaseVerifyEmailTitle: "メール認証をしてください",
    pleaseVerifyEmailDesc:
      "メール認証が完了していないため、APIドキュメント機能が利用できません。マイ情報ページに移動して認証を完了してください。",
    loadingProfile: "プロファイルを読み込み中...",
  },
};

export default function DashboardApiDocs() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = locales[currentLocale] || locales.en;

  const { token: userToken } = useAuth();
  const [userProfile, setUserProfile] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(false);

  useEffect(() => {
    if (userToken) {
      const fetchUserProfile = async () => {
        setLoadingProfile(true);
        try {
          const res = await fetch("https://api.docsaid.org/auth/me", {
            headers: { Authorization: `Bearer ${userToken}` },
          });
          if (!res.ok) {
            throw new Error("Failed to fetch profile");
          }
          const data = await res.json();
          setUserProfile(data);
        } catch (err) {
          console.error(err);
        } finally {
          setLoadingProfile(false);
        }
      };
      fetchUserProfile();
    }
  }, [userToken]);

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
        <p>{text.pleaseVerifyEmailDesc}</p>
      </div>
    );
  }

  return (
    <div className={styles.apiKeyContainer}>
      <h2>{text.title}</h2>
      <p>{text.description}</p>
      <ApiUsageExamples />
    </div>
  );
}
