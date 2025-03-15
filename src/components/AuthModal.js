// /src/components/AuthModal.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Divider, Modal, Tabs, Typography } from "antd";
import React, { useEffect, useState } from "react";
import useAuthHandler from "../hooks/useAuthHandler";
import ForgotPasswordForm from "./forms/ForgotPasswordForm";
import LoginForm from "./forms/LoginForm";
import RegisterForm from "./forms/RegisterForm";

/**
 * 簡易字典，可根據專案需求擴充
 */
const i18nTexts = {
  "zh-hant": {
    modalTitle: "會員中心",
    loginTab: "登入",
    registerTab: "註冊",
    backToLogin: "回到登入",
  },
  en: {
    modalTitle: "Member Center",
    loginTab: "Login",
    registerTab: "Register",
    backToLogin: "Back to Login",
  },
  ja: {
    modalTitle: "会員センター",
    loginTab: "ログイン",
    registerTab: "新規登録",
    backToLogin: "ログインに戻る",
  },
};

export default function AuthModal({ visible, onCancel }) {
  const { login, register, loading } = useAuthHandler();
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const lang = currentLocale;
  const texts = i18nTexts[lang] || i18nTexts.en; // 預設英語
  const [mode, setMode] = useState("login");

  // 當 modal 關閉時，自動重置模式為 "login"
  useEffect(() => {
    if (!visible) {
      setMode("login");
    }
  }, [visible]);

  const goToForgotPassword = () => setMode("forgotPassword");
  const goToLogin = () => setMode("login");
  const goToRegister = () => setMode("register");

  const renderLoginContent = () => (
    <LoginForm
      onLogin={login}
      loading={loading}
      onSuccess={onCancel}
      onToggleForgotPassword={goToForgotPassword}
    />
  );

  const renderRegisterContent = () => (
    <RegisterForm
      onLogin={login}
      onRegister={register}
      loading={loading}
      onSuccess={onCancel}
    />
  );

  const renderForgotPasswordContent = () => (
    <>
      <ForgotPasswordForm onSuccess={onCancel} />
      <Divider />
      <Typography.Text
        style={{ cursor: "pointer", color: "#1890ff" }}
        onClick={goToLogin}
      >
        {texts.backToLogin}
      </Typography.Text>
    </>
  );

  return (
    <Modal
      open={visible}
      title={texts.modalTitle} // 動態顯示標題
      onCancel={onCancel}
      footer={null}
    >
      {/* 若當前是忘記密碼模式，就不使用 Tabs */}
      {mode === "forgotPassword" ? (
        renderForgotPasswordContent()
      ) : (
        <Tabs
          activeKey={mode}
          onChange={(key) => setMode(key)}
          items={[
            {
              key: "login",
              label: texts.loginTab, // 動態顯示「登入」
              children: renderLoginContent(),
            },
            {
              key: "register",
              label: texts.registerTab, // 動態顯示「註冊」
              children: renderRegisterContent(),
            },
          ]}
        />
      )}
    </Modal>
  );
}
