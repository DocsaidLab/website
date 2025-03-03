// /src/components/AuthModal.js
import { Divider, Modal, Tabs, Typography } from "antd";
import React, { useState } from "react";
import useAuthHandler from "../hooks/useAuthHandler";
import ForgotPasswordForm from "./forms/ForgotPasswordForm";
import LoginForm from "./forms/LoginForm";
import RegisterForm from "./forms/RegisterForm";

export default function AuthModal({ visible, onCancel }) {
  // 1. 從自訂 Hook 取得登入 / 註冊 函式與 loading 狀態
  const { login, register, loading } = useAuthHandler();

  // 2. 目前顯示的分頁 (login / register / forgotPassword)
  const [mode, setMode] = useState("login");

  // 3. 切換畫面
  const goToForgotPassword = () => setMode("forgotPassword");
  const goToLogin = () => setMode("login");
  const goToRegister = () => setMode("register");

  // 4. 「登入」畫面內容
  const renderLoginContent = () => (
    <LoginForm
      onLogin={login}
      loading={loading}
      onSuccess={onCancel}
      onToggleForgotPassword={goToForgotPassword}
    />
  );

  // 5. 「註冊」畫面內容
  const renderRegisterContent = () => (
    <RegisterForm
      onLogin={login}
      onRegister={register}
      loading={loading}
      onSuccess={onCancel}
    />
  );

  // 6. 「忘記密碼」畫面內容
  const renderForgotPasswordContent = () => (
    <>
      <ForgotPasswordForm onSuccess={onCancel} />
      <Divider />
      <Typography.Text
        style={{ cursor: "pointer", color: "#1890ff" }}
        onClick={goToLogin}
      >
        回到登入
      </Typography.Text>
    </>
  );

  // 7. 依照 mode 顯示不同內容
  const renderContent = () => {
    switch (mode) {
      case "login":
        return renderLoginContent();
      case "register":
        return renderRegisterContent();
      case "forgotPassword":
        return renderForgotPasswordContent();
      default:
        return null;
    }
  };

  return (
    <Modal
      open={visible}
      title="會員中心"
      onCancel={onCancel}
      footer={null}
      // 注意：如果想保留密碼輸入狀態，可移除 destroyOnClose
      // destroyOnClose
    >
      {/* 如果當前是忘記密碼模式，就脫離 Tabs，獨立顯示 */}
      {mode === "forgotPassword" ? (
        renderContent()
      ) : (
        <Tabs
          activeKey={mode}
          onChange={(key) => setMode(key)}
          items={[
            {
              key: "login",
              label: "登入",
              children: renderLoginContent(),
            },
            {
              key: "register",
              label: "註冊",
              children: renderRegisterContent(),
            },
          ]}
        />
      )}
    </Modal>
  );
}
