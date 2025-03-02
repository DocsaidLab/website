import { FacebookOutlined, GoogleOutlined } from "@ant-design/icons";
import { Button, Divider, Modal, Space, Tabs, Typography } from "antd";
import React, { useState } from "react";
import useAuthHandler from "../hooks/useAuthHandler";
import ForgotPasswordForm from "./forms/ForgotPasswordForm";
import LoginForm from "./forms/LoginForm";
import RegisterForm from "./forms/RegisterForm";

export default function AuthModal({ visible, onCancel }) {
  // 用同一個 hook => 只管理登入 / 註冊 / 社群登入的 loading
  const { login, register, socialLogin, loading } = useAuthHandler();
  const [activeKey, setActiveKey] = useState("login");

  // 這裡額外用一個 state 記錄當前模式 => "login" / "register" / "forgotPassword"
  const [mode, setMode] = useState("login");

  const goToForgotPassword = () => setMode("forgotPassword");
  const goToLogin = () => setMode("login");
  const goToRegister = () => setMode("register");

  // 社群登入按鈕
  const socialButtons = (
    <Space style={{ width: "100%", justifyContent: "center" }}>
      <Button
        icon={<GoogleOutlined />}
        onClick={async () => {
          const success = await socialLogin("Google");
          if (success) onCancel?.();
        }}
        loading={loading}
      >
        Google
      </Button>
      <Button
        icon={<FacebookOutlined />}
        onClick={async () => {
          const success = await socialLogin("Facebook");
          if (success) onCancel?.();
        }}
        loading={loading}
      >
        Facebook
      </Button>
    </Space>
  );

  // 對應登入 / 註冊 UI
  const renderLoginContent = () => (
    <>
      <LoginForm onLogin={login} loading={loading} onSuccess={onCancel}
        // 讓 LoginForm 內可用 props 方式呼叫 goToForgotPassword
        onToggleForgotPassword={goToForgotPassword}
      />
      <Divider>或使用以下帳號登入</Divider>
      {socialButtons}
    </>
  );

  const renderRegisterContent = () => (
    <>
      <RegisterForm onRegister={register} loading={loading} onSuccess={onCancel} />
      <Divider>或使用以下帳號註冊 / 登入</Divider>
      {socialButtons}
    </>
  );

  // 忘記密碼畫面只需要一個 Email => 發送重設信
  const renderForgotPasswordContent = () => (
    <>
      <ForgotPasswordForm onSuccess={onCancel} />
      <Divider />
      <Typography.Text style={{ cursor: 'pointer', color: '#1890ff' }} onClick={goToLogin}>
        回到登入
      </Typography.Text>
    </>
  );

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
      destroyOnClose
    >
      {/* 這裡若想維持 Tabs，也可以把三種狀態都做成 Tabs */}
      {/* 不過常見做法是 forgotPassword 就脫離 Tabs，獨立顯示 */}

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
              children: renderLoginContent()
            },
            {
              key: "register",
              label: "註冊",
              children: renderRegisterContent()
            }
          ]}
        />
      )}
    </Modal>
  );
}
