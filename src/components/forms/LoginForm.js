// /components/forms/LoginForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Input, Typography } from "antd";
import React, { useState } from "react";


const localeText = {
  "zh-hant": {
    usernameLabel: "帳號",
    usernameError: "請輸入帳號",
    passwordLabel: "密碼",
    passwordRequired: "請輸入密碼",
    loginBtn: "登入",
    forgotPassword: "忘記密碼？",
    loginSuccessMsg: "登入成功！",
  },
  en: {
    usernameLabel: "Username",
    usernameError: "Please enter your username",
    passwordLabel: "Password",
    passwordRequired: "Please enter your password",
    loginBtn: "Login",
    forgotPassword: "Forgot password?",
    loginSuccessMsg: "Login successful!",
  },
  ja: {
    usernameLabel: "ユーザー名",
    usernameError: "ユーザー名を入力してください",
    passwordLabel: "パスワード",
    passwordRequired: "パスワードを入力してください",
    loginBtn: "ログイン",
    forgotPassword: "パスワードをお忘れですか？",
    loginSuccessMsg: "ログイン成功！",
  },
};

export default function LoginForm({
  onLogin,                // 呼叫後端 /auth/login 的函式
  onSuccess,             // 登入成功後的 callback (可用來跳轉 / 關閉 Modal)
  loading,               // 登入按鈕的加載狀態
  onToggleForgotPassword // 切換到「忘記密碼」畫面的函式 (若需要)
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;

  // 顯示錯誤與成功提示
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // antd form
  const [form] = Form.useForm();

  /**
   * 提交表單:
   *   1. 清空錯誤、成功提示
   *   2. 呼叫 onLogin(username, password)
   *   3. 若成功 => 顯示成功訊息 / 呼叫 onSuccess
   *   4. 若失敗 => 顯示錯誤訊息
   */
  const onFinish = async (values) => {
    setSubmitError("");
    setSuccessMessage("");

    const ok = await onLogin(values.username, values.password);
    if (ok) {
      // 登入成功
      setSuccessMessage(text.loginSuccessMsg);
      onSuccess?.(); // 若需要跳轉 => 在父層做 window.location.href 或關閉 Modal
    } else {
      // 登入失敗
      setSubmitError("帳號或密碼錯誤，請再試一次。");
    }
  };

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onFinish}
      style={{ maxWidth: 400, margin: "0 auto" }}
    >
      {/* 帳號欄位 */}
      <Form.Item
        label={text.usernameLabel}
        name="username"
        rules={[{ required: true, message: text.usernameError }]}
      >
        <Input />
      </Form.Item>

      {/* 密碼欄位 */}
      <Form.Item
        label={text.passwordLabel}
        name="password"
        rules={[{ required: true, message: text.passwordRequired }]}
      >
        <Input.Password />
      </Form.Item>

      {/* 登入成功訊息 */}
      {successMessage && (
        <Alert
          style={{ marginBottom: 10 }}
          message={successMessage}
          type="success"
          showIcon
        />
      )}

      {/* 錯誤訊息 */}
      {submitError && (
        <Alert
          style={{ marginBottom: 10 }}
          message={submitError}
          type="error"
          showIcon
        />
      )}

      {/* 提交按鈕 */}
      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading}>
          {text.loginBtn}
        </Button>
      </Form.Item>

      {/* 忘記密碼連結 */}
      {onToggleForgotPassword && (
        <Typography.Link
          style={{ float: "right", marginTop: 8 }}
          onClick={onToggleForgotPassword}
        >
          {text.forgotPassword}
        </Typography.Link>
      )}
    </Form>
  );
}
