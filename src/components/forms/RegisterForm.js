// /components/forms/RegisterForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Input, Progress } from "antd";
import React, { useState } from "react";
import zxcvbn from "zxcvbn";
import { useAuth } from "../../context/AuthContext";

function getPasswordScore(password) {
  if (!password) return 0;
  const result = zxcvbn(password);
  return result.score; // 0~4
}

// 多國語系
const localeText = {
  "zh-hant": {
    usernameLabel: "帳號",
    usernameRequired: "請輸入帳號",
    passwordLabel: "密碼",
    passwordRequired: "請輸入密碼",
    registerBtn: "註冊",
    passphraseHint: "建議使用可記憶的長密碼（如短語）提升安全性。",
    successMsg: "註冊成功！",
    pwnedWarning: "此密碼曾出現在外洩紀錄中，請使用更安全的密碼。",
    passwordTooShort: "密碼長度需至少 8 碼",
    passwordStrengthTitle: "密碼強度：",
    strengthTexts: ["非常弱", "弱", "中等", "強", "非常強"],
  },
  en: {
    usernameLabel: "Username",
    usernameRequired: "Please enter your username",
    passwordLabel: "Password",
    passwordRequired: "Please enter your password",
    registerBtn: "Register",
    passphraseHint: "Consider using a memorable passphrase for better security.",
    successMsg: "Registration successful!",
    pwnedWarning: "This password has appeared in data breaches. Please use a more secure one.",
    passwordTooShort: "Password must be at least 8 characters",
    passwordStrengthTitle: "Password Strength: ",
    strengthTexts: ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"],
  },
  ja: {
    usernameLabel: "ユーザー名",
    usernameRequired: "ユーザー名を入力してください",
    passwordLabel: "パスワード",
    passwordRequired: "パスワードを入力してください",
    registerBtn: "登録",
    passphraseHint: "覚えやすいパスフレーズを使用してセキュリティを向上させましょう。",
    successMsg: "登録が完了しました！",
    pwnedWarning: "このパスワードは漏洩した履歴があります。より安全なものを使用してください。",
    passwordTooShort: "パスワードは8文字以上である必要があります",
    passwordStrengthTitle: "パスワードの強度：",
    strengthTexts: ["非常に弱い", "弱い", "普通", "強い", "非常に強い"],
  },
};

export default function RegisterForm({
  onLogin,
  onSuccess,
  onRegister,
  loading
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;

  // 如果後端回 token，可配合 AuthContext 寫入 localStorage
  const { loginSuccess } = useAuth();

  // antd form
  const [form] = Form.useForm();

  // 狀態
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // 用來顯示即時強度
  const [passwordScore, setPasswordScore] = useState(0);

  /**
   * 表單送出
   * 不允許弱密碼 => 只要後端回 pwned=true => fail => 顯示錯誤
   */
  const onFinish = async (values) => {
    setSubmitError("");
    setSuccessMessage("");
    const result = await onRegister({
      username: values.username,
      password: values.password,
    });

    if (!result) {
      setSubmitError("Unknown error");
      return;
    }

    if (result.success) {
      finishRegisterSuccess(result);
      const ok = await onLogin(values.username, values.password);
      if (ok) {
        onSuccess?.();
      }
      window.location.href = "/dashboard";
    } else {
      // 若 pwned=true => 後端要求不允許 => 顯示錯誤
      if (result.pwned) {
        setSubmitError(text.pwnedWarning);
      } else {
        // 其他錯誤 (ex: 帳號重複)
        setSubmitError(result.error || "Registration failed");
      }
    }
  };

  /**
   * 註冊成功 => 重置表單 + (若需要) loginSuccess(token) + 跳轉
   */
  const finishRegisterSuccess = (result) => {
    setSuccessMessage(text.successMsg);
    form.resetFields();
    setPasswordScore(0);

    // 若後端回 token => 自動登入
    if (result.token) {
      loginSuccess(result.token);
    }
  };

  // 基本規則: 必填 & >=8 chars
  const passwordRules = [
    {
      required: true,
      message: text.passwordRequired,
    },
    {
      validator: async (_, value) => {
        if (!value) {
          return Promise.reject(new Error(text.passwordRequired));
        }
        if (value.length < 8) {
          return Promise.reject(new Error(text.passwordTooShort));
        }
        return Promise.resolve();
      },
    },
  ];

  // 即時強度
  const handlePasswordChange = (e) => {
    const pwd = e.target.value;
    setPasswordScore(getPasswordScore(pwd));
  };

  // zxcvbn: 0~4 => 0..100
  const progressPercent = passwordScore * 25;
  const strengthText = text.strengthTexts[passwordScore] || "";
  const strokeColor = [
    "#ff4d4f", // 0 = 非常弱
    "#ff7a45", // 1 = 弱
    "#faad14", // 2 = 中等
    "#52c41a", // 3 = 強
    "#1677ff", // 4 = 非常強
  ][passwordScore] || "#ff4d4f";

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onFinish}
      style={{ maxWidth: 400, margin: "0 auto" }}
    >
      <Form.Item
        label={text.usernameLabel}
        name="username"
        rules={[{ required: true, message: text.usernameRequired }]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        label={text.passwordLabel}
        name="password"
        hasFeedback
        rules={passwordRules}
      >
        <Input.Password onChange={handlePasswordChange} />
      </Form.Item>

      {/* 即時強度顯示 */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ marginBottom: 5 }}>
          {text.passwordStrengthTitle}
          <strong>{strengthText}</strong>
        </div>
        <Progress percent={progressPercent} showInfo={false} strokeColor={strokeColor} />
      </div>

      {/* 顯示建議：使用長密碼短語 */}
      <Alert
        style={{ marginBottom: 10 }}
        message={text.passphraseHint}
        type="info"
        showIcon
      />

      {/* 成功訊息 */}
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

      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading}>
          {text.registerBtn}
        </Button>
      </Form.Item>
    </Form>
  );
}
