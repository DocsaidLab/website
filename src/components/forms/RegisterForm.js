// /components/forms/RegisterForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Input } from "antd";
import React, { useState } from "react";
import { useAuth } from "../../context/AuthContext";
import PasswordInput from "../PasswordInput";

const localeText = {
  "zh-hant": {
    usernameLabel: "帳號",
    usernameRequired: "請輸入帳號",
    passwordLabel: "密碼",
    passwordRequired: "請輸入密碼",
    confirmPasswordLabel: "確認密碼",
    confirmPasswordRequired: "請再次輸入密碼",
    passwordMismatch: "兩次輸入的密碼不一致",
    registerBtn: "註冊",
    passphraseHint: "建議使用可記憶的長密碼（如短語）提升安全性。",
    successMsg: "註冊成功！",
    pwnedWarning: "此密碼曾出現在外洩紀錄中，請使用更安全的密碼。",
    passwordTooShort: "密碼長度需至少 8 碼",
  },
  en: {
    usernameLabel: "Username",
    usernameRequired: "Please enter your username",
    passwordLabel: "Password",
    passwordRequired: "Please enter your password",
    confirmPasswordLabel: "Confirm Password",
    confirmPasswordRequired: "Please confirm your password",
    passwordMismatch: "The two passwords that you entered do not match",
    registerBtn: "Register",
    passphraseHint: "Consider using a memorable passphrase for better security.",
    successMsg: "Registration successful!",
    pwnedWarning: "This password has appeared in data breaches. Please use a more secure one.",
    passwordTooShort: "Password must be at least 8 characters",
  },
  ja: {
    usernameLabel: "ユーザー名",
    usernameRequired: "ユーザー名を入力してください",
    passwordLabel: "パスワード",
    passwordRequired: "パスワードを入力してください",
    confirmPasswordLabel: "パスワードを再入力",
    confirmPasswordRequired: "パスワードをもう一度入力してください",
    passwordMismatch: "入力された2つのパスワードが一致しません",
    registerBtn: "登録",
    passphraseHint: "覚えやすいパスフレーズを使用してセキュリティを向上させましょう。",
    successMsg: "登録が完了しました！",
    pwnedWarning: "このパスワードは漏洩した履歴があります。より安全なものを使用してください。",
    passwordTooShort: "パスワードは8文字以上である必要があります",
  },
};

export default function RegisterForm({ onLogin, onSuccess, onRegister, loading }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const { loginSuccess } = useAuth();
  const [form] = Form.useForm();

  // 狀態
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const onFinish = async (values) => {
    setSubmitError("");
    setSuccessMessage("");

    // 這裡的 values 會包含 username, password, confirmPassword
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

      let dashboardPath = "/dashboard";
      if (currentLocale === "en") {
        dashboardPath = "/en/dashboard";
      } else if (currentLocale === "ja") {
        dashboardPath = "/ja/dashboard";
      }
      window.location.href = dashboardPath;
    } else {
      if (result.pwned) {
        setSubmitError(text.pwnedWarning);
      } else {
        setSubmitError(result.error || "Registration failed");
      }
    }
  };

  const finishRegisterSuccess = (result) => {
    setSuccessMessage(text.successMsg);
    form.resetFields();
    if (result.token) {
      loginSuccess(result.token);
    }
  };

  // 密碼欄位的基本規則
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
        <PasswordInput />
      </Form.Item>

      {/* 新增：確認密碼欄位 */}
      <Form.Item
        label={text.confirmPasswordLabel}
        name="confirmPassword"
        dependencies={["password"]} // 依賴 password 欄位
        hasFeedback
        rules={[
          {
            required: true,
            message: text.confirmPasswordRequired,
          },
          ({ getFieldValue }) => ({
            validator(_, value) {
              if (!value || getFieldValue("password") === value) {
                return Promise.resolve();
              }
              return Promise.reject(new Error(text.passwordMismatch));
            },
          }),
        ]}
      >
        <PasswordInput hideStrength={true} />
      </Form.Item>

      <Alert style={{ marginBottom: 10 }} message={text.passphraseHint} type="info" showIcon />

      {successMessage && (
        <Alert style={{ marginBottom: 10 }} message={successMessage} type="success" showIcon />
      )}
      {submitError && (
        <Alert style={{ marginBottom: 10 }} message={submitError} type="error" showIcon />
      )}

      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading}>
          {text.registerBtn}
        </Button>
      </Form.Item>
    </Form>
  );
}
