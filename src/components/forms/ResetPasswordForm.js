// /src/components/forms/ResetPasswordForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Modal, Typography } from "antd";
import React, { useEffect, useState } from "react";
import PasswordInput from "../PasswordInput";

const { Text } = Typography;

const localeText = {
  "zh-hant": {
    newPasswordLabel: "新密碼",
    newPasswordRequired: "請輸入新密碼",
    confirmPasswordLabel: "確認新密碼",
    confirmPasswordRequired: "請再次輸入新密碼",
    passwordMismatch: "兩次輸入的密碼不一致",
    passwordTooShort: "密碼長度需至少 8 碼",
    resetBtn: "重設密碼",
    successMsg: "密碼重設成功！",
    errorMsg: "密碼重設失敗，請稍後再試",
    missingToken: "無效或缺少重設密碼 token",
    errorModalTitle: "密碼重設失敗",
    errorModalOk: "確定",
    passwordStrengthLabel: "密碼強度：",
    strengthTexts: ["非常弱", "弱", "中等", "強", "非常強"],
  },
  en: {
    newPasswordLabel: "New Password",
    newPasswordRequired: "Please enter your new password",
    confirmPasswordLabel: "Confirm New Password",
    confirmPasswordRequired: "Please confirm your new password",
    passwordMismatch: "The two passwords do not match",
    passwordTooShort: "Password must be at least 8 characters",
    resetBtn: "Reset Password",
    successMsg: "Password reset successfully!",
    errorMsg: "Failed to reset password, please try again later",
    missingToken: "Missing or invalid reset token",
    errorModalTitle: "Reset Password Failed",
    errorModalOk: "OK",
    passwordStrengthLabel: "Password Strength: ",
    strengthTexts: ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"],
  },
  ja: {
    newPasswordLabel: "新しいパスワード",
    newPasswordRequired: "新しいパスワードを入力してください",
    confirmPasswordLabel: "新しいパスワードの確認",
    confirmPasswordRequired: "もう一度新しいパスワードを入力してください",
    passwordMismatch: "入力されたパスワードが一致しません",
    passwordTooShort: "パスワードは8文字以上である必要があります",
    resetBtn: "パスワードをリセット",
    successMsg: "パスワードのリセットに成功しました！",
    errorMsg: "パスワードのリセットに失敗しました。後でもう一度お試しください",
    missingToken: "リセットトークンが無効か不足しています",
    errorModalTitle: "パスワードリセット失敗",
    errorModalOk: "確定",
    passwordStrengthLabel: "Password Strength: ",
    strengthTexts: ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"],
  },
};

export default function ResetPasswordForm({ onSuccess }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  // 錯誤訊息以 Modal 呈現
  const [errorModalVisible, setErrorModalVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  // 從 URL 提取 token
  const [token, setToken] = useState(null);
  // 密碼強度分數，僅用於新密碼欄位
  const [passwordScore, setPasswordScore] = useState(0);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const tokenFromUrl = params.get("token");
    setToken(tokenFromUrl);
  }, []);

  // 新密碼欄位 onChange 以計算密碼強度
  const handlePasswordChange = (e) => {
    const value = e.target.value;
    if (!value) {
      setPasswordScore(0);
    } else if (value.length < 6) {
      setPasswordScore(1);
    } else if (value.length < 10) {
      setPasswordScore(2);
    } else if (value.length < 14) {
      setPasswordScore(3);
    } else {
      setPasswordScore(4);
    }
  };

  // 客戶端密碼驗證規則
  const passwordRules = [
    {
      required: true,
      message: text.newPasswordRequired,
    },
    {
      validator: async (_, value) => {
        if (value && value.length < 8) {
          return Promise.reject(new Error(text.passwordTooShort));
        }
        return Promise.resolve();
      },
    },
  ];

  const onFinish = async (values) => {
    setSuccessMessage("");
    setErrorMessage("");
    if (!token) {
      setErrorMessage(text.missingToken);
      setErrorModalVisible(true);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/reset-password", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": currentLocale,
        },
        body: JSON.stringify({
          token: token,
          new_password: values.newPassword,
        }),
      });
      if (!res.ok) {
        const data = await res.json();
        console.log("Error response:", data);
        const errMsg =
          data.error ||
          (data.detail && typeof data.detail === "object" ? data.detail.error : data.detail) ||
          text.errorMsg;
        throw new Error(errMsg);
      }
      await res.json();
      setSuccessMessage(text.successMsg);
      form.resetFields();
      if (onSuccess) {
        onSuccess();
      }
      setTimeout(() => {
        window.location.href = "/";
      }, 1500);
    } catch (error) {
      console.error("Reset password error:", error);
      setErrorMessage(error.message || text.errorMsg);
      setErrorModalVisible(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Form
        form={form}
        onFinish={onFinish}
        layout="vertical"
        style={{ maxWidth: 400, margin: "0 auto" }}
      >
        {successMessage && (
          <Alert
            style={{ marginBottom: 10 }}
            message={successMessage}
            type="success"
            showIcon
          />
        )}
        <Form.Item
          label={text.newPasswordLabel}
          name="newPassword"
          rules={passwordRules}
        >
          <PasswordInput onChange={handlePasswordChange} />
        </Form.Item>
        {passwordScore > 0 && (
          <Text type="secondary" style={{ marginBottom: 16, display: "block" }}>
            {text.passwordStrengthLabel}{text.strengthTexts[passwordScore]}
          </Text>
        )}
        <Form.Item
          label={text.confirmPasswordLabel}
          name="confirmPassword"
          dependencies={["newPassword"]}
          rules={[
            { required: true, message: text.confirmPasswordRequired },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue("newPassword") === value) {
                  return Promise.resolve();
                }
                return Promise.reject(new Error(text.passwordMismatch));
              },
            }),
          ]}
        >
          <PasswordInput hideStrength />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" block loading={loading}>
            {text.resetBtn}
          </Button>
        </Form.Item>
      </Form>

      {/* 錯誤訊息 Modal */}
      <Modal
        open={errorModalVisible}
        title={text.errorModalTitle}
        onCancel={() => setErrorModalVisible(false)}
        footer={[
          <Button key="ok" type="primary" onClick={() => setErrorModalVisible(false)}>
            {text.errorModalOk}
          </Button>,
        ]}
      >
        <p>{errorMessage}</p>
      </Modal>
    </>
  );
}
