// /components/forms/ForgotPasswordForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Input } from "antd";
import React, { useState } from "react";

const localeText = {
  "zh-hant": {
    emailLabel: "Email",
    emailRequired: "請輸入經過驗證的 Email",
    submitBtn: "發送重設密碼信",
    successMsg: "重設密碼信已寄出，請檢查您的信箱！",
    errorMsg: "寄送失敗，請稍後再試",
  },
  en: {
    emailLabel: "Email",
    emailRequired: "Please enter your verified email",
    submitBtn: "Send Reset Password Email",
    successMsg: "Reset password email sent, please check your inbox!",
    errorMsg: "Failed to send, please try again later",
  },
  ja: {
    emailLabel: "メールアドレス",
    emailRequired: "確認済みのメールアドレスを入力してください",
    submitBtn: "パスワードリセットメールを送信する",
    successMsg: "パスワードリセットメールを送信しました。受信トレイを確認してください！",
    errorMsg: "送信に失敗しました。後でもう一度お試しください",
  },
};

export default function ForgotPasswordForm({ onSuccess }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const [form] = Form.useForm();
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const languageHeader = currentLocale;

  const onFinish = async (values) => {
    setSubmitError("");
    setSuccessMessage("");
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/forgot-password", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": languageHeader,
        },
        body: JSON.stringify({ email: values.email }),
      });
      if (!res.ok) {
        const data = await res.json();
        const errorMsg =
          (data.detail &&
            typeof data.detail === "object" &&
            data.detail.error) ||
          data.detail ||
          text.errorMsg;
        throw new Error(errorMsg);
      }
      await res.json();
      setSuccessMessage(text.successMsg);
      form.resetFields();
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      setSubmitError(error.message || text.errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form
      form={form}
      onFinish={onFinish}
      layout="vertical"
      style={{ maxWidth: 400, margin: "0 auto" }}
    >
      {submitError && (
        <Alert style={{ marginBottom: 10 }} message={submitError} type="error" showIcon />
      )}
      {successMessage && (
        <Alert style={{ marginBottom: 10 }} message={successMessage} type="success" showIcon />
      )}
      <Form.Item
        label={text.emailLabel}
        name="email"
        rules={[
          { required: true, message: text.emailRequired },
          { type: "email", message: text.emailRequired },
        ]}
      >
        <Input />
      </Form.Item>
      <Form.Item>
        <Button type="primary" block htmlType="submit" loading={loading}>
          {text.submitBtn}
        </Button>
      </Form.Item>
    </Form>
  );
}
