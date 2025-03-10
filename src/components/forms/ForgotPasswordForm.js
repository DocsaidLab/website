import { Button, Form, Input, message } from "antd";
import React, { useState } from "react";

export default function ForgotPasswordForm({ onSuccess }) {
  const [loading, setLoading] = useState(false);

  const onFinish = async (values) => {
    setLoading(true);
    try {
      // 呼叫後端的 /forgot-password API
      const res = await fetch("https://api.docsaid.org/auth/forgot-password", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // 後端預期接收的欄位是 "username"，
        // 這裡以表單填入的 email 當作 username 傳遞
        body: JSON.stringify({ username: values.email }),
      });
      if (!res.ok) {
        // 若回傳非 2xx，就嘗試解析回傳訊息並拋出錯誤
        const data = await res.json();
        throw new Error(data.detail || "寄送失敗，請稍後再試");
      }
      // API 正常回傳時，即視為成功
      message.success("重設密碼信已寄出，請檢查您的信箱！");
      onSuccess?.(); // 成功後關閉 Modal 或做其他操作
    } catch (error) {
      message.error(error.message || "寄送失敗，請稍後再試");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form onFinish={onFinish} layout="vertical">
      <Form.Item
        label="帳號 Email"
        name="email"
        rules={[
          { required: true, message: "請輸入 Email" },
          { type: "email", message: "Email 格式不正確" },
        ]}
      >
        <Input />
      </Form.Item>
      <Form.Item>
        <Button type="primary" block htmlType="submit" loading={loading}>
          發送重設密碼信
        </Button>
      </Form.Item>
    </Form>
  );
}
