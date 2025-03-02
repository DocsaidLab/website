// components/forms/LoginForm.js
import { Button, Form, Input, Typography } from "antd";
import React from "react";

const weakPasswords = [
  "123456", "password", "123456789", "12345678", "12345", "1234567", "qwerty",
  "abc123", "password1", "111111", "123123", "admin", "welcome", "iloveyou",
  "1q2w3e4r", "monkey", "sunshine", "letmein", "football", "dragon", "shadow",
  "1234", "princess", "baseball", "superman", "starwars"
];

export default function LoginForm({ onLogin, onSuccess, loading }) {
  const onFinish = async (values) => {
    const ok = await onLogin(values.username, values.password);
    if (ok) {
      onSuccess?.();
    }
  };

  return (
    <Form onFinish={onFinish} layout="vertical">
      <Form.Item
        label="帳號"
        name="username"
        rules={[{ required: true, message: "請輸入帳號" }]}
      >
        <Input />
      </Form.Item>

      <Form.Item
        label="密碼"
        name="password"
        rules={[
          { required: true, message: "請輸入密碼" },
          { min: 8, message: "密碼長度至少 8 碼" },
          {
            pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,64}$/,
            message: "密碼需包含大小寫字母、數字與特殊字符",
          },
          ({ getFieldValue }) => ({
            validator(_, value) {
              if (weakPasswords.includes(value?.toLowerCase())) {
                return Promise.reject(
                  new Error("此密碼過於常見，請使用更安全的密碼")
                );
              }
              return Promise.resolve();
            },
          }),
        ]}
      >
        <Input.Password />
      </Form.Item>

      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading}>
          登入
        </Button>
      </Form.Item>
      <Typography.Link
        style={{ float: "right", marginTop: 8 }}
        onClick={() => {
          // 假設要在同一個 Modal 內顯示 reset flow
          // 可透過 props 或 context 切換狀態
          onToggleForgotPassword?.();
        }}
      >
        忘記密碼？
      </Typography.Link>
    </Form>
  );
}
