// components/forms/RegisterForm.js
import { Button, Form, Input } from "antd";
import React from "react";

export default function RegisterForm({ onRegister, onSuccess, loading }) {
  const onFinish = async (values) => {
    const ok = await onRegister(values.username, values.password);
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
        ]}
      >
        <Input.Password />
      </Form.Item>

      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading}>
          註冊
        </Button>
      </Form.Item>
    </Form>
  );
}
