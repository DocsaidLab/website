import { Button, Form, Input, message } from "antd";
import React, { useState } from "react";
import { forgotPasswordApi } from "../../utils/mockApi";

export default function ForgotPasswordForm({ onSuccess }) {
  const [loading, setLoading] = useState(false);

  const onFinish = async (values) => {
    setLoading(true);
    try {
      await forgotPasswordApi(values.email);
      message.success("重設密碼信已寄出，請檢查您的信箱！");
      onSuccess?.(); // 成功後關閉 Modal，或跳轉其它狀態
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
