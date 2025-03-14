// src/components/Dashboard/ApiKey/CreateTokenModal.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Checkbox, Form, Input, InputNumber, Modal, Select, Space } from "antd";
import React from "react";
import { apiKeyLocale } from "./locales";

export default function CreateTokenModal({
  visible,
  onCancel,
  onSubmit,
  loading = false,
  defaultValues = {
    usage_plan_id: 1,
    expires_minutes: 60,
    isPermanent: false,
  },
}) {
  const [form] = Form.useForm();
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  return (
    <Modal
      title={text.createModalTitle}
      open={visible}
      onCancel={onCancel}
      footer={null}
      destroyOnClose
      afterClose={() => form.resetFields()}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={defaultValues}
        onFinish={onSubmit}
      >
        <Form.Item
          label={text.formTokenNameLabel}
          name="name"
          tooltip={text.formTokenNameTooltip}
        >
          <Input placeholder={text.formTokenNamePlaceholder} />
        </Form.Item>

        <Form.Item
          label={text.formPlanLabel}
          name="usage_plan_id"
          rules={[{ required: true }]}
        >
          <Select>
            <Select.Option value={1}>{text.planBasic}</Select.Option>
            <Select.Option value={2}>{text.planProfessional}</Select.Option>
            <Select.Option value={3}>{text.planPayAsYouGo}</Select.Option>
          </Select>
        </Form.Item>

        <Form.Item
          label={text.formPermanentLabel}
          name="isPermanent"
          valuePropName="checked"
        >
          <Checkbox>{text.formPermanentCheckbox}</Checkbox>
        </Form.Item>

        <Form.Item
          label={text.formExpiryLabel}
          name="expires_minutes"
          rules={[
            { required: true, message: text.formExpiryValidationMessage },
            { type: "number", min: 10, max: 999999 },
          ]}
        >
          <InputNumber style={{ width: "100%" }} />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button onClick={onCancel}>{text.cancelButton}</Button>
            <Button type="primary" htmlType="submit" loading={loading}>
              {text.createButton}
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
}
