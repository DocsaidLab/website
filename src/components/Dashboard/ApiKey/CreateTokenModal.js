// src/components/Dashboard/ApiKey/CreateTokenModal.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Checkbox, Form, Input, InputNumber, Modal, Space } from "antd";
import PropTypes from "prop-types";
import React, { useEffect } from "react";

const apiKeyLocale = {
  "zh-hant": {
    createModalTitle: "建立新的公開 Token",
    formTokenNameLabel: "Token 名稱",
    formTokenNameTooltip: "可給 Token 一個易識別的名稱",
    formTokenNamePlaceholder: "如：My DocAligner Key",
    formIsLongTermLabel: "申請一年效期",
    formIsLongTermCheckbox: "勾選後自動設定 1 年效期 (525600 分鐘)",
    formExpiryLabel: "有效期 (分鐘)",
    formExpiryValidationMessage: "請輸入有效期",
    cancelButton: "取消",
    createButton: "建立",
  },
  en: {
    createModalTitle: "Create New Public Token",
    formTokenNameLabel: "Token Name",
    formTokenNameTooltip: "Give the token an easily recognizable name",
    formTokenNamePlaceholder: "e.g., My DocAligner Key",
    formIsLongTermLabel: "Apply for a one-year term",
    formIsLongTermCheckbox: "Check to automatically set a 1-year term (525600 minutes)",
    formExpiryLabel: "Expiry (minutes)",
    formExpiryValidationMessage: "Please enter the expiration time",
    cancelButton: "Cancel",
    createButton: "Create",
  },
  ja: {
    createModalTitle: "新しい公開トークンを作成",
    formTokenNameLabel: "トークン名",
    formTokenNameTooltip: "トークンにわかりやすい名前を付けることができます",
    formTokenNamePlaceholder: "例：My DocAligner Key",
    formIsLongTermLabel: "一年間の有効期限を申請",
    formIsLongTermCheckbox: "チェックすると、自動的に1年間の有効期限（525600分）を設定します",
    formExpiryLabel: "有効期限（分）",
    formExpiryValidationMessage: "有効期限を入力してください",
    cancelButton: "キャンセル",
    createButton: "作成",
  },
};

export default function CreateTokenModal({
  visible,
  onCancel,
  onSubmit,
  loading = false,
  defaultValues = {
    expires_minutes: 60,
    isLongTerm: false,
  },
}) {
  const [form] = Form.useForm();
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  useEffect(() => {
    form.setFieldsValue(defaultValues);
  }, [defaultValues, form]);

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
          label={text.formIsLongTermLabel}
          name="isLongTerm"
          valuePropName="checked"
        >
          <Checkbox>{text.formIsLongTermCheckbox}</Checkbox>
        </Form.Item>

        <Form.Item
          label={text.formExpiryLabel}
          name="expires_minutes"
          rules={[
            { required: true, message: text.formExpiryValidationMessage },
            { type: "number", min: 10, max: 525600 },
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

CreateTokenModal.propTypes = {
  visible: PropTypes.bool.isRequired,
  onCancel: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  loading: PropTypes.bool,
  defaultValues: PropTypes.shape({
    expires_minutes: PropTypes.number,
    isLongTerm: PropTypes.bool,
    name: PropTypes.string,
  }),
};
