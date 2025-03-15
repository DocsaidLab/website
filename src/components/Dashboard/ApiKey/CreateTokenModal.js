import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Checkbox, Form, Input, InputNumber, Modal, Space } from "antd";
import PropTypes from "prop-types";
import React, { useEffect } from "react";
import { apiKeyLocale } from "./locales";

export default function CreateTokenModal({
  visible,
  onCancel,
  onSubmit,
  loading = false,
  defaultValues = {
    expires_minutes: 60,
    isLongTerm: false, // 改用 isLongTerm
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

        {/* isLongTerm: 勾選即用 1年 */}
        <Form.Item
          label="申請一年效期" // 你可改任何文字
          name="isLongTerm"
          valuePropName="checked"
        >
          <Checkbox>勾選後自動設定 1 年效期 (525600 分鐘)</Checkbox>
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
