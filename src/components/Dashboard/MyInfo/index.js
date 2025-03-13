import { UploadOutlined, UserOutlined } from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  Alert,
  Avatar,
  Button,
  Col,
  DatePicker,
  Divider,
  Form,
  Input,
  message,
  Modal,
  Row,
  Spin,
  Typography,
  Upload,
} from "antd";
import moment from "moment";
import React, { useCallback, useEffect, useState } from "react";
import PasswordInput from "../../../components/PasswordInput";
import { useAuth } from "../../../context/AuthContext";
import { changePasswordLocale, dashboardLocale, deleteAccountLocale } from "./locales";

const { Text } = Typography;

export default function DashboardMyInfo() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = dashboardLocale[currentLocale] || dashboardLocale.en;

  const {
    token,
    user,
    setUser,
    updateProfile,
    sendVerificationEmail,
    changePassword,
    deleteAccount,
  } = useAuth();

  const [infoLoading, setInfoLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [pwdModalVisible, setPwdModalVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [verificationModalVisible, setVerificationModalVisible] = useState(false);
  const [profileForm] = Form.useForm();
  const [pwdForm] = Form.useForm();
  const [uploadLoading, setUploadLoading] = useState(false);
  const [showEmailAlert, setShowEmailAlert] = useState(false);

  // 禁用未來日期
  const disabledFutureDates = useCallback(
    (current) => current && current.isAfter(moment(), "day"),
    []
  );

  // 上次登入資訊
  const lastLoginTime = user?.last_login_at
    ? moment(user.last_login_at).format("YYYY-MM-DD HH:mm") + " (UTC+0)"
    : text.notSet;
  const lastLoginIp = user?.last_login_ip || text.notSet;

  // =========== 1. 取得使用者資訊 ===========
  const refreshUserInfo = async () => {
    if (!token) return;
    setInfoLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/me", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        throw new Error(text.fetchUserInfoFailure);
      }
      const data = await res.json();
      setUser(data);
      // 若 email 存在但未驗證，顯示警示
      setShowEmailAlert(data.email && data.is_email_verified === false);
    } catch (err) {
      message.error(err.message || text.fetchUserInfoFailure);
    } finally {
      setInfoLoading(false);
    }
  };

  useEffect(() => {
    refreshUserInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // 若 user 存在但尚未設定 email，自動切換至編輯模式
  useEffect(() => {
    if (user && !user.email) {
      setEditing(true);
    }
  }, [user]);

  // =========== 2. 編輯個人資料 ===========
  const onEditProfile = () => {
    setEditing(true);
    if (user) {
      profileForm.setFieldsValue({
        username: user.username,
        email: user.email,
        phone: user.phone,
        birth: user.birth ? moment(user.birth, "YYYY-MM-DD") : null,
      });
    }
  };

  // =========== 3. 儲存個人資料 ===========
  const onSaveProfile = async (values) => {
    try {
      const birthString = values.birth
        ? moment(values.birth).format("YYYY-MM-DD")
        : null;
      const payload = { ...values, birth: birthString };
      const updatedUser = await updateProfile(payload);
      message.success(text.successMsg);
      setUser(updatedUser);
      setShowEmailAlert(
        updatedUser.email && updatedUser.is_email_verified === false
      );
      setEditing(false);
    } catch (err) {
      if (err.message.includes("Email already exists")) {
        profileForm.setFields([{ name: "email", errors: [err.message] }]);
      }
      message.error(err.message || text.fetchUserInfoFailure);
    }
  };

  // =========== 4. 上傳頭像 ===========
  const onUploadAvatar = async ({ file }) => {
    if (!token) return;
    setUploadLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("https://api.docsaid.org/auth/avatar", {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || text.avatarUploadFailure);
      }
      const result = await res.json();
      message.success(text.avatarUploadSuccess);
      setUser((prev) => ({
        ...prev,
        avatar: result.avatar,
      }));
    } catch (err) {
      message.error(err.message || text.avatarUploadFailure);
    } finally {
      setUploadLoading(false);
    }
  };

  // =========== 5. 變更密碼 ===========
  const openChangePwdModal = () => {
    setPwdModalVisible(true);
  };

  const [errorModalVisible, setErrorModalVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const onChangePassword = async (values) => {
    if (values.newPassword !== values.confirmPassword) {
      setErrorMessage(text.passwordMismatch);
      setErrorModalVisible(true);
      return;
    }
    try {
      await changePassword(values.oldPassword, values.newPassword);
      message.success(text.changePasswordSuccess);
      setPwdModalVisible(false);
      pwdForm.resetFields();
    } catch (err) {
      console.error("變更密碼錯誤:", err);
      const errorMsg =
        err.response?.data?.detail || err.message || text.changePasswordFailureTitle;
      setErrorMessage(errorMsg);
      setErrorModalVisible(true);
    }
  };

  // =========== 6. 重新寄送驗證信 ===========
  const onResendVerification = () => {
    if (!user?.email) return;
    setVerificationModalVisible(true);
  };

  const handleVerificationOk = async () => {
    try {
      await sendVerificationEmail(user.email);
      message.success(
        text.verificationModalSent || "驗證信已寄出，請檢查信箱"
      );
      setVerificationModalVisible(false);
    } catch (err) {
      message.error(err.message || "寄送驗證信失敗");
    }
  };

  // =========== 7. 刪除帳號 ===========
  const onDeleteAccount = async () => {
    try {
      await deleteAccount();
      message.success(text.deleteAccountSuccess);
      setDeleteModalVisible(false);
      window.location.href = "/";
    } catch (err) {
      message.error(err.message || "刪除帳號失敗");
    }
  };

  if (infoLoading) {
    return <Spin style={{ margin: "50px auto", display: "block" }} />;
  }

  // 修正 Alert 中替換字串的問題
  const renderNoEmailAlertDescription = () => {
    // 假設 text.noEmailAlertDesc 為 "請設定您的 Email，點擊 {editLink} 進行設定"
    const parts = text.noEmailAlertDesc.split("{editLink}");
    return (
      <>
        {parts[0]}
        <Button type="link" onClick={onEditProfile}>
          {text.editButton}
        </Button>
        {parts[1]}
      </>
    );
  };

  const renderEmailStatus = () => {
    if (!user?.email) {
      return <Text type="secondary">{text.notSet}</Text>;
    }
    if (user.is_email_verified) {
      return <Text style={{ color: "green" }}>{text.verified}</Text>;
    }
    return (
      <span style={{ color: "red" }}>
        {text.notVerified}
        <Button type="link" onClick={onResendVerification}>
          {text.resendVerification}
        </Button>
      </span>
    );
  };

  return (
    <div>
      <h2>{text.myInfoTitle}</h2>

      {showEmailAlert && user?.email && !user.is_email_verified && (
        <Alert
          style={{ marginBottom: 16 }}
          message={text.emailNotVerifiedAlertTitle}
          description={text.emailNotVerifiedAlertDesc}
          type="warning"
          showIcon
        />
      )}

      {!user?.email && !editing && (
        <Alert
          style={{ marginBottom: 16 }}
          message={text.noEmailAlertTitle}
          description={renderNoEmailAlertDescription()}
          type="info"
          showIcon
        />
      )}

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
            }}
          >
            <Avatar
              size={160}
              src={user?.avatar}
              icon={<UserOutlined />}
              onError={() => false}
            />
            <br />
            <Upload
              showUploadList={false}
              accept="image/*"
              customRequest={onUploadAvatar}
            >
              <Button icon={<UploadOutlined />} style={{ marginTop: 8 }}>
                {uploadLoading
                  ? text.uploadAvatarButtonUploading
                  : text.uploadAvatarButton}
              </Button>
            </Upload>
          </div>
        </Col>

        <Col span={16}>
          {editing ? (
            <Form
              form={profileForm}
              layout="vertical"
              onFinish={onSaveProfile}
              initialValues={{
                username: user?.username,
                email: user?.email,
                phone: user?.phone,
                birth: user?.birth ? moment(user.birth, "YYYY-MM-DD") : null,
              }}
            >
              <Form.Item
                label={text.accountLabel}
                name="username"
                rules={[{ required: true, message: text.usernameRequired }]}
              >
                <Input disabled />
              </Form.Item>
              <Form.Item
                label={text.emailLabel}
                name="email"
                rules={[
                  {
                    required: true,
                    message: text.emailRequired || "請輸入 Email",
                  },
                  {
                    type: "email",
                    message: text.invalidEmail || "Email 格式錯誤",
                  },
                ]}
              >
                {user?.is_email_verified === false ? <Input /> : <Input disabled />}
              </Form.Item>
              <Form.Item label={text.phoneLabel} name="phone">
                <Input />
              </Form.Item>
              <Form.Item
                label={text.birthLabel}
                name="birth"
                getValueProps={(value) => ({
                  value: value ? moment(value, "YYYY-MM-DD").startOf("day") : null,
                })}
                getValueFromEvent={(date) =>
                  date ? date.format("YYYY-MM-DD") : null
                }
              >
                <DatePicker
                  style={{ width: "100%" }}
                  disabledDate={disabledFutureDates}
                  onOpenChange={(open) => {
                    if (open) {
                      profileForm.setFieldsValue({ birth: null });
                    }
                  }}
                />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" style={{ marginRight: 8 }}>
                  {text.saveButton || "儲存"}
                </Button>
                <Button onClick={() => setEditing(false)}>
                  {text.cancelButton || "取消"}
                </Button>
              </Form.Item>
            </Form>
          ) : (
            <div>
              <p>
                {text.accountLabel}：{user?.username || text.notSet}
              </p>
              <p>
                {text.emailLabel}：{user?.email || text.notSet}
                {user?.email && (
                  <span style={{ marginLeft: 8 }}>
                    （{text.statusLabel}：{renderEmailStatus()}）
                  </span>
                )}
              </p>
              <p>
                {text.phoneLabel}：{user?.phone || text.notSet}
              </p>
              <p>
                {text.birthLabel}：
                {user?.birth
                  ? moment(user.birth).format("YYYY-MM-DD")
                  : text.notSet}
              </p>
              <Divider />
              <p>
                {text.lastLoginTimeLabel}
                <Text type="secondary">{lastLoginTime}</Text>
                <br />
                {text.lastLoginIpLabel}
                <Text type="secondary">{lastLoginIp}</Text>
              </p>
              <Button type="link" onClick={onEditProfile}>
                {text.editButton}
              </Button>
            </div>
          )}
        </Col>
      </Row>

      <Divider />

      <Row gutter={[16, 16]}>
        <Col>
          <Button type="primary" onClick={openChangePwdModal}>
            {text.changePasswordButton}
          </Button>
        </Col>
        <Col>
          <Button danger onClick={() => setDeleteModalVisible(true)}>
            {text.deleteAccountButton}
          </Button>
        </Col>
      </Row>

      <ChangePasswordModal
        visible={pwdModalVisible}
        onCancel={() => {
          setPwdModalVisible(false);
          pwdForm.resetFields();
        }}
        onSubmit={onChangePassword}
        form={pwdForm}
      />

      <DeleteAccountModal
        visible={deleteModalVisible}
        onCancel={() => setDeleteModalVisible(false)}
        onDelete={onDeleteAccount}
      />

      {/* 驗證信 Modal */}
      <Modal
        open={verificationModalVisible}
        title={text.verificationModalTitle}
        getContainer={() => document.body}
        onOk={handleVerificationOk}
        onCancel={() => setVerificationModalVisible(false)}
        okText={text.verificationModalOk}
        cancelText={text.verificationModalCancel}
        okButtonProps={{ disabled: /@example\.com$/i.test(user?.email) }}
      >
        {/@example\.com$/i.test(user?.email) ? (
          <p style={{ color: "red" }}>{text.verificationModalExampleEmail}</p>
        ) : (
          <>
            <p>{text.verificationModalDesc.replace("{email}", user?.email)}</p>
          </>
        )}
      </Modal>

      <Modal
        open={errorModalVisible}
        title={text.changePasswordFailureTitle}
        onCancel={() => setErrorModalVisible(false)}
        footer={[
          <Button key="ok" type="primary" onClick={() => setErrorModalVisible(false)}>
            {text.changePasswordModalOk}
          </Button>,
        ]}
      >
        <p>{errorMessage}</p>
      </Modal>
    </div>
  );
}

function ChangePasswordModal({ visible, onCancel, onSubmit, form }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = changePasswordLocale[currentLocale] || changePasswordLocale.en;

  const [passwordStrength, setPasswordStrength] = useState("");

  const onFinish = (values) => {
    onSubmit(values);
    setPasswordStrength("");
  };

  const handlePasswordChange = (e) => {
    const pwd = e.target.value;
    if (pwd.length < 6) {
      setPasswordStrength("弱");
    } else if (pwd.length < 10) {
      setPasswordStrength("中");
    } else {
      setPasswordStrength("強");
    }
  };

  return (
    <Modal
      open={visible}
      title={text.modalTitle}
      onCancel={() => {
        onCancel();
        form.resetFields();
        setPasswordStrength("");
      }}
      onOk={() => form.submit()}
      okText={text.okText}
      cancelText={text.cancelText}
    >
      <Form form={form} layout="vertical" onFinish={onFinish}>
        <Form.Item
          label={text.oldPasswordLabel}
          name="oldPassword"
          rules={[{ required: true, message: text.oldPasswordRequired }]}
        >
          <Input.Password />
        </Form.Item>
        <Form.Item
          label={
            <>
              {text.newPasswordLabel}
              {passwordStrength && (
                <span style={{ marginLeft: 8, color: "#999" }}>
                  (強度：{passwordStrength})
                </span>
              )}
            </>
          }
          name="newPassword"
          rules={[
            { required: true, message: text.newPasswordRequired },
            {
              validator: async (_, value) => {
                if (!value) {
                  return Promise.reject(new Error(text.newPasswordRequired));
                }
                if (value.length < 8) {
                  return Promise.reject(new Error(text.newPasswordTooShort));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <PasswordInput onChange={handlePasswordChange} />
        </Form.Item>
        <Form.Item
          label={text.confirmNewPasswordLabel}
          name="confirmPassword"
          dependencies={["newPassword"]}
          rules={[
            { required: true, message: text.confirmNewPasswordRequired },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue("newPassword") === value) {
                  return Promise.resolve();
                }
                return Promise.reject(text.passwordMismatch);
              },
            }),
          ]}
        >
          <Input.Password />
        </Form.Item>
      </Form>
    </Modal>
  );
}

function DeleteAccountModal({ visible, onCancel, onDelete }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = deleteAccountLocale[currentLocale] || deleteAccountLocale.en;

  return (
    <Modal
      open={visible}
      title={text.modalTitle}
      onCancel={onCancel}
      onOk={onDelete}
      okText={text.okText}
      cancelText={text.cancelText}
      okButtonProps={{ danger: true }}
    >
      <p>{text.modalContent}</p>
    </Modal>
  );
}
