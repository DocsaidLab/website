// /src/components/Dashboard/MyInfo/index.js
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

const { Text } = Typography;

const localeText = {
  "zh-hant": {
    myInfoTitle: "我的資訊",
    emailNotVerifiedAlertTitle: "您的 Email 尚未驗證",
    emailNotVerifiedAlertDesc: "沒有驗證信箱，帳號密碼丟失後無法找回。請點選下方寄送驗證信，並檢查您的信箱。",
    noEmailAlertTitle: "尚未填寫 Email",
    noEmailAlertDesc: "您尚未綁定 Email，請點 {editLink} 補上 Email。",
    uploadAvatarButtonUploading: "上傳中...",
    uploadAvatarButton: "上傳頭像",
    accountLabel: "帳號",
    emailLabel: "Email",
    phoneLabel: "電話",
    birthLabel: "生日",
    notSet: "（未設定）",
    lastLoginTimeLabel: "上次登入時間：",
    lastLoginIpLabel: "上次登入 IP：",
    statusLabel: "狀態",
    editButton: "編輯",
    changePasswordButton: "變更密碼",
    deleteAccountButton: "刪除帳號",
    successMsg: "個人資料更新成功",
    fetchUserInfoFailure: "取得使用者資訊失敗",
    avatarUploadFailure: "頭像上傳失敗",
    avatarUploadSuccess: "頭像已更新",
    verified: "已驗證",
    notVerified: "未驗證",
    resendVerification: "寄送驗證信",
    changePasswordSuccess: "密碼變更成功！",
    passwordMismatch: "兩次輸入的密碼不一致",
    changePasswordFailureTitle: "變更密碼失敗",
    verificationModalTitle: "寄送驗證信",
    verificationModalOk: "確定",
    verificationModalCancel: "取消",
    verificationModalExampleEmail: "預設信箱 (example.com) 無法用於驗證，請更換為有效的 Email。",
    verificationModalDesc: "系統將發送驗證信至：{email}\n請確認此 Email 是否正確？",
    deleteAccountSuccess: "帳號已刪除，將導回首頁",
    saveButton: "儲存",
    cancelButton: "取消",
    changePasswordModalOk: "確定",
  },
  en: {
    myInfoTitle: "My Information",
    emailNotVerifiedAlertTitle: "Your Email is not verified",
    emailNotVerifiedAlertDesc: "Please click the button below to resend the verification email and check your inbox.",
    noEmailAlertTitle: "Email not set",
    noEmailAlertDesc: "You have not bound an Email. Please click {editLink} to add an Email.",
    uploadAvatarButtonUploading: "Uploading...",
    uploadAvatarButton: "Upload Avatar",
    accountLabel: "Account",
    emailLabel: "Email",
    phoneLabel: "Phone",
    birthLabel: "Birth",
    notSet: "(Not set)",
    lastLoginTimeLabel: "Last Login Time: ",
    lastLoginIpLabel: "Last Login IP: ",
    statusLabel: "Status",
    editButton: "Edit",
    changePasswordButton: "Change Password",
    deleteAccountButton: "Delete Account",
    successMsg: "Profile updated successfully",
    fetchUserInfoFailure: "Failed to fetch user information",
    avatarUploadFailure: "Avatar upload failed",
    avatarUploadSuccess: "Avatar updated",
    verified: "Verified",
    notVerified: "Not verified",
    resendVerification: "Resend Verification",
    changePasswordSuccess: "Password changed successfully!",
    passwordMismatch: "Passwords do not match",
    changePasswordFailureTitle: "Change Password Failed",
    verificationModalTitle: "Resend Verification Email",
    verificationModalOk: "Confirm",
    verificationModalCancel: "Cancel",
    verificationModalExampleEmail: "Default email (example.com) cannot be used for verification. Please change to a valid Email.",
    verificationModalDesc: "The system will send a verification email to: {email}\nPlease confirm if this Email is correct.",
    deleteAccountSuccess: "Account deleted, redirecting to homepage",
    saveButton: "Save",
    cancelButton: "Cancel",
    changePasswordModalOk: "Confirm",
  },
  ja: {
    myInfoTitle: "私の情報",
    emailNotVerifiedAlertTitle: "メールが未認証です",
    emailNotVerifiedAlertDesc: "下のボタンをクリックして認証メールを再送信し、受信箱を確認してください。",
    noEmailAlertTitle: "メール未設定",
    noEmailAlertDesc: "メールが登録されていません。{editLink} をクリックしてメールを追加してください。",
    uploadAvatarButtonUploading: "アップロード中...",
    uploadAvatarButton: "アバターをアップロード",
    accountLabel: "アカウント",
    emailLabel: "メール",
    phoneLabel: "電話",
    birthLabel: "生年月日",
    notSet: "(未設定)",
    lastLoginTimeLabel: "最終ログイン時間：",
    lastLoginIpLabel: "最終ログイン IP：",
    statusLabel: "状態",
    editButton: "編集",
    changePasswordButton: "パスワード変更",
    deleteAccountButton: "アカウント削除",
    successMsg: "プロフィール更新成功",
    fetchUserInfoFailure: "ユーザー情報の取得に失敗しました",
    avatarUploadFailure: "アバターのアップロードに失敗しました",
    avatarUploadSuccess: "アバターが更新されました",
    verified: "認証済み",
    notVerified: "未認証",
    resendVerification: "認証メール再送信",
    changePasswordSuccess: "パスワード変更成功！",
    passwordMismatch: "入力したパスワードが一致しません",
    changePasswordFailureTitle: "パスワード変更失敗",
    verificationModalTitle: "認証メール再送信",
    verificationModalOk: "確定",
    verificationModalCancel: "キャンセル",
    verificationModalExampleEmail: "デフォルトのメール (example.com) は認証に使用できません。有効なメールに変更してください。",
    verificationModalDesc: "システムは以下のメールに認証メールを送信します：{email}\nこのメールが正しいか確認してください。",
    deleteAccountSuccess: "アカウントが削除されました。ホームページにリダイレクトします。",
    saveButton: "保存",
    cancelButton: "キャンセル",
    changePasswordModalOk: "確定",
  },
};

export default function DashboardMyInfo() {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;

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
          description={
            <>
              {text.noEmailAlertDesc.replace(
                "{editLink}",
                <Button type="link" onClick={onEditProfile} key="edit">
                  {text.editButton}
                </Button>
              )}
            </>
          }
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
          <Upload showUploadList={false} accept="image/*" customRequest={onUploadAvatar}>
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
                  { required: true, message: text.emailRequired || "請輸入 Email" },
                  { type: "email", message: text.invalidEmail || "Email 格式錯誤" },
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
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const localeText = {
    "zh-hant": {
      modalTitle: "變更密碼",
      oldPasswordLabel: "舊密碼",
      oldPasswordRequired: "請輸入舊密碼",
      newPasswordLabel: "新密碼",
      newPasswordRequired: "請輸入新密碼",
      newPasswordTooShort: "至少 8 碼",
      confirmNewPasswordLabel: "確認新密碼",
      confirmNewPasswordRequired: "請再次輸入新密碼",
      passwordMismatch: "兩次輸入的密碼不一致",
      okText: "儲存",
      cancelText: "取消",
    },
    en: {
      modalTitle: "Change Password",
      oldPasswordLabel: "Old Password",
      oldPasswordRequired: "Please enter your old password",
      newPasswordLabel: "New Password",
      newPasswordRequired: "Please enter your new password",
      newPasswordTooShort: "At least 8 characters",
      confirmNewPasswordLabel: "Confirm New Password",
      confirmNewPasswordRequired: "Please re-enter your new password",
      passwordMismatch: "Passwords do not match",
      okText: "Save",
      cancelText: "Cancel",
    },
    ja: {
      modalTitle: "パスワード変更",
      oldPasswordLabel: "旧パスワード",
      oldPasswordRequired: "旧パスワードを入力してください",
      newPasswordLabel: "新パスワード",
      newPasswordRequired: "新パスワードを入力してください",
      newPasswordTooShort: "8文字以上である必要があります",
      confirmNewPasswordLabel: "新パスワード確認",
      confirmNewPasswordRequired: "新パスワードを再入力してください",
      passwordMismatch: "入力したパスワードが一致しません",
      okText: "保存",
      cancelText: "キャンセル",
    },
  };
  const text = localeText[currentLocale] || localeText.en;

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
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const localeText = {
    "zh-hant": {
      modalTitle: "刪除帳號",
      modalContent: "您確定要刪除帳號嗎？此操作無法復原！",
      okText: "確定刪除",
      cancelText: "取消",
    },
    en: {
      modalTitle: "Delete Account",
      modalContent:
        "Are you sure you want to delete your account? This action cannot be undone!",
      okText: "Confirm Delete",
      cancelText: "Cancel",
    },
    ja: {
      modalTitle: "アカウント削除",
      modalContent: "本当にアカウントを削除しますか？この操作は元に戻せません！",
      okText: "削除を確定",
      cancelText: "キャンセル",
    },
  };
  const text = localeText[currentLocale] || localeText.en;

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
