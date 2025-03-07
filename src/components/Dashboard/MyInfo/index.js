// /src/components/Dashboard/MyInfo/index.js
import { UploadOutlined, UserOutlined } from "@ant-design/icons";
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

export default function DashboardMyInfo() {
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
    : "無資料";
  const lastLoginIp = user?.last_login_ip || "無資料";

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
        throw new Error("取得使用者資訊失敗");
      }
      const data = await res.json();
      setUser(data);
      // 若 email 存在但未驗證，顯示警示
      setShowEmailAlert(data.email && data.is_email_verified === false);
    } catch (err) {
      message.error(err.message || "取得使用者資訊失敗");
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
      message.success("個人資料更新成功");
      setUser(updatedUser);
      setShowEmailAlert(
        updatedUser.email && updatedUser.is_email_verified === false
      );
      setEditing(false);
    } catch (err) {
      if (err.message.includes("Email already exists")) {
        profileForm.setFields([{ name: "email", errors: [err.message] }]);
      }
      message.error(err.message || "個人資料更新失敗");
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
        throw new Error(errData.detail || "頭像上傳失敗");
      }
      const result = await res.json();
      message.success("頭像已更新");
      setUser((prev) => ({
        ...prev,
        avatar: result.avatar,
      }));
    } catch (err) {
      message.error(err.message || "頭像上傳失敗");
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
      setErrorMessage("兩次輸入的密碼不一致");
      setErrorModalVisible(true);
      return;
    }
    try {
      await changePassword(values.oldPassword, values.newPassword);
      message.success("密碼變更成功！");
      setPwdModalVisible(false);
      pwdForm.resetFields();
    } catch (err) {
      console.error("變更密碼錯誤:", err);
      const errorMsg = err.response?.data?.detail || err.message || "密碼變更失敗";
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
      message.success("驗證信已寄出，請檢查信箱");
      setVerificationModalVisible(false);
    } catch (err) {
      message.error(err.message || "寄送驗證信失敗");
    }
  };

  // =========== 7. 刪除帳號 ===========
  const onDeleteAccount = async () => {
    try {
      await deleteAccount();
      message.success("帳號已刪除，將導回主頁");
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
      return <Text type="secondary">尚未設定</Text>;
    }
    if (user.is_email_verified) {
      return <Text style={{ color: "green" }}>已驗證</Text>;
    }
    return (
      <span style={{ color: "red" }}>
        未驗證
        <Button type="link" onClick={onResendVerification}>
          寄送驗證信
        </Button>
      </span>
    );
  };

  return (
    <div>
      <h2>我的資訊</h2>

      {showEmailAlert && user?.email && !user.is_email_verified && (
        <Alert
          style={{ marginBottom: 16 }}
          message="您的 Email 尚未驗證"
          description="請點選下方重新寄送驗證信，並檢查您的信箱。"
          type="warning"
          showIcon
        />
      )}

      {!user?.email && !editing && (
        <Alert
          style={{ marginBottom: 16 }}
          message="尚未填寫 Email"
          description={
            <>
              您尚未綁定 Email，請點
              <Button type="link" onClick={onEditProfile}>
                編輯
              </Button>
              補上 Email。
            </>
          }
          type="info"
          showIcon
        />
      )}

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <div style={{ textAlign: "center" }}>
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
                {uploadLoading ? "上傳中..." : "上傳頭像"}
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
                label="帳號"
                name="username"
                rules={[{ required: true, message: "請輸入帳號" }]}
              >
                <Input disabled />
              </Form.Item>
              <Form.Item
                label="Email"
                name="email"
                rules={[
                  { required: true, message: "請輸入 Email" },
                  { type: "email", message: "Email 格式錯誤" },
                ]}
              >
                {user?.is_email_verified === false ? <Input /> : <Input disabled />}
              </Form.Item>
              <Form.Item label="電話" name="phone">
                <Input />
              </Form.Item>
              <Form.Item
                label="生日"
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
                  儲存
                </Button>
                <Button onClick={() => setEditing(false)}>取消</Button>
              </Form.Item>
            </Form>
          ) : (
            <div>
              <p>帳號：{user?.username || "（未設定）"}</p>
              <p>
                Email：{user?.email || "（未設定）"}
                {user?.email && (
                  <span style={{ marginLeft: 8 }}>
                    （狀態：{renderEmailStatus()}）
                  </span>
                )}
              </p>
              <p>電話：{user?.phone || "（未設定）"}</p>
              <p>
                生日：
                {user?.birth
                  ? moment(user.birth).format("YYYY-MM-DD")
                  : "（未設定）"}
              </p>
              <Divider />
              <p>
                上次登入時間：
                <Text type="secondary">{lastLoginTime}</Text>
                <br />
                上次登入 IP：<Text type="secondary">{lastLoginIp}</Text>
              </p>
              <Button type="link" onClick={onEditProfile}>
                編輯
              </Button>
            </div>
          )}
        </Col>
      </Row>

      <Divider />

      <Row gutter={[16, 16]}>
        <Col>
          <Button type="primary" onClick={openChangePwdModal}>
            變更密碼
          </Button>
        </Col>
        <Col>
          <Button danger onClick={() => setDeleteModalVisible(true)}>
            刪除帳號
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

      {/* 狀態控制的驗證信 Modal */}
      <Modal
        open={verificationModalVisible}
        title="寄送驗證信"
        getContainer={() => document.body}
        onOk={handleVerificationOk}
        onCancel={() => setVerificationModalVisible(false)}
        okText="確定"
        cancelText="取消"
        okButtonProps={{ disabled: /@example\.com$/i.test(user?.email) }}
      >
        {/@example\.com$/i.test(user?.email) ? (
          <p style={{ color: "red" }}>
            預設信箱 (example.com) 無法用於驗證，請更換為有效的 Email。
          </p>
        ) : (
          <>
            <p>
              系統將發送驗證信至：
              <strong style={{ marginLeft: 5 }}>{user?.email}</strong>
            </p>
            <p>請確認此 Email 是否正確？</p>
          </>
        )}
      </Modal>

      <Modal
        open={errorModalVisible}
        title="變更密碼失敗"
        onCancel={() => setErrorModalVisible(false)}
        footer={[
          <Button key="ok" type="primary" onClick={() => setErrorModalVisible(false)}>
            確定
          </Button>
        ]}
      >
        <p>{errorMessage}</p>
      </Modal>

    </div>
  );
}

function ChangePasswordModal({ visible, onCancel, onSubmit, form }) {
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
      open={visible}  // 改用 open 屬性
      title="變更密碼"
      onCancel={() => {
        onCancel();
        form.resetFields();
        setPasswordStrength("");
      }}
      onOk={() => form.submit()}
      okText="儲存"
      cancelText="取消"
    >
      <Form form={form} layout="vertical" onFinish={onFinish}>
        <Form.Item
          label="舊密碼"
          name="oldPassword"
          rules={[{ required: true, message: "請輸入舊密碼" }]}
        >
          <Input.Password />
        </Form.Item>
        <Form.Item
          label={
            <>
              新密碼
              {passwordStrength && (
                <span style={{ marginLeft: 8, color: "#999" }}>
                  (強度：{passwordStrength})
                </span>
              )}
            </>
          }
          name="newPassword"
          rules={[
            { required: true, message: "請輸入新密碼" },
            {
              validator: async (_, value) => {
                if (!value) {
                  return Promise.reject(new Error("請輸入新密碼"));
                }
                if (value.length < 8) {
                  return Promise.reject(new Error("至少 8 碼"));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <PasswordInput onChange={handlePasswordChange} />
        </Form.Item>
        <Form.Item
          label="確認新密碼"
          name="confirmPassword"
          dependencies={["newPassword"]}
          rules={[
            { required: true, message: "請再次輸入新密碼" },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue("newPassword") === value) {
                  return Promise.resolve();
                }
                return Promise.reject("兩次輸入的密碼不一致");
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
  return (
    <Modal
      open={visible}  // 改用 open 屬性
      title="刪除帳號"
      onCancel={onCancel}
      onOk={onDelete}
      okText="確定刪除"
      cancelText="取消"
      okButtonProps={{ danger: true }}
    >
      <p>您確定要刪除帳號嗎？此操作無法復原！</p>
    </Modal>
  );
}
