// src/components/dashboard/DashboardMyInfo.jsx
import { UploadOutlined } from "@ant-design/icons";
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
import React, { useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";
import {
  deleteAccountApi,
  getUserInfo,
  resendVerificationEmailApi,
  updatePasswordApi,
  updateProfileApi,
  uploadAvatarApi,
} from "../../../utils/mockApi";

const { Text } = Typography;

export default function DashboardMyInfo() {
  const { token, user, setUser } = useAuth();
  const [infoLoading, setInfoLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [pwdModalVisible, setPwdModalVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [profileForm] = Form.useForm();

  const [showEmailAlert, setShowEmailAlert] = useState(false);

  // 額外顯示使用者的「最後登入時間」與「Email 是否驗證」等資料
  const lastLoginTime = user?.lastLoginTime
    ? moment(user.lastLoginTime).format("YYYY-MM-DD HH:mm")
    : "無資料";

  // 取得最新 user 資料
  const refreshUserInfo = async () => {
    if (!token) return;
    setInfoLoading(true);
    try {
      const data = await getUserInfo(token);
      setUser(data);

      // 依據回傳的 isEmailVerified 判斷是否要顯示警告
      if (data.isEmailVerified === false) {
        setShowEmailAlert(true);
      } else {
        setShowEmailAlert(false);
      }
    } catch (error) {
      message.error(error.message || "取得資料失敗");
    } finally {
      setInfoLoading(false);
    }
  };

  useEffect(() => {
    refreshUserInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 編輯個人資料
  const onEditProfile = () => {
    setEditing(true);
    if (user) {
      profileForm.setFieldsValue({
        nickname: user.nickname,
        email: user.email,
        phone: user.phone,
        birthday: user.birthday ? moment(user.birthday) : null,
      });
    }
  };

  const onSaveProfile = async (values) => {
    try {
      // 將 moment 格式的 birthday 轉為字串或後端需要的格式
      const birthdayString = values.birthday
        ? moment(values.birthday).format("YYYY-MM-DD")
        : null;

      await updateProfileApi(token, {
        ...values,
        birthday: birthdayString,
      });
      message.success("更新成功");
      setUser((prev) => ({
        ...prev,
        nickname: values.nickname,
        email: values.email,
        phone: values.phone,
        birthday: birthdayString,
      }));
      setEditing(false);
    } catch (err) {
      message.error(err.message || "更新失敗");
    }
  };

  // 上傳頭像
  const onUploadAvatar = async ({ file }) => {
    try {
      const newUrl = await uploadAvatarApi(token, file);
      message.success("頭像已更新");
      setUser((prev) => ({
        ...prev,
        avatar: newUrl,
      }));
    } catch (err) {
      message.error(err.message || "頭像上傳失敗");
    }
  };

  // 變更密碼
  const openChangePwdModal = () => {
    setPwdModalVisible(true);
  };
  const onChangePassword = async (values) => {
    if (values.newPassword !== values.confirmPassword) {
      return message.error("兩次輸入密碼不一致");
    }
    try {
      await updatePasswordApi(token, values.oldPassword, values.newPassword);
      message.success("密碼已變更！");
      setPwdModalVisible(false);
    } catch (err) {
      message.error(err.message || "變更密碼失敗");
    }
  };

  // 重新寄送驗證信
  const onResendVerification = async () => {
    try {
      await resendVerificationEmailApi(token);
      message.success("驗證信已重新寄送，請檢查您的信箱");
    } catch (err) {
      message.error(err.message || "寄送驗證信失敗");
    }
  };

  // 刪除帳號
  const onDeleteAccount = async () => {
    try {
      await deleteAccountApi(token);
      message.success("帳號已刪除，將導回主頁");
      // 這裡也可以登出並導回主頁
      // window.location.href = "/";
      setDeleteModalVisible(false);
    } catch (err) {
      message.error(err.message || "刪除帳號失敗");
    }
  };

  if (infoLoading) {
    return <Spin style={{ margin: "50px auto", display: "block" }} />;
  }

  return (
    <div>
      <h2>我的資訊</h2>

      {showEmailAlert && (
        <Alert
          style={{ marginBottom: 16 }}
          message="您的 Email 尚未驗證"
          description={
            <div>
              請至信箱收信並點擊驗證連結，
              <Button type="link" onClick={onResendVerification}>
                或點此重新寄送
              </Button>
            </div>
          }
          type="warning"
          showIcon
        />
      )}

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <div style={{ textAlign: "center" }}>
            <Avatar
              size={100}
              src={
                user?.avatar || "https://via.placeholder.com/100?text=No+Avatar"
              }
            />
            <br />
            <Upload
              showUploadList={false}
              accept="image/*"
              customRequest={onUploadAvatar}
            >
              <Button icon={<UploadOutlined />} style={{ marginTop: 8 }}>
                上傳頭像
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
                nickname: user?.nickname,
                email: user?.email,
                phone: user?.phone,
                birthday: user?.birthday
                  ? moment(user.birthday, "YYYY-MM-DD")
                  : null,
              }}
            >
              <Form.Item
                label="暱稱"
                name="nickname"
                rules={[{ required: true, message: "請輸入暱稱" }]}
              >
                <Input />
              </Form.Item>

              <Form.Item
                label="Email"
                name="email"
                rules={[{ required: true, type: "email" }]}
              >
                <Input />
              </Form.Item>

              <Form.Item
                label="電話"
                name="phone"
                rules={[{ pattern: /^09\d{8}$/, message: "請輸入正確手機格式" }]}
              >
                <Input placeholder="09xxxxxxxx" />
              </Form.Item>

              <Form.Item label="生日" name="birthday">
                <DatePicker
                  style={{ width: "100%" }}
                  disabledDate={(current) => current && current > moment()}
                  placeholder="選擇生日"
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
              <p>暱稱：{user?.nickname || "（未設定）"}</p>
              <p>Email：{user?.email || "未知"}</p>
              <p>電話：{user?.phone || "（未設定）"}</p>
              <p>
                生日：
                {user?.birthday
                  ? moment(user.birthday).format("YYYY-MM-DD")
                  : "（未設定）"}
              </p>
              <Divider />
              <p>
                上次登入時間：<Text type="secondary">{lastLoginTime}</Text>
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

      {/* 密碼變更 Modal */}
      <ChangePasswordModal
        visible={pwdModalVisible}
        onCancel={() => setPwdModalVisible(false)}
        onSubmit={onChangePassword}
      />

      {/* 刪除帳號 Modal */}
      <DeleteAccountModal
        visible={deleteModalVisible}
        onCancel={() => setDeleteModalVisible(false)}
        onDelete={onDeleteAccount}
      />
    </div>
  );
}

/** 變更密碼 Modal */
function ChangePasswordModal({ visible, onCancel, onSubmit }) {
  const [form] = Form.useForm();
  const [passwordStrength, setPasswordStrength] = useState("");

  const onFinish = (values) => {
    onSubmit(values);
    setPasswordStrength("");
    form.resetFields();
  };

  // 簡單示範一個密碼強度偵測
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
      title="變更密碼"
      open={visible}
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
            <div>
              新密碼
              {passwordStrength && (
                <span style={{ marginLeft: 8, color: "#999" }}>
                  (強度：{passwordStrength})
                </span>
              )}
            </div>
          }
          name="newPassword"
          rules={[
            { required: true, message: "請輸入新密碼" },
            { min: 8, message: "至少 8 碼" },
          ]}
        >
          <Input.Password onChange={handlePasswordChange} />
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

/** 刪除帳號 Modal */
function DeleteAccountModal({ visible, onCancel, onDelete }) {
  return (
    <Modal
      title="刪除帳號"
      open={visible}
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
