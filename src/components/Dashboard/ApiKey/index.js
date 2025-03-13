// src/components/Dashboard/ApiKey/index.js

import {
  CopyOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  InfoCircleOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Checkbox,
  Collapse,
  Drawer,
  Form,
  Input,
  InputNumber,
  List,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Tabs,
  Tooltip,
} from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";
import styles from "./index.module.css";

// API 基底路徑
const API_BASE_URL = "https://api.docsaid.org/public/token";

// 依據 usage_plan_id 回傳方案名稱
const getPlanName = (id) => {
  switch (id) {
    case 1:
      return "Basic (Free)";
    case 2:
      return "Pro (Paid)";
    case 3:
      return "PayAsYouGo";
    default:
      return "Unknown";
  }
};

// Token 列表中的單一項目元件
const TokenCard = ({ item, copyToken, handleRevoke, openDrawer, maskToken }) => {
  const plan = getPlanName(item.usage_plan_id);
  return (
    <List.Item className={styles.tokenListItem}>
      <Card
        className={styles.tokenCard}
        title={
          <div className={styles.tokenTitle}>
            <span className={styles.tokenName}>
              {item.name || "Untitled Key"}
            </span>
            <span className={styles.tokenPlan}>{plan}</span>
          </div>
        }
        extra={
          item.is_active && (
            <Popconfirm
              title="確定撤銷這個 Token 嗎？"
              icon={<ExclamationCircleOutlined style={{ color: "red" }} />}
              onConfirm={() => handleRevoke(item)}
            >
              <Button danger icon={<DeleteOutlined />}>
                Revoke
              </Button>
            </Popconfirm>
          )
        }
      >
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>Token：</span>
          <Tooltip title="點擊複製 Token">
            <a onClick={() => copyToken(item.jti)}>
              {maskToken(item.jti)}
              <CopyOutlined style={{ marginLeft: 6 }} />
            </a>
          </Tooltip>
        </div>
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>到期時間：</span>
          {item.expires_at || "永久"}
        </div>
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>狀態：</span>
          {item.is_active ? "Active" : "Revoked"}
        </div>

        {item.is_active && (
          <div style={{ marginTop: 12, textAlign: "right" }}>
            <Button onClick={() => openDrawer(item)}>詳細 / 用量</Button>
          </div>
        )}
      </Card>
    </List.Item>
  );
};

export default function DashboardApiKey() {
  const { token } = useAuth(); // 使用者登入 Token
  const [loading, setLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState([]);

  // 新增 Token 的 Modal
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [createForm] = Form.useForm();

  // Drawer 用於查看用量、詳細
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [drawerToken, setDrawerToken] = useState(null); // 目前開啟 Drawer 所顯示的 Token

  // 是否顯示「明碼」(馬賽克開關)
  const [showTokenPlain, setShowTokenPlain] = useState(false);

  // ===============================
  // 1) 載入 Token 列表
  // ===============================
  const fetchTokens = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/my-tokens`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        throw new Error(`Fetch tokens failed: ${res.status}`);
      }
      const data = await res.json();
      // data: [{ jti, usage_plan_id, expires_at, is_active }, ...]
      setApiKeys(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchTokens();
  }, [fetchTokens]);

  // ===============================
  // 2) 新增 Token
  // ===============================
  const handleOpenCreateModal = useCallback(() => {
    createForm.resetFields();
    setCreateModalVisible(true);
  }, [createForm]);

  const handleCreateToken = async (values) => {
    // 從表單取 usage_plan_id, isPermanent, expires_minutes, name
    const { usage_plan_id, isPermanent, expires_minutes, name } = values;
    // 若勾選永久 => expires_minutes = 999999
    const finalExpires = isPermanent ? 999999 : expires_minutes;

    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE_URL}/?usage_plan_id=${usage_plan_id}&expires_minutes=${finalExpires}`,
        {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Create token failed ${res.status}`);
      }
      const data = await res.json();
      // data: { access_token, expires_at, usage_plan_id, ... }

      // 用 Modal.success 提示一次性顯示完整 Token
      Modal.success({
        title: "Token 已建立！",
        content: (
          <div>
            <p>請妥善保存以下 Token，將不會再次顯示：</p>
            <div className={styles.tokenBox}>{data.access_token}</div>
          </div>
        ),
      });

      // 新建列表中的項目
      setApiKeys((prev) => [
        {
          jti: data.access_token,
          usage_plan_id: data.usage_plan_id,
          expires_at: data.expires_at,
          is_active: true,
          name: name || "", // 目前後端可能沒存 name
        },
        ...prev,
      ]);

      setCreateModalVisible(false);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===============================
  // 3) 撤銷 Token
  // ===============================
  const handleRevoke = useCallback(
    async (item) => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/revoke`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ jti: item.jti }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e.detail || `Revoke token failed: ${res.status}`);
        }
        message.success("Token 已撤銷");
        // 從列表移除
        setApiKeys((prev) => prev.filter((key) => key.jti !== item.jti));
        // 若 Drawer 正在看該 Token => 關閉
        if (drawerToken && drawerToken.jti === item.jti) {
          setDrawerVisible(false);
          setDrawerToken(null);
        }
      } catch (err) {
        message.error(err.message);
      } finally {
        setLoading(false);
      }
    },
    [token, drawerToken]
  );

  // ===============================
  // 4) 查看詳細(含用量)
  // ===============================
  const openDrawer = useCallback((item) => {
    setDrawerToken(item);
    setDrawerVisible(true);
  }, []);

  const closeDrawer = useCallback(() => {
    setDrawerVisible(false);
    setDrawerToken(null);
  }, []);

  // ===============================
  // 5) 查詢用量
  // ===============================
  const handleCheckUsage = async (publicToken) => {
    if (!publicToken) return;
    try {
      const res = await fetch(`${API_BASE_URL}/usage`, {
        headers: { Authorization: `Bearer ${publicToken}` },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Check usage error: ${res.status}`);
      }
      const usageData = await res.json();
      // usageData: { billing_type, used_this_hour, remaining, ... }
      if (usageData.billing_type === "rate_limit") {
        message.info(
          `使用次數：${usageData.used_this_hour}，剩餘：${usageData.remaining}`
        );
      } else {
        message.info(`已用：${usageData.used_this_hour}（Pay-Per-Use, 無限）`);
      }
    } catch (err) {
      message.error(err.message);
    }
  };

  // ===============================
  // 6) Token 顯示（馬賽克 + 複製）
  // ===============================
  const copyToken = useCallback(async (val) => {
    try {
      await navigator.clipboard.writeText(val);
      message.success("已複製 Token");
    } catch {
      message.error("複製失敗");
    }
  }, []);

  const maskToken = useCallback(
    (val) => {
      if (!val) return "";
      if (showTokenPlain) return val; // 顯示全文
      // 預設只顯示前6 & 後4
      const front = val.slice(0, 6);
      const back = val.slice(-4);
      return front + "****" + back;
    },
    [showTokenPlain]
  );

  // ===============================
  // Render
  // ===============================
  return (
    <div className={styles.apiKeyContainer}>
      <header className={styles.header}>
        <h2>My API Keys</h2>
        <p>在此管理、檢視、撤銷你的公開 Token</p>
      </header>

      <div className={styles.actions}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleOpenCreateModal}
          style={{ marginRight: 16 }}
        >
          建立新 Token
        </Button>
        <Button onClick={() => setShowTokenPlain((prev) => !prev)}>
          {showTokenPlain ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          {showTokenPlain ? "隱藏全部 Token" : "顯示全部 Token"}
        </Button>
      </div>

      <Modal
        title="建立新的公開 Token"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
        destroyOnClose
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={handleCreateToken}
          initialValues={{
            usage_plan_id: 1,
            expires_minutes: 60,
            isPermanent: false,
          }}
        >
          <Form.Item
            label="Token 名稱"
            name="name"
            tooltip="選填，給 Token 一個易識別的名稱"
          >
            <Input placeholder="e.g. MyDocAlignerKey" />
          </Form.Item>
          <Form.Item
            label="方案"
            name="usage_plan_id"
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value={1}>Basic (Free)</Select.Option>
              <Select.Option value={2}>Pro (Paid)</Select.Option>
              <Select.Option value={3}>PayAsYouGo</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            label="永久 (不過期)"
            name="isPermanent"
            valuePropName="checked"
          >
            <Checkbox>若勾選，忽略「有效期」</Checkbox>
          </Form.Item>
          <Form.Item
            label="有效期 (分鐘)"
            name="expires_minutes"
            rules={[
              { required: true, message: "請輸入有效期" },
              { type: "number", min: 10, max: 999999 },
            ]}
          >
            <InputNumber style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button onClick={() => setCreateModalVisible(false)}>取消</Button>
              <Button type="primary" htmlType="submit">
                建立
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <div style={{ marginTop: 24 }}>
        <Collapse
          bordered={false}
          className={styles.collapseRoot}
          defaultActiveKey={["tokens"]}
        >
          <Collapse.Panel
            key="tokens"
            header={
              <div style={{ display: "flex", alignItems: "center" }}>
                <InfoCircleOutlined style={{ marginRight: 8 }} />
                <span>我的 Token 列表</span>
              </div>
            }
          >
            <List
              loading={loading}
              dataSource={apiKeys}
              rowKey={(item) => item.jti}
              renderItem={(item) => (
                <TokenCard
                  item={item}
                  copyToken={copyToken}
                  handleRevoke={handleRevoke}
                  openDrawer={openDrawer}
                  maskToken={maskToken}
                />
              )}
            />
          </Collapse.Panel>
        </Collapse>
      </div>

      <Drawer
        title={
          drawerToken
            ? `${drawerToken.name || "My Token"} - 詳細`
            : "Token Info"
        }
        open={drawerVisible}
        onClose={closeDrawer}
        width={420}
      >
        {drawerToken && (
          <>
            <p>
              <strong>Token (遮罩): </strong>
              <Tooltip title="點擊複製 Token">
                <a onClick={() => copyToken(drawerToken.jti)}>
                  {maskToken(drawerToken.jti)}
                  <CopyOutlined style={{ marginLeft: 6 }} />
                </a>
              </Tooltip>
            </p>
            <p>
              <strong>到期時間:</strong>{" "}
              {drawerToken.expires_at || "永久"}
            </p>
            <p>
              <strong>狀態:</strong>{" "}
              {drawerToken.is_active ? "Active" : "Revoked"}
            </p>
            <Tabs defaultActiveKey="usage" style={{ marginTop: 16 }}>
              <Tabs.TabPane tab="用量資訊" key="usage">
                <p>
                  這裡可以顯示用量統計，或按下「查詢用量」按鈕。
                </p>
                <Button
                  icon={<EyeOutlined />}
                  onClick={() => handleCheckUsage(drawerToken.jti)}
                >
                  查詢用量
                </Button>
              </Tabs.TabPane>
              <Tabs.TabPane tab="其他" key="others">
                <p>可以放一些額外的說明 / log / 版本紀錄…</p>
              </Tabs.TabPane>
            </Tabs>
          </>
        )}
      </Drawer>
    </div>
  );
}
