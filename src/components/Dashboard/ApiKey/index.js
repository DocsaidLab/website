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
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
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
  Progress,
  Select,
  Space,
  Tabs,
  Tooltip,
} from "antd";
import React, { useEffect, useState } from "react";
import { useAuth } from "../../../context/AuthContext";
import styles from "./index.module.css";
import { apiKeyLocale } from "./locales";

const API_BASE_URL = "https://api.docsaid.org/public/token";

function getPlanName(id, text) {
  switch (id) {
    case 1:
      return text.planBasic;
    case 2:
      return text.planProfessional;
    case 3:
      return text.planPayAsYouGo;
    default:
      return text.planUnknown;
  }
}

/**
 * 單一 Token 卡片
 */
function TokenCard({
  item,
  copyToken,
  handleRevoke,
  handleDelete,
  openDrawer,
  maskToken,
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;
  const plan = getPlanName(item.usage_plan_id, text);

  const ActionButtons = () => {
    if (item.is_active) {
      return (
        <Popconfirm
          title={text.popconfirmRevokeTitle}
          icon={<ExclamationCircleOutlined style={{ color: "red" }} />}
          onConfirm={() => handleRevoke(item)}
        >
          <Button danger icon={<DeleteOutlined />}>
            {text.revokeButton}
          </Button>
        </Popconfirm>
      );
    } else {
      return (
        <Popconfirm
          title={text.popconfirmDeleteTitle}
          icon={<ExclamationCircleOutlined style={{ color: "red" }} />}
          onConfirm={() => handleDelete(item)}
        >
          <Button danger icon={<DeleteOutlined />}>
            {text.deleteButton}
          </Button>
        </Popconfirm>
      );
    }
  };

  return (
    <List.Item className={styles.tokenListItem}>
      <Card
        className={styles.tokenCard}
        title={
          <div className={styles.tokenTitle}>
            <span className={styles.tokenName}>
              {item.name || text.defaultTokenName}
            </span>
            <span className={styles.tokenPlan}>{plan}</span>
          </div>
        }
        extra={<Space>{<ActionButtons />}</Space>}
      >
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>{text.tokenIdLabel}</span>
          <Tooltip title={text.tooltipCopyToken}>
            <a onClick={() => copyToken(item.jti)}>
              {maskToken(item.jti)}
              <CopyOutlined style={{ marginLeft: 6 }} />
            </a>
          </Tooltip>
        </div>
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>{text.expiryLabel}</span>
          {item.expires_at || text.forever}
        </div>
        <div className={styles.tokenItemRow}>
          <span className={styles.label}>{text.statusLabel}</span>
          {item.is_active ? text.active : text.revoked}
        </div>
        {item.is_active && (
          <div style={{ marginTop: 12, textAlign: "right" }}>
            <Button onClick={() => openDrawer(item)}>
              {text.detailUsageButton}
            </Button>
          </div>
        )}
      </Card>
    </List.Item>
  );
}

export default function DashboardApiKey() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;
  const { token: userToken } = useAuth();
  const [loading, setLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState([]);

  // 新增 Token 的 Modal
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [createForm] = Form.useForm();

  // Drawer 用於查看用量與詳細
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [drawerToken, setDrawerToken] = useState(null);

  // 用來存查詢回來的用量資訊
  const [drawerUsage, setDrawerUsage] = useState(null);

  // 是否顯示明碼 (馬賽克開關)
  const [showTokenPlain, setShowTokenPlain] = useState(false);

  // 載入 Token 列表
  const fetchTokens = React.useCallback(async () => {
    if (!userToken) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/my-tokens`, {
        headers: { Authorization: `Bearer ${userToken}` },
      });
      if (!res.ok) {
        throw new Error(`Fetch tokens failed: ${res.status}`);
      }
      const data = await res.json();
      setApiKeys(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [userToken]);

  useEffect(() => {
    fetchTokens();
  }, [fetchTokens]);

  // 打開 "新建 Token" modal
  const handleOpenCreateModal = React.useCallback(() => {
    createForm.resetFields();
    setCreateModalVisible(true);
  }, [createForm]);

  // 建立 Token
  const handleCreateToken = async (values) => {
    const { usage_plan_id, isPermanent, expires_minutes, name } = values;
    const finalExpires = isPermanent ? 999999 : expires_minutes;

    if (!userToken) {
      message.error(text.notLoggedIn);
      return;
    }

    setLoading(true);
    try {
      const nameParam = name ? `&name=${encodeURIComponent(name)}` : "";
      const url = `${API_BASE_URL}/?usage_plan_id=${usage_plan_id}&expires_minutes=${finalExpires}${nameParam}`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${userToken}`,
        },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Create token failed: ${res.status}`);
      }
      const data = await res.json();

      Modal.success({
        title: text.createTokenSuccessTitle,
        content: (
          <>
            <p>{text.createTokenSuccessContent}</p>
            <div className={styles.tokenBox}>{data.access_token}</div>
          </>
        ),
      });

      // 解析 jti
      const jti = parseJtiFromJWT(data.access_token);
      if (!jti) {
        message.warning(text.parseJtiWarning);
      }

      const newTokenItem = {
        jti: jti || `temp-${Date.now()}`,
        usage_plan_id: data.usage_plan_id,
        expires_at: data.expires_at,
        is_active: true,
        name: name || "",
        rawToken: data.access_token,
      };

      setApiKeys((prev) => [newTokenItem, ...prev]);
      setCreateModalVisible(false);
    } catch (err) {
      message.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  function parseJtiFromJWT(jwtStr) {
    try {
      const parts = jwtStr.split(".");
      if (parts.length !== 3) return null;
      const payloadRaw = atob(parts[1]);
      const payload = JSON.parse(payloadRaw);
      return payload.jti;
    } catch {
      return null;
    }
  }

  // 撤銷 Token
  const handleRevoke = React.useCallback(
    async (item) => {
      if (!userToken) return;
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/revoke`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${userToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ jti: item.jti }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e.detail || `Revoke token failed: ${res.status}`);
        }
        message.success(text.tokenRevoked);
        await fetchTokens();
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
    [userToken, drawerToken, fetchTokens, text]
  );

  // 刪除 Token
  const handleDelete = React.useCallback(
    async (item) => {
      if (!userToken) return;
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/remove`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${userToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ jti: item.jti }),
        });
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          throw new Error(e.detail || `Delete token failed: ${res.status}`);
        }
        message.success(text.tokenDeleted);
        await fetchTokens();
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
    [userToken, drawerToken, fetchTokens, text]
  );

  // 打開 Drawer
  const openDrawer = React.useCallback((item) => {
    setDrawerUsage(null);
    setDrawerToken(item);
    setDrawerVisible(true);
  }, []);

  // 關閉 Drawer
  const closeDrawer = React.useCallback(() => {
    setDrawerVisible(false);
    setDrawerToken(null);
    setDrawerUsage(null);
  }, []);

  // 查詢用量
  const handleCheckUsage = async (rawToken) => {
    if (!rawToken) {
      message.error(text.missingFullToken);
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/usage`, {
        headers: {
          Authorization: `Bearer ${rawToken}`,
        },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Check usage error: ${res.status}`);
      }
      const usageData = await res.json();
      setDrawerUsage(usageData);
    } catch (err) {
      message.error(err.message);
    }
  };

  // 複製 Token
  const copyToken = React.useCallback(async (val) => {
    try {
      await navigator.clipboard.writeText(val);
      message.success(text.copySuccess);
    } catch {
      message.error(text.copyFailure);
    }
  }, [text]);

  // 遮罩 Token
  const maskToken = React.useCallback(
    (val) => {
      if (!val) return "";
      if (showTokenPlain) return val;
      const front = val.slice(0, 6);
      const back = val.slice(-4);
      return front + "****" + back;
    },
    [showTokenPlain]
  );

  return (
    <div className={styles.apiKeyContainer}>
      <header className={styles.header}>
        <h2>{text.headerTitle}</h2>
        <p>{text.headerDescription}</p>
      </header>

      <div className={styles.actions}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleOpenCreateModal}
          style={{ marginRight: 16 }}
        >
          {text.createTokenButton}
        </Button>
        <Button onClick={() => setShowTokenPlain((prev) => !prev)}>
          {showTokenPlain ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          {showTokenPlain ? text.toggleHideTokens : text.toggleShowTokens}
        </Button>
      </div>

      <Modal
        title={text.createModalTitle}
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
              <Select.Option value={2} disabled>
                {text.planProfessional}
              </Select.Option>
              <Select.Option value={3} disabled>
                {text.planPayAsYouGo}
              </Select.Option>
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
              <Button onClick={() => setCreateModalVisible(false)}>
                {text.cancelButton}
              </Button>
              <Button type="primary" htmlType="submit">
                {text.createButton}
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
                <span>{text.collapseHeader}</span>
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
                  handleDelete={handleDelete}
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
            ? `${drawerToken.name || text.defaultTokenName} - ${text.drawerDetail}`
            : text.drawerInfo
        }
        open={drawerVisible}
        onClose={closeDrawer}
        width={420}
      >
        {drawerToken && (
          <>
            <p>
              <strong>{text.drawerTokenIdLabel}</strong>
              <Tooltip title={text.tooltipCopyToken}>
                <a onClick={() => copyToken(drawerToken.jti)}>
                  {maskToken(drawerToken.jti)}
                  <CopyOutlined style={{ marginLeft: 6 }} />
                </a>
              </Tooltip>
            </p>
            <p>
              <strong>{text.drawerExpiryLabel}</strong>{" "}
              {drawerToken.expires_at || text.forever}
            </p>
            <p>
              <strong>{text.drawerStatusLabel}</strong>{" "}
              {drawerToken.is_active ? text.active : text.revoked}
            </p>

            <Tabs defaultActiveKey="usage" style={{ marginTop: 16 }}>
              <Tabs.TabPane tab={text.tabUsageInfo} key="usage">
                <p>{text.usageInfoInstruction}</p>
                <Button
                  icon={<EyeOutlined />}
                  onClick={() => handleCheckUsage(drawerToken.rawToken)}
                  disabled={!drawerToken.is_active}
                >
                  {text.checkUsageButton}
                </Button>
                <p style={{ marginTop: 8, fontSize: 12, color: "#999" }}>
                  {text.usageNote}
                </p>

                {drawerUsage && (
                  <div style={{ marginTop: 16 }}>
                    {drawerUsage.billing_type === "rate_limit" ? (
                      <>
                        <p>{`${text.usageThisHourLabel}${drawerUsage.used_this_hour} / ${drawerUsage.limit_per_hour}`}</p>
                        <p>{`${text.remainingLabel}${drawerUsage.remaining}`}</p>
                        <Progress
                          percent={Math.min(
                            100,
                            (drawerUsage.used_this_hour /
                              drawerUsage.limit_per_hour) *
                              100
                          )}
                          status={
                            drawerUsage.used_this_hour >=
                            drawerUsage.limit_per_hour
                              ? "exception"
                              : "active"
                          }
                        />
                      </>
                    ) : (
                      <>
                        <p>{`${text.usageThisHourLabel}${drawerUsage.used_this_hour} (${text.payPerUseInfo})`}</p>
                        <p>{drawerUsage.note || ""}</p>
                      </>
                    )}
                  </div>
                )}
              </Tabs.TabPane>

              <Tabs.TabPane tab={text.tabOthers} key="others">
                <p>{text.othersContent}</p>
              </Tabs.TabPane>
            </Tabs>
          </>
        )}
      </Drawer>
    </div>
  );
}
