// 檔案： src/components/dashboard/DashboardApiKey/index.js

import { EyeInvisibleOutlined, EyeOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Drawer,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Progress,
  Row,
  Space,
  Switch,
  Table,
  Tag,
} from "antd";
import moment from "moment";
import React, { useEffect, useState } from "react";
import {
  createApiKeyApi,
  deleteApiKeyApi,
  getApiKeyUsageApi,
  getMyApiKeysApi,
  regenerateApiKeyApi,
  updateApiKeyNameApi,
} from "../../../utils/mockApi";

export default function DashboardApiKey() {
  const [loading, setLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState([]);
  const [createModalVisible, setCreateModalVisible] = useState(false);

  // Drawer 狀態：顯示單一 API Key 詳細資訊
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailData, setDetailData] = useState(null);

  // 「顯示 / 隱藏」 Key 內容
  const [showKeys, setShowKeys] = useState({});
  // 例如 { keyId: true/false }

  // 建立新的 Key 表單
  const [form] = Form.useForm();

  useEffect(() => {
    fetchApiKeys();
  }, []);

  const fetchApiKeys = async () => {
    setLoading(true);
    try {
      const data = await getMyApiKeysApi();
      setApiKeys(data);
    } catch (err) {
      message.error(err.message || "取得 API Key 失敗");
    } finally {
      setLoading(false);
    }
  };

  // Modal：建立新的 API Key
  const handleCreateKey = async (values) => {
    try {
      const newKeyObj = await createApiKeyApi(values.name);
      message.success("已成功建立新 API Key");
      setApiKeys((prev) => [...prev, newKeyObj]);
      setCreateModalVisible(false);
      form.resetFields();
    } catch (err) {
      message.error(err.message || "建立失敗");
    }
  };

  // 「刪除」某把 Key
  const handleDeleteKey = async (record) => {
    try {
      await deleteApiKeyApi(record.id);
      message.success(`已刪除 API Key: ${record.name}`);
      setApiKeys((prev) => prev.filter((k) => k.id !== record.id));
    } catch (err) {
      message.error(err.message || "刪除失敗");
    }
  };

  // 「重新生成」某把 Key
  const handleRegenerateKey = async (record) => {
    try {
      const newKeyStr = await regenerateApiKeyApi(record.id);
      message.success("已重新生成 API Key");
      // 更新該 Key 的 keyString
      setApiKeys((prev) =>
        prev.map((item) =>
          item.id === record.id ? { ...item, keyString: newKeyStr } : item
        )
      );
    } catch (err) {
      message.error(err.message || "重新生成失敗");
    }
  };

  // 「重新命名」Key
  const handleRenameKey = async (record, newName) => {
    try {
      await updateApiKeyNameApi(record.id, newName);
      message.success("名稱已更新");
      setApiKeys((prev) =>
        prev.map((item) =>
          item.id === record.id ? { ...item, name: newName } : item
        )
      );
    } catch (err) {
      message.error(err.message || "更新失敗");
    }
  };

  // 打開 Drawer，顯示更詳細資訊 (例如：用量, IP白名單, 失效日期, etc.)
  const openDetailDrawer = async (record) => {
    setDetailDrawerVisible(true);
    setDetailLoading(true);
    try {
      // 可能要呼叫後端取得更完整資訊 (例如 getApiKeyUsageApi)
      const usageData = await getApiKeyUsageApi(record.id);
      setDetailData({
        ...record,
        usage: usageData.usage,
        limit: usageData.limit,
        whitelist: usageData.ipWhitelist || [],
      });
    } catch (err) {
      message.error("取得詳細資料失敗：" + err.message);
    } finally {
      setDetailLoading(false);
    }
  };

  // 顯示/隱藏某把 Key
  const toggleShowKey = (record) => {
    setShowKeys((prev) => ({
      ...prev,
      [record.id]: !prev[record.id],
    }));
  };

  // Table欄位定義
  const columns = [
    {
      title: "名稱",
      dataIndex: "name",
      render: (text, record) => (
        <EditableText
          text={text}
          onSave={(newVal) => handleRenameKey(record, newVal)}
        />
      ),
    },
    {
      title: "API Key",
      dataIndex: "keyString",
      render: (text, record) => {
        // 顯示 / 隱藏 Key
        const showing = showKeys[record.id];
        // 若不顯示則遮蔽中間段
        const masked = text
          ? showing
            ? text
            : maskKeyString(text)
          : "N/A";
        return (
          <Space>
            <span style={{ fontFamily: "monospace" }}>{masked}</span>
            <Button
              icon={showing ? <EyeInvisibleOutlined /> : <EyeOutlined />}
              size="small"
              onClick={() => toggleShowKey(record)}
            />
          </Space>
        );
      },
    },
    {
      title: "建立日期",
      dataIndex: "createdAt",
      render: (val) => moment(val).format("YYYY-MM-DD HH:mm"),
      width: 150,
    },
    {
      title: "到期日",
      dataIndex: "expireAt",
      render: (val) =>
        val ? moment(val).format("YYYY-MM-DD") : <Tag color="green">無到期</Tag>,
      width: 120,
    },
    {
      title: "狀態",
      dataIndex: "status",
      render: (val) => {
        // val = "active" | "expired" | "disabled"
        if (val === "active") return <Tag color="blue">啟用中</Tag>;
        if (val === "expired") return <Tag color="red">已過期</Tag>;
        return <Tag color="default">已停用</Tag>;
      },
      width: 100,
    },
    {
      title: "操作",
      width: 180,
      render: (text, record) => (
        <Space>
          <Button size="small" onClick={() => openDetailDrawer(record)}>
            詳細
          </Button>

          <Button size="small" onClick={() => handleRegenerateKey(record)}>
            重生
          </Button>

          <Popconfirm
            title={`確定刪除「${record.name}」？`}
            onConfirm={() => handleDeleteKey(record)}
          >
            <Button danger size="small">
              刪除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Card title="API Key 管理">
      <Row justify="space-between" style={{ marginBottom: 16 }}>
        <Col>
          <Button type="primary" onClick={() => setCreateModalVisible(true)}>
            建立新 API Key
          </Button>
        </Col>
        <Col>
          <Button onClick={fetchApiKeys} loading={loading}>
            重新整理
          </Button>
        </Col>
      </Row>

      <Table
        rowKey="id"
        dataSource={apiKeys}
        columns={columns}
        loading={loading}
        pagination={{ pageSize: 5 }}
        scroll={{ x: "max-content" }}
      />

      {/* 建立新 API Key 的 Modal */}
      <Modal
        title="建立新的 API Key"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        okText="建立"
        cancelText="取消"
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateKey}
          preserve={false}
        >
          <Form.Item
            label="Key 名稱"
            name="name"
            rules={[{ required: true, message: "請輸入名稱" }]}
          >
            <Input placeholder="e.g. MyProject Dev Key" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 詳細資訊 Drawer */}
      <Drawer
        title={`API Key 詳細資訊：${detailData?.name || ""}`}
        placement="right"
        width={480}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {detailLoading ? (
          <p>載入中...</p>
        ) : detailData ? (
          <DetailContent detailData={detailData} />
        ) : (
          <p>無法載入資料</p>
        )}
      </Drawer>
    </Card>
  );
}

/**
 * Drawer 內容
 * 這裡可以顯示更深入資訊，如「用量統計」、「IP 白名單」、「是否啟用 / 停用」等。
 */
function DetailContent({ detailData }) {
  const { usage = 0, limit = 1000, whitelist = [] } = detailData;

  const usagePercent = Math.min(100, Math.round((usage / limit) * 100));

  return (
    <div>
      <p>顯示更多資訊，例如 IP 白名單、用量概覽等。</p>
      <Row style={{ marginBottom: 12 }}>
        <Col span={8}>目前用量：</Col>
        <Col span={16}>
          <Progress
            percent={usagePercent}
            status={usagePercent < 100 ? "active" : "exception"}
          />
          <div style={{ marginTop: 8 }}>
            {usage} / {limit} 次呼叫
          </div>
        </Col>
      </Row>

      <Row style={{ marginBottom: 12 }}>
        <Col span={8}>IP 白名單：</Col>
        <Col span={16}>
          {whitelist.length === 0 ? (
            <Tag color="blue">未設定</Tag>
          ) : (
            whitelist.map((ip) => <Tag key={ip}>{ip}</Tag>)
          )}
        </Col>
      </Row>

      <Row style={{ marginBottom: 12 }}>
        <Col span={8}>啟用狀態：</Col>
        <Col span={16}>
          <Switch
            checked={detailData.status === "active"}
            // onChange={(checked) => ...} // 可在此呼叫後端 API 更新狀態
          />
        </Col>
      </Row>
    </div>
  );
}

/**
 * 編輯 API Key 名稱的小組件：點擊文字 -> 變成可輸入
 */
function EditableText({ text, onSave }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(text);

  const handleSubmit = () => {
    if (val.trim() && val !== text) {
      onSave(val.trim());
    }
    setEditing(false);
  };

  if (editing) {
    return (
      <Input
        autoFocus
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onBlur={handleSubmit}
        onPressEnter={handleSubmit}
        style={{ width: 150 }}
      />
    );
  }
  return <span onClick={() => setEditing(true)} style={{ cursor: "pointer" }}>{text}</span>;
}

/**
 * 將字串中段以 `*` 隱藏，例如: ABCDEFG123 => AB****3123
 */
function maskKeyString(keyString = "") {
  if (keyString.length < 8) {
    return "****";
  }
  const prefix = keyString.slice(0, 2);
  const suffix = keyString.slice(-4);
  return prefix + "****" + suffix;
}
