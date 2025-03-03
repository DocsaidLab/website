// 檔案： src/components/dashboard/DashboardMyComments.jsx

import {
  Button,
  Card,
  Col,
  DatePicker,
  Form,
  Input,
  message,
  Modal,
  Row,
  Space,
  Table,
} from "antd";
import moment from "moment";
import React, { useEffect, useState } from "react";
import {
  deleteCommentApi,
  getMyCommentsApi,
  updateCommentApi,
} from "../../../utils/mockApi";

const { RangePicker } = DatePicker;

export default function DashboardMyComments() {
  const [loading, setLoading] = useState(false);
  const [comments, setComments] = useState([]);

  // 用來控制「編輯留言」的彈窗
  const [editingComment, setEditingComment] = useState(null);
  const [editModalVisible, setEditModalVisible] = useState(false);

  // Table 搜尋功能
  const [searchText, setSearchText] = useState("");

  // 批次刪除（row selection）
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);

  // 建立日期篩選
  const [dateRange, setDateRange] = useState([]); // [moment|null, moment|null]

  useEffect(() => {
    fetchComments();
  }, []);

  const fetchComments = async () => {
    setLoading(true);
    try {
      const data = await getMyCommentsApi();
      setComments(data);
    } catch (err) {
      message.error(err.message || "取得留言失敗");
    } finally {
      setLoading(false);
    }
  };

  // 「編輯」按鈕 → 打開 Modal
  const handleEdit = (record) => {
    setEditingComment({ ...record });
    setEditModalVisible(true);
  };

  // 「刪除」按鈕（單筆）
  const handleDelete = async (id) => {
    try {
      await deleteCommentApi(id);
      message.success("留言已刪除");
      setComments((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      message.error(err.message || "刪除失敗");
    }
  };

  // 「儲存」編輯後的結果
  const handleSaveComment = async (values) => {
    try {
      await updateCommentApi(values.id, values.content);
      message.success("留言已更新");
      setComments((prev) =>
        prev.map((c) => (c.id === values.id ? { ...c, ...values } : c))
      );
      setEditModalVisible(false);
    } catch (err) {
      message.error(err.message || "更新失敗");
    }
  };

  // 批次刪除
  const handleBatchDelete = async () => {
    if (selectedRowKeys.length === 0) return;
    try {
      for (let commentId of selectedRowKeys) {
        await deleteCommentApi(commentId);
      }
      message.success("批次刪除成功");
      setComments((prev) => prev.filter((c) => !selectedRowKeys.includes(c.id)));
      setSelectedRowKeys([]); // 清空選取
    } catch (err) {
      message.error("批次刪除失敗：" + err.message);
    }
  };

  // Table columns
  const columns = [
    { title: "ID", dataIndex: "id", width: 80 },
    {
      title: "留言內容",
      dataIndex: "content",
      sorter: (a, b) => a.content.localeCompare(b.content),
      render: (text) => <span style={{ whiteSpace: "pre-wrap" }}>{text}</span>,
    },
    {
      title: "建立日期",
      dataIndex: "createdAt",
      width: 180,
      sorter: (a, b) =>
        new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
    },
    {
      title: "操作",
      width: 120,
      render: (_, record) => (
        <>
          <Button type="link" onClick={() => handleEdit(record)}>
            編輯
          </Button>
          <Button type="link" danger onClick={() => handleDelete(record.id)}>
            刪除
          </Button>
        </>
      ),
    },
  ];

  // 給 Table 的 rowSelection
  const rowSelection = {
    selectedRowKeys,
    onChange: (newSelectedRowKeys) => {
      setSelectedRowKeys(newSelectedRowKeys);
    },
  };

  // 前端資料篩選：根據「留言內容」與「建立日期區間」
  const filteredComments = comments.filter((c) => {
    // 1. 關鍵字篩選
    const matchSearch = c.content
      .toLowerCase()
      .includes(searchText.toLowerCase());

    // 2. 日期區間篩選
    let matchDate = true;
    if (dateRange && dateRange.length === 2 && dateRange[0] && dateRange[1]) {
      const start = dateRange[0].startOf("day");
      const end = dateRange[1].endOf("day");
      const created = moment(c.createdAt, "YYYY-MM-DD HH:mm");
      matchDate = created.isBetween(start, end, null, "[]");
    }

    return matchSearch && matchDate;
  });

  return (
    <Card>
      <h2>我的留言</h2>

      {/* 上方操作列：搜尋、日期篩選、批次刪除 */}
      <Row gutter={[16, 16]} justify="space-between" style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={8}>
          <Input.Search
            placeholder="搜尋留言內容..."
            allowClear
            enterButton
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </Col>

        <Col xs={24} sm={12} md={10}>
          <Space style={{ width: "100%" }} wrap>
            <RangePicker
              style={{ minWidth: 250 }}
              onChange={(dates) => setDateRange(dates || [])}
            />
            <Button
              onClick={handleBatchDelete}
              disabled={selectedRowKeys.length === 0}
              danger
            >
              批次刪除
            </Button>
          </Space>
        </Col>
      </Row>

      <Table
        rowKey="id"
        columns={columns}
        dataSource={filteredComments}
        loading={loading}
        pagination={{ pageSize: 5 }}
        rowSelection={rowSelection}
        scroll={{ x: "max-content" }}
      />

      {/* 編輯留言 Modal */}
      <EditCommentModal
        visible={editModalVisible}
        comment={editingComment}
        onCancel={() => setEditModalVisible(false)}
        onSave={handleSaveComment}
      />
    </Card>
  );
}

/** 編輯留言 Modal */
function EditCommentModal({ visible, onCancel, comment, onSave }) {
  const [form] = Form.useForm();

  useEffect(() => {
    if (comment) {
      form.setFieldsValue(comment);
    } else {
      form.resetFields();
    }
  }, [comment, form]);

  const onFinish = (values) => {
    onSave(values);
  };

  return (
    <Modal
      title="編輯留言"
      open={visible}
      onCancel={onCancel}
      onOk={() => form.submit()}
      okText="儲存"
      cancelText="取消"
      destroyOnClose
    >
      <Form form={form} onFinish={onFinish} layout="vertical">
        <Form.Item name="id" hidden>
          <Input />
        </Form.Item>
        <Form.Item
          label="留言內容"
          name="content"
          rules={[{ required: true, message: "請輸入留言內容" }]}
        >
          <Input.TextArea rows={4} />
        </Form.Item>
      </Form>
    </Modal>
  );
}

export async function loginApi(username, password) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!username || !password) {
        reject(new Error("登入失敗，請輸入帳號與密碼"));
      } else {
        if (username === "admin" && password === "admin123") {
          resolve({ token: "fake-admin-token" });
        } else {
          resolve({ token: "fake-jwt-token" });
        }
      }
    }, 500);
  });
}

export async function registerApi(username, password) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!username || !password) {
        reject(new Error("註冊失敗，請輸入帳號與密碼"));
      } else {
        resolve({ token: "fake-register-token" });
      }
    }, 500);
  });
}

export async function socialLoginApi(provider) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (provider === "Google" || provider === "Facebook") {
        resolve({ token: `fake-${provider.toLowerCase()}-token` });
      } else {
        reject(new Error("不支援此社群登入方式"));
      }
    }, 500);
  });
}

export async function forgotPasswordApi(email) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!email) {
        reject(new Error("請輸入 Email"));
      } else if (email.includes("@")) {
        resolve(true);
      } else {
        reject(new Error("Email 格式不正確，或該 Email 不存在於系統"));
      }
    }, 500);
  });
}

export async function getUserInfo(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        reject(new Error("無效的 token"));
      } else {
        // 假裝回傳使用者資訊
        resolve({
          id: 1,
          name: "Mock User",
          email: "mock@example.com",
          isEmailVerified: false,
          avatar: null,
        });
      }
    }, 500);
  });
}

export async function updateProfileApi(token, profileData) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入，無法更新"));
      }
      resolve(true);
    }, 500);
  });
}

export async function updatePasswordApi(token, oldPassword, newPassword) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      if (!oldPassword || !newPassword) {
        return reject(new Error("密碼資料不完整"));
      }
      resolve(true);
    }, 500);
  });
}

export async function uploadAvatarApi(token, file) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      resolve("https://via.placeholder.com/100?text=New+Avatar");
    }, 500);
  });
}

export async function resendVerificationEmailApi(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      resolve(true);
    }, 500);
  });
}

export async function deleteAccountApi(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      resolve(true);
    }, 500);
  });
}

/* ========== 留言相關 ========== */
export async function getMyCommentsApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        { id: 101, content: "Great post!", createdAt: "2023-01-01 10:20" },
        { id: 102, content: "Nice article.", createdAt: "2023-02-02 15:10" },
      ]);
    }, 500);
  });
}

export async function updateCommentApi(commentId, newContent) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!commentId || !newContent.trim()) {
        return reject(new Error("更新留言失敗：留言不可空白"));
      }
      resolve(true);
    }, 500);
  });
}

export async function deleteCommentApi(commentId) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!commentId) {
        return reject(new Error("無法刪除：缺少留言 ID"));
      }
      resolve(true);
    }, 500);
  });
}

/* ==========  API Key (單把) ==========
 * 你先前的程式只操作單把 Key, 這裡保留原本的 getMyApiKeyApi, regenerateApiKeyApi
 * 但對應你最新版的程式需要多把 Key, 故以下是「多把Key」的實作。
 * 如果你確定只需要『單把 Key』邏輯，可在前端程式中刪除對多把Key的使用。
 */
export async function getMyApiKeyApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve("fake-single-api-key-123456789");
    }, 500);
  });
}
export async function regenerateApiKeyApi(idOrNothing) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const newKey =
        "fake-key-" + Math.random().toString(36).slice(2, 8);
      resolve(newKey);
    }, 500);
  });
}

/* ========== API Key (多把) ========== */
/** 假裝後端維護多把 Key 的資料 */
let FAKE_API_KEYS = [
  {
    id: "k1",
    name: "Default Key",
    keyString: "fake-multi-key-11111111",
    createdAt: "2023-01-10T09:00:00Z",
    expireAt: "",  // 空字串代表無到期
    status: "active", // active | expired | disabled
  },
  {
    id: "k2",
    name: "Staging Env Key",
    keyString: "fake-multi-key-22222222",
    createdAt: "2023-02-15T15:30:00Z",
    expireAt: "2024-01-01T00:00:00Z",
    status: "active",
  },
];

/** 取得多把 key */
export async function getMyApiKeysApi() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      // 假設已登入
      resolve([...FAKE_API_KEYS]);
    }, 500);
  });
}

/** 新增一把 key */
export async function createApiKeyApi(name) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!name.trim()) {
        return reject(new Error("名稱不可空白"));
      }
      const newId = `k${Math.floor(Math.random() * 10000)}`;
      const newKeyStr = "fake-multi-key-" + Math.random().toString(36).slice(2, 8);
      const now = new Date().toISOString();
      const newObj = {
        id: newId,
        name: name.trim(),
        keyString: newKeyStr,
        createdAt: now,
        expireAt: "",
        status: "active",
      };
      FAKE_API_KEYS.push(newObj);
      resolve(newObj);
    }, 500);
  });
}

/** 刪除某把 key */
export async function deleteApiKeyApi(keyId) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      FAKE_API_KEYS = FAKE_API_KEYS.filter((k) => k.id !== keyId);
      resolve(true);
    }, 500);
  });
}

/** 更新 key 名稱 */
export async function updateApiKeyNameApi(keyId, newName) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!newName.trim()) {
        return reject(new Error("名稱不可空白"));
      }
      FAKE_API_KEYS = FAKE_API_KEYS.map((k) =>
        k.id === keyId ? { ...k, name: newName.trim() } : k
      );
      resolve(true);
    }, 500);
  });
}

/** 查詢 API Key 用量 (展示在 Drawer) */
export async function getApiKeyUsageApi(keyId) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      // 假裝回傳該 Key 的用量、白名單 etc.
      const findKey = FAKE_API_KEYS.find((k) => k.id === keyId);
      if (!findKey) {
        return reject(new Error("找不到此 API Key"));
      }
      // 自訂 usage, limit, ipWhitelist...
      resolve({
        usage: Math.floor(Math.random() * 500), // 已使用 0~499
        limit: 500,
        ipWhitelist: ["192.168.0.1", "127.0.0.1"],
      });
    }, 500);
  });
}

/* ==========  API Usage  ========== */
/** 後端回傳使用紀錄列表 (僅範例) */
export async function getApiUsageApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 1,
          timestamp: "2023-03-01 09:15:00",
          endpoint: "/v1/some-api",
          statusCode: 200,
          latency: 123,
        },
        {
          id: 2,
          timestamp: "2023-03-01 09:16:10",
          endpoint: "/v1/some-api",
          statusCode: 400,
          latency: 45,
        },
      ]);
    }, 500);
  });
}

/* ========== 其他(若有) ========== */
/** e.g. createCommentApi (只留範例) */
export async function createCommentApi(content) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!content) {
        return reject(new Error("留言內容不可為空"));
      }
      resolve({
        id: Math.floor(Math.random() * 100000),
        content,
        createdAt: "2023-08-01 12:00",
      });
    }, 500);
  });
}