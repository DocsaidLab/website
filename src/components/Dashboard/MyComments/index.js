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
  Spin,
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

  // 搜尋關鍵字
  const [searchText, setSearchText] = useState("");
  // 建立日期篩選 (start ~ end)
  const [dateRange, setDateRange] = useState([]);
  // 批次刪除多選
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);

  // 編輯留言 Modal
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editingComment, setEditingComment] = useState(null);

  // 初次載入 or 重新整理 → 取得留言
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

  // 單筆：開啟「編輯 Modal」
  const handleEdit = (record) => {
    setEditingComment({ ...record });
    setEditModalVisible(true);
  };

  // 單筆：執行刪除
  const handleDelete = async (id) => {
    try {
      await deleteCommentApi(id);
      message.success("留言已刪除");
      setComments((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      message.error(err.message || "刪除失敗");
    }
  };

  // 編輯後：儲存變更
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
      setSelectedRowKeys([]);
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

  // 多選設定
  const rowSelection = {
    selectedRowKeys,
    onChange: (keys) => setSelectedRowKeys(keys),
  };

  // 前端篩選：依「關鍵字 + 日期區間」
  const filteredComments = comments.filter((item) => {
    // 1) 關鍵字
    const matchSearch = item.content
      .toLowerCase()
      .includes(searchText.toLowerCase());

    // 2) 日期
    let matchDate = true;
    if (dateRange?.[0] && dateRange?.[1]) {
      const start = dateRange[0].startOf("day");
      const end = dateRange[1].endOf("day");
      const created = moment(item.createdAt, "YYYY-MM-DD HH:mm");
      matchDate = created.isBetween(start, end, null, "[]");
    }
    return matchSearch && matchDate;
  });

  return (
    <Card>
      <h2>我的留言</h2>

      {/* 若正在 loading，顯示 Spin */}
      {loading ? (
        <Spin style={{ display: "block", margin: "16px 0" }} />
      ) : (
        <Row gutter={[16, 16]} justify="space-between" style={{ marginBottom: 16 }}>
          {/* 搜尋、日期篩選、批次刪除、重新整理 */}
          <Col xs={24} sm={12} md={8}>
            <Input.Search
              placeholder="搜尋留言內容..."
              allowClear
              enterButton
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Space wrap>
              <RangePicker
                style={{ minWidth: 220 }}
                onChange={(dates) => setDateRange(dates || [])}
              />
              <Button
                danger
                onClick={handleBatchDelete}
                disabled={selectedRowKeys.length === 0}
              >
                批次刪除
              </Button>
              <Button onClick={fetchComments}>重新整理</Button>
            </Space>
          </Col>
        </Row>
      )}

      {/* 主表格 */}
      <Table
        rowKey="id"
        columns={columns}
        dataSource={filteredComments}
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

/** 編輯留言彈窗 */
function EditCommentModal({ visible, comment, onCancel, onSave }) {
  const [form] = Form.useForm();

  useEffect(() => {
    if (comment) {
      form.setFieldsValue(comment);
    } else {
      form.resetFields();
    }
  }, [comment, form]);

  const handleFinish = (values) => {
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
      <Form form={form} onFinish={handleFinish} layout="vertical">
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
