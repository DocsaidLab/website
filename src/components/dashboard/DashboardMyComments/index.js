// src/components/dashboard/DashboardMyComments.jsx
import { Button, Form, Input, message, Modal, Table } from "antd";
import React, { useEffect, useState } from "react";
import { deleteCommentApi, getMyCommentsApi, updateCommentApi } from "../../../utils/mockApi";

export default function DashboardMyComments() {
  const [loading, setLoading] = useState(false);
  const [comments, setComments] = useState([]);
  const [editingComment, setEditingComment] = useState(null);
  const [editModalVisible, setEditModalVisible] = useState(false);

  useEffect(() => {
    fetchComments();
  }, []);

  const fetchComments = async () => {
    setLoading(true);
    try {
      // 呼叫後端抓此用戶的所有留言
      const data = await getMyCommentsApi();
      setComments(data);
    } catch (err) {
      message.error(err.message || "取得留言失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (record) => {
    setEditingComment({ ...record });
    setEditModalVisible(true);
  };

  const handleDelete = async (id) => {
    try {
      await deleteCommentApi(id);
      message.success("留言已刪除");
      setComments((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      message.error(err.message || "刪除失敗");
    }
  };

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

  const columns = [
    { title: "ID", dataIndex: "id", width: 80 },
    {
      title: "留言內容",
      dataIndex: "content",
      render: (text) => <>{text}</>,
    },
    {
      title: "建立日期",
      dataIndex: "createdAt",
      width: 160,
    },
    {
      title: "操作",
      render: (text, record) => (
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

  return (
    <div>
      <h2>我的留言</h2>
      <Table
        rowKey="id"
        columns={columns}
        dataSource={comments}
        loading={loading}
        pagination={{ pageSize: 5 }}
      />

      <EditCommentModal
        visible={editModalVisible}
        comment={editingComment}
        onCancel={() => setEditModalVisible(false)}
        onSave={handleSaveComment}
      />
    </div>
  );
}

function EditCommentModal({ visible, onCancel, comment, onSave }) {
  const [form] = Form.useForm();

  useEffect(() => {
    if (comment) {
      form.setFieldsValue(comment);
    }
  }, [comment]);

  const onFinish = (values) => {
    onSave(values);
  };

  return (
    <Modal
      title="編輯留言"
      open={visible}
      onCancel={onCancel}
      onOk={() => form.submit()}
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
