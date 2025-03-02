// src/components/dashboard/DashboardApiUsage.jsx
import { message, Spin, Table } from "antd";
import React, { useEffect, useState } from "react";
import { getApiUsageApi } from "../../../utils/mockApi";

export default function DashboardApiUsage() {
  const [loading, setLoading] = useState(false);
  const [usageList, setUsageList] = useState([]);

  useEffect(() => {
    fetchUsage();
  }, []);

  const fetchUsage = async () => {
    setLoading(true);
    try {
      const data = await getApiUsageApi();
      setUsageList(data);
    } catch (err) {
      message.error(err.message || "取得 API 使用紀錄失敗");
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    { title: "呼叫時間", dataIndex: "timestamp", width: 200 },
    { title: "API 路徑", dataIndex: "endpoint" },
    { title: "狀態碼", dataIndex: "statusCode", width: 120 },
    { title: "耗時 (ms)", dataIndex: "latency", width: 120 },
  ];

  return (
    <div>
      <h2>API 使用紀錄</h2>
      {loading ? (
        <Spin />
      ) : (
        <Table
          rowKey="id"
          columns={columns}
          dataSource={usageList}
          pagination={{ pageSize: 10 }}
        />
      )}
    </div>
  );
}
