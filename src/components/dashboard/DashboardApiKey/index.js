import { Button, Card, message } from "antd";
import React, { useEffect, useState } from "react";
import { getMyApiKeyApi, regenerateApiKeyApi } from "../../../utils/mockApi";

export default function DashboardApiKey() {
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const fetchApiKey = async () => {
    setLoading(true);
    try {
      const key = await getMyApiKeyApi();
      setApiKey(key);
    } catch (err) {
      message.error(err.message || "取得 API Key 失敗");
    } finally {
      setLoading(false);
    }
  };

  const regenerateKey = async () => {
    setLoading(true);
    try {
      const newKey = await regenerateApiKeyApi();
      setApiKey(newKey);
      message.success("已重新生成 API Key");
    } catch (err) {
      message.error(err.message || "重新生成失敗");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApiKey();
  }, []);

  return (
    <Card>
      <h2>我的 API Key</h2>
      <p style={{ marginTop: 16 }}>目前的 API Key：</p>
      <pre style={{ background: "#f5f5f5", padding: 8 }}>
        {apiKey || "N/A"}
      </pre>
      <Button onClick={regenerateKey} loading={loading} style={{ marginTop: 8 }}>
        重新生成
      </Button>
    </Card>
  );
}
