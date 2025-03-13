import { Button, Card, Descriptions, Input, message, Space, Tabs } from "antd";
import React, { useState } from "react";
import { useAuth } from "../../../context/AuthContext";

/**
 *  假設我們只要展示某一支「公開 Token」的當前用量。
 *  也可以列出多筆 Token，讓使用者選擇要查看哪一個 Token。
 *
 *  並在畫面下方顯示「DocAligner」的使用範例程式碼 (cURL / Python).
 */
export default function DashboardApiUsage() {
  const { token } = useAuth();  // 一般登入的 token (非公開 token)
  const [publicToken, setPublicToken] = useState("");
  const [usageData, setUsageData] = useState(null);
  const [loading, setLoading] = useState(false);

  // =========== 查詢使用量 ===========
  const handleCheckUsage = async () => {
    if (!publicToken) {
      message.warning("請先輸入公開 Token");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/public/token/usage", {
        headers: {
          Authorization: `Bearer ${publicToken}`,
        },
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Failed to get usage");
      }
      const data = await res.json();
      setUsageData(data);
      message.success("Usage updated!");
    } catch (err) {
      message.error(err.message);
      setUsageData(null);
    } finally {
      setLoading(false);
    }
  };

  // 範例：顯示 cURL / Python 的程式碼，使用 docAligner-public
  // 你可用 Tabs 切換不同語言
  const docalignerCodeCurl = `curl -X POST https://api.docsaid.org/docaligner-public-predict \\
  -H "Authorization: Bearer ${publicToken || "<Your-Public-Token>"}" \\
  -F "file=@/path/to/your/document.jpg"`;

  const docalignerCodePython = `import requests

url = "https://api.docsaid.org/docaligner-public-predict"
headers = {
    "Authorization": "Bearer ${publicToken || "<Your-Public-Token>"}"
}
files = {
    "file": open("/path/to/your/document.jpg", "rb")
}
response = requests.post(url, headers=headers, files=files)
print(response.json())`;

  return (
    <Card title="API Usage & DocAligner 範例">
      <Space style={{ marginBottom: 16 }}>
        <Input
          placeholder="輸入你的公開 Token"
          value={publicToken}
          onChange={(e) => setPublicToken(e.target.value)}
          style={{ width: 400 }}
        />
        <Button type="primary" onClick={handleCheckUsage} loading={loading}>
          查詢使用量
        </Button>
      </Space>

      {usageData && (
        <Descriptions bordered column={1} size="small" style={{ marginBottom: 16 }}>
          <Descriptions.Item label="計費模式">{usageData.billing_type}</Descriptions.Item>
          <Descriptions.Item label="本小時已用次數">{usageData.used_this_hour ?? "-"}</Descriptions.Item>
          {usageData.remaining !== undefined && (
            <Descriptions.Item label="剩餘次數">{usageData.remaining}</Descriptions.Item>
          )}
        </Descriptions>
      )}

      <Card title="DocAligner 使用範例" size="small">
        <Tabs
          items={[
            {
              key: "curl",
              label: "cURL",
              children: (
                <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
{docalignerCodeCurl}
                </pre>
              ),
            },
            {
              key: "python",
              label: "Python",
              children: (
                <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
{docalignerCodePython}
                </pre>
              ),
            },
          ]}
        />
      </Card>
    </Card>
  );
}
