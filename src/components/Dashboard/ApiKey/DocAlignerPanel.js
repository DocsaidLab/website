// src/components/Dashboard/ApiKey/DocAlignerPanel.jsx
import { Button, Card, Descriptions, Input, Space, Tabs } from "antd";
import React from "react";

export default function DocAlignerPanel({
  publicToken,
  setPublicToken,
  usageData,
  setUsageData,
  checkLoading,
  onCheckUsage
}) {
  // 產生範例程式碼
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
    <Card size="small" bordered={false}>
      <Space style={{ marginBottom: 16 }}>
        <Input
          placeholder="輸入你的公開 Token"
          value={publicToken}
          onChange={(e) => setPublicToken(e.target.value)}
          style={{ width: 400 }}
        />
        <Button type="primary" onClick={onCheckUsage} loading={checkLoading}>
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
