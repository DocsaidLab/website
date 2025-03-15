// src/components/Dashboard/ApiKey/DocAlignerPanel.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Card, Descriptions, Input, Space, Tabs } from "antd";
import React from "react";
import { apiKeyLocale } from "./locales";

export default function DocAlignerPanel({
  publicToken,
  setPublicToken,
  usageData,
  setUsageData,
  checkLoading,
  onCheckUsage,
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  const docalignerCodeCurl = `curl -X POST https://api.docsaid.org/docaligner-public-predict \\
  -H "Authorization: Bearer ${publicToken || "<${text.docAlignerYourPublicToken}>" }" \\
  -F "file=@/path/to/your/document.jpg"`;

  const docalignerCodePython = `import requests

url = "https://api.docsaid.org/docaligner-public-predict"
headers = {
    "Authorization": "Bearer ${publicToken || "<${text.docAlignerYourPublicToken}>" }"
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
          placeholder={text.docAlignerInputPlaceholder}
          value={publicToken}
          onChange={(e) => setPublicToken(e.target.value)}
          style={{ width: 400 }}
        />
        <Button type="primary" onClick={onCheckUsage} loading={checkLoading}>
          {text.docAlignerCheckUsageButton}
        </Button>
      </Space>

      {usageData && (
        <Descriptions bordered column={1} size="small" style={{ marginBottom: 16 }}>
          <Descriptions.Item label={text.docAlignerBillingType}>
            {usageData.billing_type}
          </Descriptions.Item>
          <Descriptions.Item label={text.docAlignerUsedThisHour}>
            {usageData.used_this_hour ?? "-"}
          </Descriptions.Item>
          {usageData.remaining !== undefined && (
            <Descriptions.Item label={text.docAlignerRemaining}>
              {usageData.remaining}
            </Descriptions.Item>
          )}
        </Descriptions>
      )}

      <Card title={text.docAlignerUsageExampleTitle} size="small">
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
