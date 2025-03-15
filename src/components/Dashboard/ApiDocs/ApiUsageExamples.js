// src/components/Dashboard/ApiDocs/ApiUsageExamples.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Card, Collapse, Divider, Tabs, Tag, Typography } from "antd";
import React from "react";
import DocAligner from "./apis/DocAligner";
import MrzScanner from "./apis/MrzScanner";
import styles from "./index.module.css"; // 確保引入新的 CSS
import ParamList from "./ParamList";

const { Panel } = Collapse;
const { Paragraph, Text } = Typography;

export default function ApiUsageExamples() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  // 從各個 API 檔案根據當前語系取得定義
  const apiDefinitions = [DocAligner(currentLocale), MrzScanner(currentLocale)];

  return (
    <Collapse style={{ marginTop: 24 }} accordion defaultActiveKey={[apiDefinitions[0].key]}>
      {apiDefinitions.map((apiDef) => (
        <Panel
          key={apiDef.key}
          header={
            <div className={styles.panelHeaderWrapper}>
              <Text strong>{apiDef.title}</Text>
              <Tag color="blue">{apiDef.route}</Tag>
            </div>
          }
        >
          <Card styles={{ body: { padding: "16px 24px" } }}>
            <Paragraph>{apiDef.overview}</Paragraph>
            <Divider orientation="left" style={{ marginTop: 24 }}>
              {apiDef.text.parameters}
            </Divider>
            <ParamList data={apiDef.params} text={apiDef.text} />
            <Divider orientation="left" style={{ marginTop: 32 }}>
              {apiDef.text.codeExamples}
            </Divider>
            <Tabs defaultActiveKey={apiDef.codeExamples[0].key} size="small" items={apiDef.codeExamples} />
          </Card>
        </Panel>
      ))}
    </Collapse>
  );
}
