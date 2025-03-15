// src/components/Dashboard/ApiDocs/CodeBlock.jsx
import { CopyOutlined } from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Tooltip } from "antd";
import React, { useState } from "react";
import styles from "./index.module.css";

const i18n = {
  "zh-hant": {
    copy: "複製程式碼",
    copied: "已複製",
  },
  en: {
    copy: "Copy Code",
    copied: "Copied",
  },
  ja: {
    copy: "コードをコピー",
    copied: "コピー済み",
  },
};

export default function CodeBlock({ codeStr }) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const localeText = i18n[currentLocale] || i18n.en;
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeStr);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("複製失敗", error);
    }
  };

  return (
    <div className={styles.codeBlockContainer}>
      <Tooltip title={copied ? localeText.copied : localeText.copy} placement="top">
        <Button
          className={styles.copyButton}
          type="text"
          icon={<CopyOutlined />}
          onClick={handleCopy}
        />
      </Tooltip>
      <pre className={styles.codeBlock}>
        <code>{codeStr}</code>
      </pre>
    </div>
  );
}
