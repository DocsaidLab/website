// src/components/Dashboard/ApiDocs/index.js
import React from "react";
import ApiUsageExamples from "./ApiUsageExamples";
import styles from "./index.module.css";

export default function DashboardApiDocs() {
  return (
    <div className={styles.apiDocsContainer}>
      <h2>API Documents</h2>
      <p>這裡是各種 API 的技術文件與使用範例介紹頁。</p>

      <ApiUsageExamples />
    </div>
  );
}
