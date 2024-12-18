import { Card } from 'antd';
import React from 'react';

export default function GPTSummaryView({ professionalReport }) {
  const MAX_LENGTH = 2000; // 可根據需求調整或之後改由props傳入

  // 檢查professionalReport的型態與內容
  let reportContent = "尚無報告";
  if (typeof professionalReport === 'string' && professionalReport.trim().length > 0) {
    reportContent = professionalReport.trim();
  }

  // 若內容過長，截斷並附上提示
  const isOverLength = reportContent.length > MAX_LENGTH;
  const displayContent = isOverLength
    ? `${reportContent.slice(0, MAX_LENGTH)}...\n\n(內容過長，已截斷顯示)`
    : reportContent;

  return (
    <Card title="分析報告" style={{ marginTop: 20 }}>
      <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {displayContent}
      </div>
    </Card>
  );
}
