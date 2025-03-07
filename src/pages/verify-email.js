// src/pages/verify-email.js
import { useLocation } from "@docusaurus/router";
import { message } from "antd";
import React, { useEffect, useState } from "react";

// 假設你的 AuthContext 放在 src/context/AuthContext.js
import { useAuth } from "../context/AuthContext";

export default function VerifyEmailPage() {
  const { verifyEmail } = useAuth();    // 你在 AuthContext 裡寫好的函式
  const location = useLocation();       // 取得目前路由資訊 (含 search)
  const [status, setStatus] = useState("驗證中...");

  useEffect(() => {
    // 解析網址後面的 "?token=xxxx"
    const searchParams = new URLSearchParams(location.search);
    const token = searchParams.get("token");
    if (!token) {
      setStatus("無效的連結，找不到 token");
      return;
    }

    // 呼叫後端進行驗證
    verifyEmail({ token })
      .then(() => {
        setStatus("Email 驗證成功！");
        message.success("Email 驗證成功，您可以使用完整功能囉！");
      })
      .catch((err) => {
        setStatus(err.message || "驗證失敗");
        message.error(err.message || "驗證失敗");
      });
  }, [location, verifyEmail]);

  return (
    <div style={{ padding: 24 }}>
      <h2>{status}</h2>
    </div>
  );
}
