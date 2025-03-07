// /src/context/AuthContext.js
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const AuthContext = createContext({
  token: null,
  user: null,
  loading: true,
  loginSuccess: () => {},
  logout: () => {},
  setUser: () => {},
  updateProfile: () => {},
  verifyEmail: () => {},
  sendVerificationEmail: () => {},
  changePassword: () => {},
  deleteAccount: () => {},
});

const API_BASE = "https://api.docsaid.org";

/**
 * 通用 API 請求工具
 * @param {string} endpoint API 路徑 (例如 /auth/me)
 * @param {string} method HTTP 方法
 * @param {string|null} token 若有 token 則自動加入 Authorization header
 * @param {object|FormData|null} body 若為物件則轉成 JSON；若為 FormData 則直接傳送
 */
async function apiRequest(endpoint, method = "GET", token = null, body = null) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const isFormData = body instanceof FormData;
  if (body && !isFormData) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method,
    headers,
    body: body || undefined,
  });

  // 判斷回傳格式
  const contentType = res.headers.get("Content-Type");
  let data = {};
  if (contentType && contentType.includes("application/json")) {
    data = await res.json();
  } else {
    data = await res.text();
  }
  if (!res.ok) {
    // 若在開發環境中，詳細錯誤資訊會輸出到 console；生產環境則只拋出簡單訊息
    if (process.env.NODE_ENV !== "production") {
      console.error(`API Request Failed [${method} ${endpoint}]:`, data);
    }
    throw new Error(data.detail || data || "請求失敗");
  }
  return data;
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // 啟動時檢查 localStorage 並取得使用者資訊
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = localStorage.getItem("token");
      if (!savedToken) {
        setLoading(false);
        return;
      }
      setToken(savedToken);
      try {
        const data = await apiRequest("/auth/me", "GET", savedToken);
        setUser(data);
      } catch (error) {
        // Token 失效 => 清空
        setToken(null);
        localStorage.removeItem("token");
      } finally {
        setLoading(false);
      }
    };
    initAuth();
  }, []);

  // 登入成功後，儲存 token 並刷新使用者資料
  const loginSuccess = async (loginToken) => {
    try {
      const userData = await apiRequest("/auth/me", "GET", loginToken);
      setToken(loginToken);
      setUser(userData);
      localStorage.setItem("token", loginToken);
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.error("登入失敗", error);
      }
    }
  };

  // 登出：清空 token 與使用者資料，並導向首頁
  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("token");
    window.location.href = "/";
  };

  // 更新個人資料，若 email 重複會拋出錯誤
  const updateProfile = async (payload) => {
    if (!token) throw new Error("尚未登入");
    try {
      const newUser = await apiRequest("/auth/profile", "PUT", token, payload);
      setUser(newUser);
      return newUser;
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.error("更新個人資料錯誤:", error);
      }
      throw new Error(error.message || "更新失敗");
    }
  };

  // 寄送驗證信
  const sendVerificationEmail = async (email) => {
    if (!token) throw new Error("尚未登入");
    return apiRequest("/auth/send-verification-email", "POST", token, { email });
  };

  // 驗證 Email，成功後刷新使用者資訊
  const verifyEmail = async (payload) => {
    const data = await apiRequest("/auth/verify-email", "POST", null, payload);
    if (token) {
      try {
        const updatedUser = await apiRequest("/auth/me", "GET", token);
        setUser(updatedUser);
      } catch (e) {
        if (process.env.NODE_ENV !== "production") {
          console.error("刷新使用者資訊失敗", e);
        }
      }
    }
    return data;
  };

  // 變更密碼
  const changePassword = async (oldPassword, newPassword) => {
    if (!token) throw new Error("尚未登入");
    return apiRequest(
      "/auth/change-password",
      "POST",
      token,
      { old_password: oldPassword, new_password: newPassword }
    );
  };

  // 刪除帳號，成功後自動登出
  const deleteAccount = async () => {
    if (!token) throw new Error("尚未登入");
    await apiRequest("/auth/delete", "DELETE", token);
    logout();
    return { message: "帳號已刪除" };
  };

  const value = useMemo(
    () => ({
      token,
      user,
      loading,
      loginSuccess,
      logout,
      setUser,
      updateProfile,
      verifyEmail,
      sendVerificationEmail,
      changePassword,
      deleteAccount,
    }),
    [token, user, loading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth 必須在 AuthProvider 內使用");
  }
  return context;
}
