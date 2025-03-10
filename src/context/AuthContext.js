// /src/context/AuthContext.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
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

// 後端 API 網址
const API_BASE = "https://api.docsaid.org";


export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // 2. 取得 Docusaurus 的 currentLocale
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  // 3. 根據 currentLocale 對照，取得後端需要的語言代碼
  const serverLang = currentLocale;

  /**
   * 通用 API 請求工具
   * @param {string} endpoint API 路徑 (例如 /auth/me)
   * @param {string} method HTTP 方法
   * @param {string|null} token 若有 token 則自動加入 Authorization header
   * @param {object|FormData|null} body 若為物件則轉成 JSON；若為 FormData 則直接傳送
   */
  async function apiRequest(endpoint, method = "GET", tokenArg = null, body = null) {
    const headers = {};
    // 加入 Bearer Token
    if (tokenArg) {
      headers.Authorization = `Bearer ${tokenArg}`;
    }
    // 加入語系
    headers["Accept-Language"] = serverLang;

    // 判斷是否為 FormData
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

    const contentType = res.headers.get("Content-Type");
    let data = {};
    if (contentType && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      data = await res.text();
    }

    if (!res.ok) {
      const errorMsg = data?.error || data?.detail || data || "請求失敗";
      if (process.env.NODE_ENV !== "production") {
        console.error(`API Request Failed [${method} ${endpoint}]:`, data);
      }
      const err = new Error(errorMsg);
      // 若後端回傳特定欄位(如 remaining_attempts)，可一併加到 Error 物件
      if (data?.remaining_attempts !== undefined) {
        err.remaining_attempts = data.remaining_attempts;
      }
      throw err;
    }
    return data;
  }

  // App 啟動時從 localStorage 取得 token 並嘗試拉取用戶資料
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

  // 登入成功後，儲存 Token 並刷新使用者資料
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

  // 根據語系決定首頁路徑
  let homePath = '/';
  if (currentLocale === 'en') {
    homePath = '/en';
  } else if (currentLocale === 'ja') {
    homePath = '/ja';
  }

  // 登出
  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("token");
    window.location.href = homePath;
  };

  // 更新個人資料
  const updateProfile = async (payload) => {
    if (!token) throw new Error("尚未登入");
    const newUser = await apiRequest("/auth/profile", "PUT", token, payload);
    setUser(newUser);
    return newUser;
  };

  // 寄送驗證信
  const sendVerificationEmail = async (email) => {
    if (!token) throw new Error("尚未登入");
    return apiRequest("/auth/send-verification-email", "POST", token, { email });
  };

  // 驗證 Email (注意後端為 GET + Redirect 設計，fetch 可能只會拿到 redirect 前的狀態)
  const verifyEmail = async (verifyToken) => {
    // 範例：若前端還是想要直接呼叫
    const data = await apiRequest(`/auth/verify-email?token=${verifyToken}`, "GET", null, null);
    // 成功後可重新抓取 /auth/me 更新 user 狀態
    if (token) {
      const updatedUser = await apiRequest("/auth/me", "GET", token);
      setUser(updatedUser);
    }
    return data;
  };

  // 變更密碼
  const changePassword = async (oldPassword, newPassword) => {
    if (!token) throw new Error("尚未登入");
    return apiRequest("/auth/change-password", "POST", token, {
      old_password: oldPassword,
      new_password: newPassword,
    });
  };

  // 刪除帳號
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
