// /src/context/AuthContext.js
import React, { createContext, useContext, useEffect, useMemo, useState } from "react";


const AuthContext = createContext({
  token: null,
  user: null,
  loading: true,
  loginSuccess: () => {},
  logout: () => {},
  setUser: () => {},
});

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // 1. 啟動時檢查 localStorage，有 token 就去後端拿使用者資料
  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    if (!savedToken) {
      setLoading(false);
      return;
    }

    setToken(savedToken);

    // 呼叫 /auth/me
    fetch("https://api.docsaid.org/auth/me", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${savedToken}`,
        "Content-Type": "application/json",
      },
    })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error("Token invalid or expired");
        }
        return res.json();
      })
      .then((data) => {
        // 後端若成功回傳使用者資訊 => 設置 user
        setUser(data);
      })
      .catch(() => {
        // token 無效 => 移除
        setToken(null);
        localStorage.removeItem("token");
      })
      .finally(() => setLoading(false));
  }, []);

  // 2. 登入成功後 => 存 token, localStorage, 也可再次呼叫 /auth/me
  const loginSuccess = async (loginToken) => {
    try {
      // 確認 token 可用 => 再呼叫 /auth/me 拿資料
      const res = await fetch("https://api.docsaid.org/auth/me", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${loginToken}`,
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) throw new Error("Invalid token");
      const userData = await res.json();

      setToken(loginToken);
      setUser(userData);
      localStorage.setItem("token", loginToken);
    } catch (error) {
      console.error("登入失敗，無效的 token", error);
      // 若需要，也可在這裡彈錯誤訊息
    }
  };

  // 3. 登出 => 清空 token, user
  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("token");
  };

  const value = useMemo(() => ({
    token,
    user,
    loading,
    loginSuccess,
    logout,
    setUser,
  }), [token, user, loading]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth 必須在 AuthProvider 內使用");
  }
  return context;
}
