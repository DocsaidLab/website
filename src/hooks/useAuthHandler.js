// /src/hooks/useAuthHandler.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function useAuthHandler() {
  const { loginSuccess } = useAuth();
  const [loading, setLoading] = useState(false);

  // 從 Docusaurus 取得當前語系
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  // 後端所需語系
  const serverLang = currentLocale;

  /**
   * 登入
   * @param {string} username
   * @param {string} password
   * @returns {Promise<{
   *   success: boolean;
   *   errorMessage?: string;
   *   status?: number;
   *   userNotFound?: boolean;
   *   remainingAttempts?: number;
   * }>}
   */
  const login = async (username, password) => {
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": serverLang,
        },
        body: JSON.stringify({ username, password }),
      });

      let data;
      try {
        data = await res.json();
      } catch {
        data = {};
      }

      if (!res.ok) {
        const errorMsg = data?.error || "登入失敗";
        console.error(`Error ${res.status}:`, data);
        switch (res.status) {
          case 404:
            return {
              success: false,
              status: 404,
              userNotFound: true,
              errorMessage: errorMsg,
            };
          case 429:
            return {
              success: false,
              status: 429,
              errorMessage: errorMsg,
              remainingAttempts: data.remaining_attempts ?? 0,
            };
          case 401:
            return {
              success: false,
              status: 401,
              errorMessage: errorMsg,
              remainingAttempts: data.remaining_attempts,
            };
          default:
            return {
              success: false,
              status: res.status,
              errorMessage: errorMsg,
            };
        }
      }

      // 成功情況：回傳 access_token
      if (data.access_token) {
        await loginSuccess(data.access_token);
        return { success: true, status: res.status };
      }

      return {
        success: false,
        status: res.status,
        errorMessage: "登入失敗: token 不存在",
      };
    } catch (error) {
      console.error("login error:", error);
      return {
        success: false,
        status: 500,
        errorMessage: error.message || "登入請求失敗",
      };
    } finally {
      setLoading(false);
    }
  };

  /**
   * 註冊
   * @param {object} param0 包含 username 與 password
   * @returns {Promise<{ success: boolean; error?: string; pwned?: boolean; }>}
   */
  const register = async ({ username, password }) => {
    setLoading(true);
    try {
      const fakeEmail = `${username}@example.com`;  // 後端 register 必填 email
      const res = await fetch("https://api.docsaid.org/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept-Language": serverLang,
        },
        body: JSON.stringify({
          username,
          password,
          email: fakeEmail,
          phone: null,
          birth: null,
          avatar: null,
        }),
      });

      let data = {};
      try {
        data = await res.json();
      } catch (parseError) {
        console.error("parse error:", parseError);
      }

      if (!res.ok) {
        let errorMsg = data?.error || "Registration failed";
        if (Array.isArray(data.detail)) {
          const msgs = data.detail.map((d) => d.msg).join("; ");
          errorMsg = msgs || errorMsg;
        }
        return {
          success: false,
          error: errorMsg,
        };
      }

      // 後端 register 回傳 RegisterResponse，不含 token
      // 前端可以提示使用者註冊成功，請進行登入
      return {
        success: !data.pwned,
        pwned: data.pwned || false,
        // 可回傳註冊後的使用者資料，例如 data.id 等
        userId: data.id,
      };
    } catch (error) {
      console.error("register error:", error);
      return {
        success: false,
        error: error.message || "Network error",
      };
    } finally {
      setLoading(false);
    }
  };

  return {
    login,
    register,
    loading,
  };
}
