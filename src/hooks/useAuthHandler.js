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

  // 對應到後端需要的語系
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

      // 若非成功 (2xx)
      if (!res.ok) {
        const errorMsg = data?.error || "登入失敗";
        console.error(`Error ${res.status}:`, data);

        switch (res.status) {
          case 404:
            // 使用者不存在
            return {
              success: false,
              status: 404,
              userNotFound: true,            // 前端可用於判斷「使用者不存在」
              errorMessage: errorMsg,
            };
          case 429:
            // 達到最大嘗試次數 (後端封鎖)
            return {
              success: false,
              status: 429,
              errorMessage: errorMsg,
              remainingAttempts: data.remaining_attempts ?? 0,
            };
          case 401:
            // 密碼錯誤 / 尚有剩餘次數
            return {
              success: false,
              status: 401,
              errorMessage: errorMsg,
              remainingAttempts: data.remaining_attempts, // 可能為 0, 1, 2, ...
            };
          default:
            // 其他錯誤
            return {
              success: false,
              status: res.status,
              errorMessage: errorMsg,
            };
        }
      }

      // 成功情況：帶有 access_token
      if (data.access_token) {
        await loginSuccess(data.access_token);
        return { success: true, status: res.status };
      }

      // 成功但沒有 token (較罕見)
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
   */
  const register = async ({ username, password }) => {
    setLoading(true);
    try {
      const fakeEmail = `${username}@example.com`;  // 後端必填 email

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
        // 若為 422 => { detail: [ {loc, msg, type} ] }
        if (Array.isArray(data.detail)) {
          const msgs = data.detail.map((d) => d.msg).join("; ");
          errorMsg = msgs || errorMsg;
        }
        return {
          success: false,
          error: errorMsg,
        };
      }

      // 成功 => 回傳
      if (data.token) {
        await loginSuccess(data.token);
      }

      return {
        success: !data.pwned,
        pwned: data.pwned || false,
        token: data.token,
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
