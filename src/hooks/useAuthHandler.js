// /src/hooks/useAuthHandler.js
import { message } from "antd";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";


export default function useAuthHandler() {
  const { loginSuccess } = useAuth();
  const [loading, setLoading] = useState(false);

  /**
   * 登入
   * @param {string} username
   * @param {string} password
   * @returns {boolean} 是否成功
   */
  const login = async (username, password) => {
    setLoading(true);
    try {
      const res = await fetch("https://api.docsaid.org/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      let data;
      try {
        data = await res.json();
      } catch (err) {
        data = {};
      }

      // 若非成功狀態，根據回傳內容整理錯誤訊息並回傳包含狀態碼的物件
      if (!res.ok) {
        let errorMsg = "登入失敗";
        if (data.detail) {
          if (typeof data.detail === "object") {
            errorMsg = Object.entries(data.detail)
              .map(([key, value]) => `${key}: ${value}`)
              .join(" | ");
          } else {
            errorMsg = data.detail;
          }
        }
        console.error(`Error ${res.status}:`, data);
        return { success: false, errorMessage: `Error ${res.status}: ${errorMsg}`, status: res.status };
      }

      if (data.access_token) {
        await loginSuccess(data.access_token);
        message.success("登入成功");
        return { success: true, status: res.status };
      }

      message.error("登入失敗: token 不存在");
      return { success: false, errorMessage: "登入失敗: token 不存在", status: res.status };
    } catch (error) {
      console.error("login error:", error);
      message.error(error.message || "登入請求失敗");
      return { success: false, errorMessage: error.message || "登入請求失敗", status: 500 };
    } finally {
      setLoading(false);
    }
  };


  /**
   * 註冊
   * @param {{username: string, password: string, force?: boolean}} payload
   * @returns {{success: boolean, pwned?: boolean, error?: string, token?: string}}
   */
  const register = async ({ username, password, force = false }) => {
    setLoading(true);
    try {
      // 補上假 email (後端必填)
      const fakeEmail = `${username}@example.com`;

      const res = await fetch("https://api.docsaid.org/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
          email: fakeEmail,
          phone: null,
          birth: null,
          avatar: null,
          force: force
        }),
      });

      if (!res.ok) {
        let errData = {};
        try {
          errData = await res.json();
        } catch (parseError) {
          console.error("parse error:", parseError);
        }

        // 若為 422 => { detail: [ {loc, msg, type} ] }
        if (Array.isArray(errData.detail)) {
          const msgs = errData.detail.map((d) => d.msg).join("; ");
          return { success: false, error: msgs || "Registration failed" };
        }
        // 其他錯誤 => 取 detail / error
        return {
          success: false,
          error: errData.detail || errData.error || "Registration failed",
        };
      }

      // 成功 => 解析回傳 JSON => e.g. { id, username, pwned, token? }
      const data = await res.json();

      // 若後端同時回傳 token => 可以立即登入
      if (data.token) {
        // 寫入 AuthContext => localStorage
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
