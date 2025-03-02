import { message } from "antd";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { loginApi, registerApi, socialLoginApi } from "../utils/mockApi";

export default function useAuthHandler() {
  const { loginSuccess } = useAuth();
  const [loading, setLoading] = useState(false);

  /**
   * 可透過參數自訂成功/失敗提示：
   * handleAuth(apiFn, args..., { successMsg, errorMsg })
   * 若沒有傳入，預設就用「操作成功」/「操作失敗」。
   */
  const handleAuth = async (apiCall, ...args) => {
    setLoading(true);
    try {
      const result = await apiCall(...args);
      loginSuccess(result.token);
      message.success("操作成功！");
      return true;
    } catch (err) {
      message.error(err.message || "操作失敗");
      return false;
    } finally {
      setLoading(false);
    }
  };

  const login = (username, password) =>
    handleAuth(loginApi, username, password);

  const register = (username, password) =>
    handleAuth(registerApi, username, password);

  const socialLogin = (provider) =>
    handleAuth(socialLoginApi, provider);

  return {
    login,
    register,
    socialLogin,
    loading,
    setLoading,
  };
}
