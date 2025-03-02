import React, { createContext, useContext, useEffect, useState } from "react";
import { getUserInfo } from "../utils/mockApi";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    if (savedToken) {
      setToken(savedToken);
      getUserInfo(savedToken)
        .then((data) => {
          setUser(data);
          setLoading(false);
        })
        .catch(() => {
          setToken(null);
          localStorage.removeItem("token");
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const loginSuccess = (loginToken) => {
    setToken(loginToken);
    localStorage.setItem("token", loginToken);
    getUserInfo(loginToken)
      .then((data) => setUser(data))
      .catch(() => {
        setToken(null);
        localStorage.removeItem("token");
      });
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("token");
  };

  const value = {
    token,
    user,
    loading,
    loginSuccess,
    logout,
    setUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}