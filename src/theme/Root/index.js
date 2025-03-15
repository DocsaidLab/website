import React from "react";
import { AuthProvider } from "../../context/AuthContext";

export default function Root({ children }) {
  return <AuthProvider>{children}</AuthProvider>;
}
