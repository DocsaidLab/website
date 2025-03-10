// src/pages/reset-password.js
import Layout from '@theme/Layout';
import React from "react";
import ResetPasswordForm from "../components/forms/ResetPasswordForm";

export default function ResetPasswordPage() {
  return (
    <Layout title="Reset Password" description="Reset your account password">
        <div style={{ padding: "2rem" }}>
        <ResetPasswordForm />
        </div>
  </Layout>
  );
}
