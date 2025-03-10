// /components/forms/LoginForm.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Alert, Button, Form, Input, Typography } from "antd";
import React, { useEffect, useState } from "react";

const localeText = {
  "zh-hant": {
    usernameLabel: "帳號",
    usernameError: "請輸入帳號",
    passwordLabel: "密碼",
    passwordRequired: "請輸入密碼",
    loginBtn: "登入",
    forgotPassword: "忘記密碼？",
    loginSuccessMsg: "登入成功！",
    loginRequestTooFrequent: "登入請求過於頻繁，請稍後再試",
    maxAttemptsReached: "已達最大嘗試次數，請稍後再試",
    wrongCredentials: "密碼錯誤，剩餘嘗試次數：",
    userNotFound: "使用者不存在",
    countdownMessage: "請等待 {lockCountdown} 秒後再試",
  },
  en: {
    usernameLabel: "Username",
    usernameError: "Please enter your username",
    passwordLabel: "Password",
    passwordRequired: "Please enter your password",
    loginBtn: "Login",
    forgotPassword: "Forgot password?",
    loginSuccessMsg: "Login successful!",
    loginRequestTooFrequent: "Too many login attempts, please try again later.",
    maxAttemptsReached: "Maximum attempts reached, please try again later.",
    wrongCredentials: "Incorrect password, remaining attempts: ",
    userNotFound: "User does not exist",
    countdownMessage: "Please wait {lockCountdown} seconds before trying again.",
  },
  ja: {
    usernameLabel: "ユーザー名",
    usernameError: "ユーザー名を入力してください",
    passwordLabel: "パスワード",
    passwordRequired: "パスワードを入力してください",
    loginBtn: "ログイン",
    forgotPassword: "パスワードをお忘れですか？",
    loginSuccessMsg: "ログイン成功！",
    loginRequestTooFrequent:
      "ログインリクエストが頻繁すぎます。後でもう一度お試しください。",
    maxAttemptsReached: "最大試行回数に達しました。後でもう一度お試しください。",
    wrongCredentials: "ユーザー名またはパスワードが正しくありません。残り試行回数：",
    userNotFound: "ユーザーが存在しません",
    countdownMessage: "{lockCountdown} 秒後に再試行してください。",
  },
};

export default function LoginForm({
  onLogin, // 從 useAuthHandler 傳進來, { success, errorMessage, status, userNotFound, remainingAttempts? }
  onSuccess,
  loading,
  onToggleForgotPassword,
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;

  const [form] = Form.useForm();
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // 錯誤嘗試計數 & 鎖定
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [isLocked, setIsLocked] = useState(false);

  const MAX_ATTEMPTS = 5; // 自訂最大嘗試次數
  const LOCK_DURATION = 900; // 鎖定秒數
  const [lockCountdown, setLockCountdown] = useState(0);

  useEffect(() => {
    let timer;
    if (isLocked) {
      setLockCountdown(LOCK_DURATION);
      timer = setInterval(() => {
        setLockCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            setIsLocked(false);
            setFailedAttempts(0);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLocked]);

  const onFinish = async (values) => {
    if (isLocked) return; // 已鎖定，不允許登入

    setSubmitError("");
    setSuccessMessage("");

    const result = await onLogin(values.username, values.password);
    if (result.success) {
      setSuccessMessage(text.loginSuccessMsg);
      setFailedAttempts(0);
      onSuccess?.();
    } else {
      // 先檢查是否為 429 => 後端封鎖
      if (result.status === 429) {
        setIsLocked(true);
        setSubmitError(text.loginRequestTooFrequent);
        return;
      }

      // 檢查是否使用者不存在
      if (result.userNotFound) {
        // 不進行失敗次數紀錄，直接顯示「使用者不存在」
        setSubmitError(text.userNotFound);
        return;
      }

      // 其餘錯誤 (包含密碼錯誤)
      // 如果後端帶有 remainingAttempts，則跟著更新
      if (typeof result.remainingAttempts === "number") {
        const rem = result.remainingAttempts;
        if (rem <= 0) {
          // 若後端表示剩餘次數0 => 也視為封鎖
          setIsLocked(true);
          setSubmitError(text.maxAttemptsReached);
        } else {
          setSubmitError(`${text.wrongCredentials}${rem}`);
        }
      } else {
        // 若後端沒有帶 remainingAttempts，則用本地邏輯做簡單累計
        const newCount = failedAttempts + 1;
        setFailedAttempts(newCount);

        if (newCount >= MAX_ATTEMPTS) {
          setIsLocked(true);
          setSubmitError(text.maxAttemptsReached);
        } else {
          setSubmitError(`${text.wrongCredentials}${MAX_ATTEMPTS - newCount}`);
        }
      }
    }
  };

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={onFinish}
      style={{ maxWidth: 400, margin: "0 auto" }}
    >
      <Form.Item
        label={text.usernameLabel}
        name="username"
        rules={[{ required: true, message: text.usernameError }]}
      >
        <Input disabled={isLocked} />
      </Form.Item>

      <Form.Item
        label={text.passwordLabel}
        name="password"
        rules={[{ required: true, message: text.passwordRequired }]}
      >
        <Input.Password disabled={isLocked} />
      </Form.Item>

      {successMessage && (
        <Alert
          style={{ marginBottom: 10 }}
          message={successMessage}
          type="success"
          showIcon
        />
      )}
      {submitError && (
        <Alert
          style={{ marginBottom: 10 }}
          message={submitError}
          type="error"
          showIcon
        />
      )}
      {isLocked && (
        <Alert
          style={{ marginBottom: 10 }}
          message={text.countdownMessage.replace(
            "{lockCountdown}",
            lockCountdown
          )}
          type="warning"
          showIcon
        />
      )}

      <Form.Item>
        <Button
          type="primary"
          htmlType="submit"
          block
          loading={loading}
          disabled={isLocked}
        >
          {text.loginBtn}
        </Button>
      </Form.Item>

      {onToggleForgotPassword && (
        <Typography.Link
          style={{ float: "right", marginTop: 8 }}
          onClick={onToggleForgotPassword}
        >
          {text.forgotPassword}
        </Typography.Link>
      )}
    </Form>
  );
}
