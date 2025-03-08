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
    loginRequestTooFrequent: "ログインリクエストが頻繁すぎます。後でもう一度お試しください。",
    maxAttemptsReached: "最大試行回数に達しました。後でもう一度お試しください。",
    wrongCredentials: "ユーザー名またはパスワードが正しくありません。残り試行回数：",
    countdownMessage: "{lockCountdown} 秒後に再試行してください。",
  },
};

export default function LoginForm({
  onLogin,                // 呼叫後端 /auth/login 的函式，回傳格式：{ success, errorMessage, status }
  onSuccess,             // 登入成功後的 callback (可用來跳轉 / 關閉 Modal)
  loading,               // 登入按鈕的加載狀態
  onToggleForgotPassword // 切換到「忘記密碼」畫面的函式 (若需要)
}) {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;

  // 顯示錯誤與成功提示
  const [submitError, setSubmitError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // antd form
  const [form] = Form.useForm();

  // 錯誤嘗試計數與鎖定狀態
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [isLocked, setIsLocked] = useState(false);
  const MAX_ATTEMPTS = 5; // 可自訂最大嘗試次數
  const LOCK_DURATION = 900; // 預設鎖定時長（秒）
  const [lockCountdown, setLockCountdown] = useState(0);

  // 當進入鎖定狀態時，啟動倒數計時
  useEffect(() => {
    let timer;
    if (isLocked) {
      setLockCountdown(LOCK_DURATION);
      timer = setInterval(() => {
        setLockCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            // 倒數結束，自動解鎖並重置失敗計數
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

  /**
   * 提交表單:
   *  1. 清空錯誤與成功提示
   *  2. 呼叫 onLogin(username, password)，預期回傳 { success, errorMessage, status }
   *  3. 若成功 => 顯示成功訊息 / 呼叫 onSuccess
   *  4. 若失敗：
   *     - 若 status 為 429，表示後端正在封鎖，直接鎖定並顯示通用 ban 訊息
   *     - 其他錯誤則累計失敗次數，達到上限時鎖定並啟動倒數，並只顯示通用錯誤訊息
   */
  const onFinish = async (values) => {
    if (isLocked) return; // 鎖定狀態下不執行登入
    setSubmitError("");
    setSuccessMessage("");

    const result = await onLogin(values.username, values.password);
    if (result.success) {
      setSuccessMessage(text.loginSuccessMsg);
      setFailedAttempts(0);
      onSuccess?.();
    } else {
      // 若回傳 429，表示後端封鎖中，顯示通用 ban 訊息
      if (result.status === 429) {
        setIsLocked(true);
        setSubmitError(text.loginRequestTooFrequent);
      } else {
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
      {/* 帳號欄位 */}
      <Form.Item
        label={text.usernameLabel}
        name="username"
        rules={[{ required: true, message: text.usernameError }]}
      >
        <Input disabled={isLocked} />
      </Form.Item>

      {/* 密碼欄位 */}
      <Form.Item
        label={text.passwordLabel}
        name="password"
        rules={[{ required: true, message: text.passwordRequired }]}
      >
        <Input.Password disabled={isLocked} />
      </Form.Item>

      {/* 登入成功訊息 */}
      {successMessage && (
        <Alert
          style={{ marginBottom: 10 }}
          message={successMessage}
          type="success"
          showIcon
        />
      )}

      {/* 錯誤訊息 */}
      {submitError && (
        <Alert
          style={{ marginBottom: 10 }}
          message={submitError}
          type="error"
          showIcon
        />
      )}

      {/* 若鎖定中，顯示倒數訊息 */}
      {isLocked && (
        <Alert
          style={{ marginBottom: 10 }}
          message={text.countdownMessage.replace("{lockCountdown}", lockCountdown)}
          type="warning"
          showIcon
        />
      )}

      {/* 提交按鈕 */}
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

      {/* 忘記密碼連結 */}
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
