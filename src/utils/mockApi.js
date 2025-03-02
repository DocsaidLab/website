// mockApi.js

export async function loginApi(username, password) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!username || !password) {
        reject(new Error("登入失敗，請輸入帳號與密碼"));
      } else {
        if (username === "admin" && password === "admin123") {
          resolve({ token: "fake-admin-token" });
        } else {
          resolve({ token: "fake-jwt-token" });
        }
      }
    }, 1000);
  });
}

export async function registerApi(username, password) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!username || !password) {
        reject(new Error("註冊失敗，請輸入帳號與密碼"));
      } else {
        resolve({ token: "fake-register-token" });
      }
    }, 1000);
  });
}

export async function getUserInfo(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (token) {
        resolve({ id: 1, name: "Mock User" });
      } else {
        reject(new Error("無效的 token"));
      }
    }, 800);
  });
}

export async function socialLoginApi(provider) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (provider === "Google" || provider === "Facebook") {
        resolve({ token: `fake-${provider.toLowerCase()}-token` });
      } else {
        reject(new Error("不支援此社群登入方式"));
      }
    }, 1200);
  });
}

// ↓↓↓新增↓↓↓
export async function forgotPasswordApi(email) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!email) {
        reject(new Error("請輸入 Email"));
      } else if (email.includes("@")) {
        resolve(true);
      } else {
        reject(new Error("Email 格式不正確，或該 Email 不存在於系統"));
      }
    }, 1000);
  });
}

export async function resetPasswordApi(token, newPassword) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        reject(new Error("無效的重設密碼連結或 Token"));
      } else if (!newPassword) {
        reject(new Error("請輸入新密碼"));
      } else if (newPassword.length < 8) {
        reject(new Error("新密碼至少 8 碼"));
      } else {
        resolve(true);
      }
    }, 1000);
  });
}

export async function updateProfileApi(token, { name, email }) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入，無法更新"));
      }
      resolve(true);
    }, 1000);
  });
}

export async function updatePasswordApi(token, oldPassword, newPassword) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      if (!oldPassword || !newPassword) {
        return reject(new Error("密碼資料不完整"));
      }
      resolve(true);
    }, 1000);
  });
}

export async function uploadAvatarApi(token, file) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      const newUrl = "https://via.placeholder.com/100?text=New+Avatar";
      resolve(newUrl);
    }, 1500);
  });
}

export async function getMyCommentsApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        { id: 101, content: "Great post!", createdAt: "2023-01-01 10:20" },
        { id: 102, content: "Nice article.", createdAt: "2023-02-02 15:10" },
      ]);
    }, 800);
  });
}

export async function updateCommentApi(commentId, newContent) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!commentId || !newContent) {
        return reject(new Error("更新留言失敗"));
      }
      resolve(true);
    }, 500);
  });
}

export async function deleteCommentApi(commentId) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!commentId) {
        return reject(new Error("無法刪除"));
      }
      resolve(true);
    }, 500);
  });
}

// API Key
export async function getMyApiKeyApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve("fake-api-key-123456789");
    }, 600);
  });
}

export async function regenerateApiKeyApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      const newKey =
        "fake-api-key-" + Math.random().toString(36).slice(2, 8);
      resolve(newKey);
    }, 800);
  });
}

// API Usage
export async function getApiUsageApi() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 1,
          timestamp: "2023-03-01 09:15:00",
          endpoint: "/v1/some-api",
          statusCode: 200,
          latency: 123,
        },
        {
          id: 2,
          timestamp: "2023-03-01 09:16:10",
          endpoint: "/v1/some-api",
          statusCode: 400,
          latency: 45,
        },
      ]);
    }, 600);
  });
}

/** 重新寄送 Email 驗證信 */
export async function resendVerificationEmailApi(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      // 假裝寄出成功
      resolve(true);
    }, 1200);
  });
}

/** 刪除帳號 */
export async function deleteAccountApi(token) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (!token) {
        return reject(new Error("尚未登入"));
      }
      // 假裝刪除成功
      resolve(true);
    }, 1000);
  });
}