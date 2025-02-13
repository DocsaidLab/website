---
sidebar_position: 1
---

# Nginx 介紹

從頭開始瑣碎的學習，肯定讓人覺得興致缺缺。

所以我們先定一個情境，之後一切的學習，都是為了完成這個情境。

## 情境說明

我們在伺服器上建立了一個模型推論服務的 API，可以透過 HTTP 的方式，將資料傳送給伺服器，並取得結果。

現在的情境就是：**我們需要對外開放這個 API 端點。**

舉例來說，我們的 API 端點是 `https://temp_api.example.com/test`，然後預期可以用 `curl` 來取得回傳值，像這樣：

```bash
API_URL="https://temp_api.example.com/test"

curl -X GET $API_URL
```

回傳值為可能是一個字串：

```json
{
  "message": "API is running!"
}
```

:::warning
上述的 API 端點是假設說明，實際上並不存在。
:::

## 前置準備

在這個的情境中，我們會透過 Let's Encrypt 來取得 SSL 憑證，以提供 HTTPS 服務。

由於 Let's Encrypt 需要域名解析，而且不接受 IP 位址，所以如果你也想要跟著做，請確保你有一個域名可以使用。

## 學習目標

我們預期要學習以下幾個內容：

1. ✅ 安裝 Nginx
2. ✅ 設定 Nginx 反向代理
3. ✅ 設定 Nginx 啟用 HTTPS
4. ✅ 設定 Nginx 安全強化
5. [ ] 設定 Nginx 提供靜態資源
6. [ ] 設定 Nginx 負載均衡
7. [ ] 設定 Nginx 日誌與監控

## 參考資料

- [**Nginx 官方文件**](https://nginx.org/en/docs/)
- [**Nginx 入門指南**](https://www.w3cschool.cn/nginx/)
- [**淺談 Nginx 基本配置、負載均衡、緩存和反向代理**](https://www.maxlist.xyz/2020/06/18/flask-nginx/)
