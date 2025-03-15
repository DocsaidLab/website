// src/pages/email-verified-success.js
import { useHistory } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { Button, Card, Typography } from 'antd';
import React from 'react';

const { Title, Paragraph } = Typography;

const localeText = {
  "zh-hant": {
    title: "驗證成功！",
    subtitle: "您的 Email 已成功驗證",
    description: "現在您可以安全登入並享受我們提供的各項服務。",
    buttonText: "返回首頁",
  },
  en: {
    title: "Email Verified Successfully",
    subtitle: "Your email has been successfully verified",
    description: "You can now safely log in and enjoy our services.",
    buttonText: "Return to Homepage",
  },
  ja: {
    title: "メール認証に成功しました！",
    subtitle: "あなたのメールは正常に認証されました",
    description: "今すぐ安全にログインして、私たちのサービスをお楽しみください。",
    buttonText: "ホームへ戻る",
  },
};

export default function EmailVerifiedSuccess() {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const history = useHistory();

  // 根據語系決定首頁路徑
  let homePath = '/';
  if (currentLocale === 'en') {
    homePath = '/en';
  } else if (currentLocale === 'ja') {
    homePath = '/ja';
  }

  const handleGoHome = () => {
    history.push(homePath);
  };

  return (
    <Layout title={text.title} description={text.description}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          padding: '2rem',
        }}
      >
        <Card
          style={{
            width: 500,
            textAlign: 'center',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
            borderRadius: 8,
          }}
          // 使用 styles.body 取代已淘汰的 bodyStyle
          styles={{ body: { padding: '2rem' } }}
        >
          <Title level={2}>{text.title}</Title>
          <Title level={4} style={{ color: '#888' }}>
            {text.subtitle}
          </Title>
          <Paragraph style={{ marginTop: '1rem', marginBottom: '2rem' }}>
            {text.description}
          </Paragraph>
          <Button type="primary" size="large" onClick={handleGoHome}>
            {text.buttonText}
          </Button>
        </Card>
      </div>
    </Layout>
  );
}
