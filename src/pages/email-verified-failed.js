// src/pages/email-verified-failed.js
import { useHistory } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { Button, Card, Typography } from 'antd';
import React from 'react';

const { Title, Paragraph } = Typography;

const localeText = {
  "zh-hant": {
    title: "驗證失敗",
    subtitle: "您的 Email 驗證未成功",
    description: "可能原因包括驗證連結已過期或無效。請嘗試重新寄送驗證信，或聯繫客服以獲得協助。",
    homeButton: "返回首頁",
    resendButton: "重新寄送驗證信",
  },
  en: {
    title: "Email Verification Failed",
    subtitle: "Your email verification was not successful",
    description: "This may be due to an expired or invalid verification link. Please try resending the verification email or contact support for assistance.",
    homeButton: "Return to Homepage",
    resendButton: "Resend Verification Email",
  },
  ja: {
    title: "メール認証に失敗しました",
    subtitle: "メール認証が正常に完了しませんでした",
    description: "リンクの有効期限が切れているか、無効な可能性があります。認証メールの再送信を試すか、サポートにお問い合わせください。",
    homeButton: "ホームへ戻る",
    resendButton: "認証メールを再送信",
  },
};

export default function EmailVerifiedFailed() {
  const { i18n: { currentLocale } } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const history = useHistory();

  // 根據語系設定首頁導向路徑
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
          styles={{ body: { padding: '2rem' } }}
        >
          <Title level={2}>{text.title}</Title>
          <Title level={4} style={{ color: '#888' }}>{text.subtitle}</Title>
          <Paragraph style={{ marginTop: '1rem', marginBottom: '2rem' }}>
            {text.description}
          </Paragraph>
          <div>
            <Button type="primary" size="large" onClick={handleGoHome} style={{ marginRight: 16 }}>
              {text.homeButton}
            </Button>
          </div>
        </Card>
      </div>
    </Layout>
  );
}
