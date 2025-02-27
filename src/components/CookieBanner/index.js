import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React, { useEffect, useState } from 'react';

// 1) 避免 SSR 時操作 window/cookie
function useIsBrowser() {
  const [isBrowser, setIsBrowser] = useState(false);
  useEffect(() => {
    setIsBrowser(typeof window !== 'undefined');
  }, []);
  return isBrowser;
}

// 2) Cookie 讀寫
function setCookie(name, value, days) {
  const expires = new Date();
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires.toUTCString()}; path=/`;
}

function getCookie(name) {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return match ? decodeURIComponent(match[2]) : null;
}

// 3) 多國語系文案
const translations = {
  'zh-TW': {
    title: 'Cookie 使用告知',
    message: '我們使用 Cookie 分析流量並提升使用者體驗。持續使用即表示您同意。詳情請見我們的隱私政策：',
    accept: '同意',
    revoke: '不同意',
    policyLink: 'https://docsaid.org/privacy-policy',
  },
  en: {
    title: 'Cookie Notice',
    message: 'We use cookies to analyze traffic and enhance user experience. By continuing, you agree. Learn more in our Privacy Policy:',
    accept: 'Accept',
    revoke: 'Reject',
    policyLink: 'https://docsaid.org/en/privacy-policy',
  },
  ja: {
    title: 'クッキー使用のお知らせ',
    message: '当サイトはクッキーを使用し、トラフィック分析と体験向上を行います。継続利用で同意とみなします。詳しくは **プライバシーポリシー** をご覧ください。',
    accept: '同意する',
    revoke: '同意しない',
    policyLink: 'https://docsaid.org/ja/privacy-policy',
  },
};

export default function CookieConsentCard() {
  const isBrowser = useIsBrowser();
  const { i18n } = useDocusaurusContext();
  const locale = i18n.currentLocale;
  const t = translations[locale] || translations['zh-TW']; // fallback: 中文

  const [showCard, setShowCard] = useState(false);

  useEffect(() => {
    if (!isBrowser) return;

    // 在開發模式下強制顯示，每次重整都會清除 cookie，方便測試
    if (process.env.NODE_ENV === 'development') {
      document.cookie = 'cookie_consent=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
    }

    const consent = getCookie('cookie_consent');
    if (!consent) {
      setShowCard(true);
    }
  }, [isBrowser]);

  const handleAccept = () => {
    if (!isBrowser) return;
    setCookie('cookie_consent', 'true', 7); // 7 天後到期
    setShowCard(false);
  };

  const handleRevoke = () => {
    if (!isBrowser) return;
    // 「不同意」就等同撤回，刪除 Cookie
    document.cookie = 'cookie_consent=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
    setShowCard(false);
  };

  if (!showCard) return null;

  return (
    <div style={styles.cardContainer}>
      <div style={styles.header}>
        <span style={styles.title}>{t.title}</span>
      </div>
      <div style={styles.content}>
        <p style={styles.text}>
          {t.message}
          <a href={t.policyLink} style={styles.link}>
            Cookie Policy
          </a>
        </p>
        <div style={styles.btnGroup}>
          <button onClick={handleAccept} style={styles.acceptBtn}>
            {t.accept}
          </button>
          <button onClick={handleRevoke} style={styles.revokeBtn}>
            {t.revoke}
          </button>
        </div>
      </div>
    </div>
  );
}

// 4) 樣式：使用灰階漸層、半透明白背景
const styles = {
  cardContainer: {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    width: '340px',
    backgroundColor: 'rgba(255, 255, 255, 0.85)', // 半透明白
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    borderRadius: '8px',
    zIndex: 9999,
    fontFamily: 'sans-serif',
    animation: 'fadeIn 0.5s ease-in-out',
  },
  header: {
    background: 'linear-gradient(135deg, #eeeeee 0%, #cccccc 100%)',
    color: '#333',
    borderTopLeftRadius: '8px',
    borderTopRightRadius: '8px',
    padding: '8px 12px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 'bold',
  },
  content: {
    padding: '12px',
  },
  text: {
    fontSize: '14px',
    marginBottom: '12px',
    lineHeight: 1.4,
    color: '#333',
  },
  link: {
    marginLeft: '4px',
    color: '#555',
    textDecoration: 'underline',
  },
  btnGroup: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  acceptBtn: {
    backgroundColor: 'rgba(120, 120, 120, 0.8)',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    padding: '8px 24px',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'background-color 0.5s ease',
  },
  revokeBtn: {
    backgroundColor: 'rgba(100, 100, 100, 0.5)',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    padding: '8px 16px',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'background-color 0.5s ease',
  },
};
