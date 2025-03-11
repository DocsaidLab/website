// src/theme/NotFound/Content/index.js
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import React, { useEffect, useState } from 'react';


const texts = {
  "en": {
    title: "Page Not Found",
    apology: "Sorry, we couldn't find the page you're looking for.",
    reason: "The site structure has been modified and you might have clicked an outdated link.",
    suggestion: "Please click on the top navigation bar to find the information you need.",
    redirect: (countdown) =>
      countdown > 0 ? `Redirecting to homepage in ${countdown} seconds...` : 'Redirecting...',
    report: "If you notice this error, please report the error details here:",
    reportLinkText: "Report 404 Error Details",
  },
  "zh-hant": {
    title: "找不到頁面",
    apology: "很抱歉，我們無法找到您要的頁面。",
    reason: "網頁結構已經修改了，而您可能選到過時的連結。",
    suggestion: "請點擊上方導航欄，或許可以找到您要的資訊。",
    redirect: (countdown) =>
      countdown > 0 ? `將在 ${countdown} 秒後自動返回首頁...` : '即將跳轉...',
    report: "如果您發現此錯誤，請回報錯誤細節到此處：",
    reportLinkText: "回報 404 錯誤細節",
  },
  "ja": {
    title: "ページが見つかりません",
    apology: "申し訳ありませんが、お探しのページが見つかりませんでした。",
    reason: "サイトの構造が変更されたため、古いリンクをクリックした可能性があります。",
    suggestion: "上部のナビゲーションバーをクリックして、必要な情報を探してください。",
    redirect: (countdown) =>
      countdown > 0 ? `ホームページへ ${countdown} 秒後にリダイレクトします...` : 'リダイレクト中...',
    report: "このエラーに気づいた場合、以下から詳細を報告してください：",
    reportLinkText: "404エラーの詳細を報告",
  },
};


export default function NotFoundContent({ className }) {
  const {i18n: { currentLocale } } = useDocusaurusContext();
  const langText = texts[currentLocale]
  const [countdown, setCountdown] = useState(15);

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prevCountdown) => (prevCountdown > 0 ? prevCountdown - 1 : 0));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  let homePath = '/img/error-icon.png';
  let redirectPath = '/';
  if (currentLocale === 'en') {
    homePath = '/en/img/error-icon.png';
    redirectPath = '/en';
  } else if (currentLocale === 'ja') {
    homePath = '/ja/img/error-icon.png';
    redirectPath = '/ja';
  }

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      window.location.href = redirectPath;
    }
  }, [countdown]);

  return (
    <main className={clsx('container margin-vert--xl', className)}>
      <div
        className="row"
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          flexDirection: 'column',
          textAlign: 'center',
          animation: 'fadeIn 0.5s ease-in-out',
        }}>
        <img
          src={homePath}
          alt="Error icon"
          style={{
            width: '150px',
            height: '150px',
            marginBottom: '20px',
            animation: 'bounce 1s infinite',
          }}
        />
        <div>
          <Heading as="h1" className="hero__title">
            {langText.title}
          </Heading>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            {langText.apology}
          </p>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            {langText.reason}
          </p>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            {langText.suggestion}
          </p>
          <p aria-live="polite" style={{ fontSize: '1rem', color: '#555' }}>
            {langText.redirect(countdown)}
          </p>
          <p style={{ fontSize: '1rem', marginTop: '10px' }}>
            {langText.report}{' '}
            <a href="https://github.com/orgs/DocsaidLab/discussions/15" target="_blank" rel="noopener noreferrer">
              {langText.reportLinkText}
            </a>
          </p>
        </div>
        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
          }
        `}</style>
      </div>
    </main>
  );
}
