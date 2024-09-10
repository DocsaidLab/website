import Translate from '@docusaurus/Translate';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import React, { useEffect, useState } from 'react';


export default function NotFoundContent({className}) {

  const [countdown, setCountdown] = useState(15);

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prevCountdown) => (prevCountdown > 0 ? prevCountdown - 1 : 0));
    }, 1000);

    if (countdown === 0) {
      window.location.href = '/';
    }

    return () => clearInterval(timer);
  }, [countdown]);

  return (
    <main className={clsx('container margin-vert--xl', className)}>
      <div className="row" style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          flexDirection: 'column',
          textAlign: 'center',
          animation: 'fadeIn 0.5s ease-in-out',
        }}>

        <img
          src="/img/error-icon.png"
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
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page">
              Page Not Found
            </Translate>
          </Heading>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            很抱歉，我們無法找到您要的頁面。
          </p>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            網頁結構已經修改了，而您可能選到過時的連結。
          </p>
          <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
            請點擊上方導航欄，或許可以找到您要的資訊。
          </p>
          <p aria-live="polite" style={{ fontSize: '1rem', color: '#555' }}>
            {countdown > 0 ? `將在 ${countdown} 秒後自動返回首頁...` : '即將跳轉...'}
          </p>
        </div>

        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes bounce {
            0%, 100% {
              transform: translateY(0);
            }
            50% {
              transform: translateY(-10px);
            }
          }
        `}</style>

      </div>
    </main>
  );
}
