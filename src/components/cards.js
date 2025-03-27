// src/components/cards.js
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React from 'react';

export const UserCard = ({ children }) => {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const userLabel = {
    "zh-hant": '我：',
    "en": 'Me:',
    "ja": '私：',
  }[currentLocale] || 'Me:';

  return (
    <div style={{
      border: '1px solid #ccc',
      borderRadius: '4px',
      padding: '10px',
      margin: '10px 0',
      background: '#fff7e6',
      maxWidth: '70%',
      marginLeft: 'auto'
    }}>
      <strong>{userLabel}</strong>
      <div>{children}</div>
    </div>
  );
};

export const ChatGPTCard = ({ children }) => {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const chatgptLabel = {
    zh: 'ChatGPT：',
    en: 'ChatGPT:',
    ja: 'ChatGPT：',
  }[currentLocale] || 'ChatGPT:';

  return (
    <div style={{
      border: '1px solid #ccc',
      borderRadius: '4px',
      padding: '10px',
      margin: '10px 0',
      background: '#e6f7ff',
      maxWidth: '80%',
      marginRight: 'auto'
    }}>
      <strong>{chatgptLabel}</strong>
      <div>{children}</div>
    </div>
  );
};
