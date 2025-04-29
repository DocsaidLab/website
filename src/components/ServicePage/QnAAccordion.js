// src/components/ServicePage/QnAAccordion.js

import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React, { useState } from 'react';
import styles from './QnAAccordion.module.css';
import { QNA_DATA } from './QnAData'; // <-- QnAData.js

export default function QnAAccordion() {
  const [activeIndex, setActiveIndex] = useState(null);

  // 1. 取得目前語系
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  // 2. 讀取對應語系的 QnA 陣列 (若找不到就預設 'zh-hant')
  const qnaList = QNA_DATA[currentLocale] || QNA_DATA['zh-hant'];

  const toggleQnA = (index) => {
    setActiveIndex(index === activeIndex ? null : index);
  };

  return (
    <section className={styles.qnaSection}>
      <div className={styles.qnaContainer}>
        {qnaList.map((item, index) => {
          const isActive = activeIndex === index;
          return (
            <div key={index} className={styles.qnaItem}>
              <button
                className={`${styles.question} ${isActive ? styles.active : ''}`}
                onClick={() => toggleQnA(index)}
              >
                {item.question}
              </button>
              {isActive && (
                <div className={styles.answer}>
                  {renderAnswer(item.answer)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * 將 answer 物件 (paragraphs, list, extraNote) 轉成 <p>, <ul>, <li>, <strong> 等 React 元素。
 */
function renderAnswer(answerObj) {
  if (!answerObj) return null;

  const { paragraphs = [], list = [], extraNote } = answerObj;

  return (
    <>
      {/* 1. paragraphs */}
      {paragraphs.map((text, idx) => (
        <p key={`p-${idx}`}>{text}</p>
      ))}

      {/* 2. list */}
      {Array.isArray(list) && list.length > 0 && (
        <ul>
          {list.map((liText, liIdx) => (
            <li key={`li-${liIdx}`}>{liText}</li>
          ))}
        </ul>
      )}

      {/* 3. extraNote (若有) */}
      {extraNote && <p>{extraNote}</p>}
    </>
  );
}
