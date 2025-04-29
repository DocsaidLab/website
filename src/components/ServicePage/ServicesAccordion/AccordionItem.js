// src/components/ServicePage/AccordionItem.js

import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React from 'react';
import styles from './index.module.css';

const LABELS = {
  'zh-hant': {
    scenarioLabel: '適用情境：',
    deliverablesLabel: '交付內容：',
    timelineLabel: '預估時程：',
    noteLabel: '備註：',
  },
  'en': {
    scenarioLabel: 'Scenario:',
    deliverablesLabel: 'Deliverables:',
    timelineLabel: 'Timeline:',
    noteLabel: 'Note:',
  },
  'ja': {
    scenarioLabel: '想定シチュエーション：',
    deliverablesLabel: '納品内容：',
    timelineLabel: '目安期間：',
    noteLabel: '備考：',
  },
};

export default function AccordionItem({
  index,
  activeIndex,
  toggleAccordion,
  service,
  detailJSX,
}) {
  const isActive = activeIndex === index;

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleAccordion(index);
    }
  };

  // 2. 取得當前語系
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  // 3. 根據 currentLocale 取得對應標籤文字，若無就退回 zh-hant
  const translations = LABELS[currentLocale] || LABELS['zh-hant'];

  return (
    <div className={styles.cardWrapper}>
      <div
        className={`${styles.card} ${isActive ? styles.activeCard : ''}`}
        onClick={() => toggleAccordion(index)}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-expanded={isActive}
        aria-label={service.brief.title}
      >
        <div className={styles.cardTitle}>
          {/* 顯示簡短摘要 - title 本身就由 servicesData 對應語系提供 */}
          <h3>{service.brief.title}</h3>

          {/* scenario */}
          <p>
            <strong>{translations.scenarioLabel}</strong>
            {service.brief.scenario}
          </p>

          {/* deliverables */}
          <p>
            <strong>{translations.deliverablesLabel}</strong>
            {service.brief.deliverables}
          </p>

          {/* timeline */}
          <p>
            <strong>{translations.timelineLabel}</strong>
            {service.brief.timeline}
          </p>

          {/* note */}
          <p>
            <strong>{translations.noteLabel}</strong>
            {service.brief.note}
          </p>
        </div>

        <span
          className={`${styles.iconArrow} ${isActive ? styles.rotate : ''}`}
          aria-hidden="true"
        >
          ►
        </span>
      </div>

      {/* 展開區 */}
      <div
        id={`accordion-content-${index}`}
        className={`${styles.detailContent} ${isActive ? styles.expanded : ''}`}
        aria-hidden={!isActive}
      >
        {isActive && detailJSX}
      </div>
    </div>
  );
}
