// src/components/MultiCardsCTA/ServiceCard.js
import { Card, Tag } from 'antd';
import React from 'react';
import styles from './index.module.css';

export default function ServiceCard({ cardData }) {
  const {
    icon,
    tag,
    title,
    concept,
    bulletTitle,
    bulletPoints,
    buttonLink,
  } = cardData;

  return (
    <Card
      className={`${styles.card} ${styles.fadeInUp} ${styles.hoverTransform}`}
      /* 讓 Card 撐滿父層 Col 的高度 */
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
      }}
      /* 移除原本按鈕，改成整張卡片點擊 */
      onClick={() => {
        if (buttonLink) {
          window.open(buttonLink, '_blank');
        }
      }}
    >
      {icon && (
        <div style={{ textAlign: 'center', marginTop: '1rem' }}>
          <img
            src={icon}
            alt={`${title} icon`}
            style={{ width: 48, height: 48 }}
          />
        </div>
      )}

      {tag && (
        <Tag color="orange" className={styles['card__tag']}>
          {tag}
        </Tag>
      )}

      <h4 className={styles['card__title']}>{title}</h4>
      <p className={styles['card__concept']}>{concept}</p>

      <div className={styles['card__bulletHeader']}>
        <h5 className={styles['card__bulletTitle']}>{bulletTitle}</h5>
      </div>
      <ul className={styles['card__bulletList']}>
        {bulletPoints?.map((bp, idx) => (
          <li key={idx} className={styles['card__bulletItem']}>
            {bp}
          </li>
        ))}
      </ul>

      {/* 移除按鈕，footer 區塊也一起拿掉 */}
    </Card>
  );
}
