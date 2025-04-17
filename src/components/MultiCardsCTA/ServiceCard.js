// src/components/MultiCardsCTA/ServiceCard.js
import { Button, Card, Tag } from 'antd';
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
    buttonText,
    buttonLink,
  } = cardData;

  return (
    <Card
      className={`${styles.card} fadeInUp hoverTransform`}
      hoverable
      // 有 icon 則可以放在 Card cover
      cover={
        icon ? (
          <div style={{ textAlign: 'center', marginTop: '1rem' }}>
            <img
              src={icon}
              alt={`${title} icon`}
              style={{ width: 48, height: 48 }}
            />
          </div>
        ) : null
      }
    >
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

      {/* Footer 區塊 */}
      {buttonText && buttonLink && (
        <div className={styles['card__footer']}>
          <Button
            type="primary"
            href={buttonLink}
            target="_blank"
            rel="noreferrer"
            block
          >
            {buttonText}
          </Button>
        </div>
      )}
    </Card>
  );
}
