// src/components/MultiCardsCTA/SimpleCTA.js
import React from 'react';
import styles from './index.module.css';

/**
 * 已移除按鈕及連結標籤，改為整塊卡片 clickable
 */
export default function SimpleCTA({
  iconSrc,
  iconAlt,
  title,
  subtitle,
  buttonLink,
  buttonImg,
  variant = 'default',
}) {
  return (
    <div
      className={`
        ${styles.simpleCta}
        ${styles[`simple-cta__${variant}`]}
        ${styles.fadeInUp}
        ${styles.hoverTransform}
      `}
      /* 如果有 link，就可以點擊 */
      onClick={() => {
        if (buttonLink) {
          window.open(buttonLink, '_blank');
        }
      }}
      style={{
        cursor: buttonLink ? 'pointer' : 'default',
      }}
    >
      {iconSrc && (
        <img
          src={iconSrc}
          alt={iconAlt || 'cta-icon'}
          className={styles['simple-cta__icon']}
        />
      )}

      <h3 className={styles['simple-cta__title']}>{title}</h3>
      <p className={styles['simple-cta__subtitle']}>{subtitle}</p>

      {/* 如果有圖片, 仍顯示; 移除按鈕, 讓整塊卡片可點擊 */}
      {buttonImg && (
        <div className={styles['simple-cta__buttonWrapper']}>
          <img
            src={buttonImg}
            alt={iconAlt || 'cta-button'}
            className={styles['simple-cta__buttonImg']}
          />
        </div>
      )}
    </div>
  );
}
