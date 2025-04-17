// src/components/MultiCardsCTA/SimpleCTA.js
import { Button } from 'antd';
import React from 'react';
import styles from './index.module.css';

export default function SimpleCTA({
  iconSrc,
  iconAlt,
  title,
  subtitle,
  buttonLink,
  buttonText,
  buttonImg,
  variant = 'default',
}) {
  const hasButton = (buttonText && buttonLink) || (buttonImg && buttonLink);

  return (
    <div
      className={`${styles.simpleCta} simple-cta__${variant} fadeInUp hoverTransform`}
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

      {hasButton && (
        <div className={styles['simple-cta__buttonWrapper']}>
          {buttonImg ? (
            <a
              href={buttonLink}
              target="_blank"
              rel="noreferrer"
            >
              <img
                src={buttonImg}
                alt={iconAlt || 'cta-button'}
                className={styles['simple-cta__buttonImg']}
              />
            </a>
          ) : (
            <Button
              type="primary"
              href={buttonLink}
              target="_blank"
              rel="noreferrer"
            >
              {buttonText}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
