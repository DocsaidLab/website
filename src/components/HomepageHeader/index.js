// src/components/HomepageHeader/index.js
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import React, { useRef } from 'react';
import styles from './styles.module.css';

export default function HomepageHeader({ siteTitle, siteTagline }) {
  const nextSectionRef = useRef(null);

  const handleScrollDown = () => {
    if (nextSectionRef.current) {
      nextSectionRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className={clsx(styles.heroBanner)}>
      <div className={styles.heroInner}>
        <Heading as="h1" className={styles.heroTitle}>
          {siteTitle}
        </Heading>
        <p className={styles.heroSubtitle}>
          <Translate id="homepage.tagline">{siteTagline}</Translate>
        </p>
        <div className={styles.buttons}>
          <Link className="button button--secondary button--lg" to="/docs">
            <Translate id="homepage.button1">開始探索</Translate>
          </Link>
        </div>
        <div className={styles.scrollDownArrow} onClick={handleScrollDown}>
          ↓
        </div>
      </div>

      <div className={styles.heroWave}>
        {/* SVG 波浪 */}
        <svg viewBox="0 0 1440 80" xmlns="http://www.w3.org/2000/svg">
          <path fill="var(--ifm-color-primary)" d="M0,10 C360,80 1080,0 1440,50 L1440,80 L0,80 Z"/>
        </svg>
      </div>

      <div ref={nextSectionRef}></div>
    </header>
  );
}
