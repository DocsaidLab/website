// src/pages/index.js

import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import React, { useEffect, useState } from 'react';

// 拆分後的檔案
import FadeInSection from '@site/src/components/FadeInSection';
import HomepageHeader from '@site/src/components/HomepageHeader';
import demoContent from '@site/src/data/demoContent';
import featuredProjectsData from '@site/src/data/featuredProjectsData';
import testimonialsData from '@site/src/data/testimonialsData';

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo/DocAlignerDemoWrapper';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import { Timeline } from 'antd';

import styles from './index.module.css';

export default function Home() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [visibleCount, setVisibleCount] = useState(5);
  const currentProjects = featuredProjectsData[currentLocale] || featuredProjectsData['en'];

  // 載入 recent_updates_data
  let recentUpdates;
  if (currentLocale === 'zh-hant') {
    recentUpdates = require('@site/papers/recent_updates_data.json');
  } else if (currentLocale === 'en') {
    recentUpdates = require('@site/i18n/en/docusaurus-plugin-content-docs-papers/current/recent_updates_data.json');
  } else if (currentLocale === 'ja') {
    recentUpdates = require('@site/i18n/ja/docusaurus-plugin-content-docs-papers/current/recent_updates_data.json');
  } else {
    recentUpdates = require('@site/papers/recent_updates_data.json');
  }

  const localeContent = demoContent[currentLocale] || demoContent['en'];

  // 監聽捲動，顯示「回到頂端」
  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

  const convertMdLinkToRoute = (mdLink) => {
    return mdLink
      .replace(/^.\//, '/papers/')
      .replace(/\.md$/, '')
      .replace(/\/index$/, '')
      .replace(/\/(\d{4}-)/, '/');
  };

  return (
    <Layout title={`Hello from ${siteConfig.title}`} description="Description">
      {/* Hero 區塊改用抽出的 HomepageHeader */}
      <HomepageHeader
        siteTitle={siteConfig.title}
        siteTagline={siteConfig.tagline}
      />

      <main className={styles.mainWrapper}>

        {/* Features */}
        <FadeInSection className={styles.sectionBox}>
          <HomepageFeatures />
        </FadeInSection>

        {/* Featured Projects */}
        <FadeInSection className={styles.sectionBox}>
          <Heading as="h2" className={styles.sectionTitle}>
            <Translate id="homepage.featuredProjectsTitle">精選作品</Translate>
          </Heading>
          <div className={styles.projectsGrid}>
            {currentProjects.map((proj, idx) => (
              <div key={idx} className={styles.projectCard}>
                <img src={proj.image} alt={proj.title} className={styles.projectImage} />
                <div className={styles.projectContent}>
                  <h3>{proj.title}</h3>
                  <p>{proj.description}</p>
                  <Link className={styles.projectLink} to={proj.link}>
                    <Translate id="homepage.learnMore">了解更多 →</Translate>
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </FadeInSection>

        {/* Timeline 區塊 (論文筆記近期更新) */}
        <FadeInSection className={styles.sectionBox}>
          <Heading as="h2" className={styles.sectionTitle}>
            <Translate id="homepage.recentUpdatesTitle">論文筆記近期更新</Translate>
          </Heading>
          <Timeline mode="alternate">
            {recentUpdates.slice(0, visibleCount).map((item, idx) => {
              const finalRoute = convertMdLinkToRoute(item.link);
              return (
                <Timeline.Item key={idx} label={item.date}>
                  <Link to={finalRoute} className={styles.timelineLink}>
                    {item.combinedTitle}
                  </Link>
                </Timeline.Item>
              );
            })}
          </Timeline>
          {visibleCount < recentUpdates.length && (
            <div className={styles.loadMoreWrapper}>
              <button onClick={() => setVisibleCount((prev) => prev + 5)} className={styles.loadMoreBtn}>
                <Translate id="homepage.loadMore">載入更多</Translate>
              </button>
            </div>
          )}
        </FadeInSection>

        {/* DocAligner Demo 區塊 */}
        <FadeInSection className={styles.sectionBox}>
          <Heading as="h2" className={styles.sectionTitle}>{localeContent.title}</Heading>
          <div className={styles.demoDescription}>
            {localeContent.description.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </div>
          <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />
        </FadeInSection>

        {/* Testimonials 區塊 */}
        <FadeInSection className={styles.sectionBox}>
          <Heading as="h2" className={styles.sectionTitle}>
            <Translate id="homepage.testimonialsTitle">讀者回饋</Translate>
          </Heading>
          <div className={styles.testimonialsWrapper}>
            {testimonialsData.map((testi, i) => (
              <div key={i} className={styles.testimonialCard}>
                <img src={testi.avatar} alt={testi.name} className={styles.testimonialAvatar} />
                <div className={styles.testimonialContent}>
                  <p className={styles.testimonialFeedback}>"{testi.feedback}"</p>
                  <p className={styles.testimonialAuthor}>— {testi.name}</p>
                </div>
              </div>
            ))}
          </div>
        </FadeInSection>

      </main>

      {/* 回到頂端按鈕 */}
      {showBackToTop && (
        <button className={styles.backToTopBtn} onClick={scrollToTop}>
          ⬆
        </button>
      )}
    </Layout>
  );
}
