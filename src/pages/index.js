import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { Timeline } from 'antd';
import { motion } from 'framer-motion';
import React, { useEffect, useState } from 'react';

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo/DocAlignerDemoWrapper';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageHeader from '@site/src/components/HomepageHeader';

import demoContent from '@site/src/data/demoContent';
import featuredProjectsData from '@site/src/data/featuredProjectsData';
import testimonialsData from '@site/src/data/testimonialsData';

import styles from './index.module.css';

// -- 1) 定義父容器 (container) 與 子元素 (item) 的 variants --
const containerVariants = {
  hidden: {}, // 可以空著，代表尚未顯示
  show: {
    transition: {
      // 讓子元素交錯進入
      staggerChildren: 0.15,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: 'easeOut' },
  },
};

// -- 2) 精選作品：交錯進入範例 --
function StaggeredFeaturedProjects({ projects }) {
  return (
    <motion.section
      className={styles.sectionBox}
      // 父容器，用 containerVariants
      variants={containerVariants}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.2 }}
      // 一次就好，捲動多少高度算進入
    >
      <motion.div variants={itemVariants}>
        <h2 className={styles.sectionTitle}>
          <Translate id="homepage.featuredProjectsTitle">精選作品</Translate>
        </h2>
      </motion.div>

      {/* 作品列表容器，也用 containerVariants，讓底下 card 逐個進場 */}
      <motion.div
        className={styles.projectsGrid}
        variants={containerVariants}
      >
        {projects.map((proj, idx) => (
          <motion.div
            key={idx}
            className={styles.projectCard}
            variants={itemVariants}
          >
            <img
              src={proj.image}
              alt={proj.title}
              className={styles.projectImage}
            />
            <div className={styles.projectContent}>
              <h3>{proj.title}</h3>
              <p>{proj.description}</p>
              <Link className={styles.projectLink} to={proj.link}>
                <Translate id="homepage.learnMore">了解更多 →</Translate>
              </Link>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </motion.section>
  );
}

// -- 3) Timeline 區塊：交錯進入範例 --
function StaggeredTimeline({ recentUpdates, visibleCount, setVisibleCount, convertMdLinkToRoute }) {
  return (
    <motion.section
      className={styles.sectionBox}
      variants={containerVariants}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.2 }}
    >
      <motion.div variants={itemVariants}>
        <h2 className={styles.sectionTitle}>
          <Translate id="homepage.recentUpdatesTitle">論文筆記近期更新</Translate>
        </h2>
      </motion.div>

      <motion.div variants={containerVariants}>
        <Timeline mode="alternate">
          {recentUpdates.slice(0, visibleCount).map((item, idx) => {
            const finalRoute = convertMdLinkToRoute(item.link);
            return (
              <Timeline.Item key={idx} label={item.date}>
                {/* 每項 Timeline 也可用 itemVariants 讓文字淡入 */}
                <motion.div variants={itemVariants}>
                  <Link to={finalRoute} className={styles.timelineLink}>
                    {item.combinedTitle}
                  </Link>
                </motion.div>
              </Timeline.Item>
            );
          })}
        </Timeline>
      </motion.div>

      {visibleCount < recentUpdates.length && (
        <motion.div variants={itemVariants} className={styles.loadMoreWrapper}>
          <button onClick={() => setVisibleCount((prev) => prev + 5)} className={styles.loadMoreBtn}>
            <Translate id="homepage.loadMore">載入更多</Translate>
          </button>
        </motion.div>
      )}
    </motion.section>
  );
}

// -- 4) Testimonials 區塊：交錯進入範例 --
function StaggeredTestimonials({ testimonialsData }) {
  return (
    <motion.section
      className={styles.sectionBox}
      variants={containerVariants}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.2 }}
    >
      <motion.div variants={itemVariants}>
        <h2 className={styles.sectionTitle}>
          <Translate id="homepage.testimonialsTitle">讀者回饋</Translate>
        </h2>
      </motion.div>

      <motion.div className={styles.testimonialsWrapper} variants={containerVariants}>
        {testimonialsData.map((testi, i) => (
          <motion.div
            key={i}
            className={styles.testimonialCard}
            variants={itemVariants}
          >
            <img src={testi.avatar} alt={testi.name} className={styles.testimonialAvatar} />
            <div className={styles.testimonialContent}>
              <p className={styles.testimonialFeedback}>"{testi.feedback}"</p>
              <p className={styles.testimonialAuthor}>— {testi.name}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </motion.section>
  );
}

// -------------------------------------------------------------------------

export default function Home() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [visibleCount, setVisibleCount] = useState(5);

  // 載入對應語系
  const currentProjects = featuredProjectsData[currentLocale] || featuredProjectsData['en'];
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

  // 監聽捲動，顯示「回到頂端」按鈕
  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

  // Timeline 轉路徑
  const convertMdLinkToRoute = (mdLink) => {
    return mdLink
      .replace(/^.\//, '/papers/')
      .replace(/\.md$/, '')
      .replace(/\/index$/, '')
      .replace(/\/(\d{4}-)/, '/');
  };

  return (
    <Layout title={`Hello from ${siteConfig.title}`} description="Description">
      {/* Hero 區塊 (保留原先) */}
      <HomepageHeader siteTitle={siteConfig.title} siteTagline={siteConfig.tagline} />

      <main className={styles.mainWrapper}>
        {/* Features 用原本 FadeIn 或繼續使用 itemVariants 皆可。 這裡保留 FadeIn 說明 */}
        <section className={styles.sectionBox}>
          <HomepageFeatures />
        </section>

        {/* 精選作品：交錯進入 */}
        <StaggeredFeaturedProjects projects={currentProjects} />

        {/* Timeline：交錯進入 */}
        <StaggeredTimeline
          recentUpdates={recentUpdates}
          visibleCount={visibleCount}
          setVisibleCount={setVisibleCount}
          convertMdLinkToRoute={convertMdLinkToRoute}
        />

        {/* DocAligner Demo：示範容器+子項目簡易交錯 */}
        <motion.section
          className={styles.sectionBox}
          variants={containerVariants}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <motion.div variants={itemVariants}>
            <h2 className={styles.sectionTitle}>{localeContent.title}</h2>
          </motion.div>

          {/* 依情況可對 <p> 也使用 itemVariants */}
          <motion.div variants={itemVariants} className={styles.demoDescription}>
            {localeContent.description.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </motion.div>

          <motion.div variants={itemVariants}>
            <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />
          </motion.div>
        </motion.section>

        {/* 讀者回饋：交錯進入 */}
        <StaggeredTestimonials testimonialsData={testimonialsData} />
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
