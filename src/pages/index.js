// index.js
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { Card, Col, Row, Timeline } from 'antd';
import { motion, useAnimation } from 'framer-motion';
import React, { useEffect, useRef, useState } from 'react';

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageHeader from '@site/src/components/HomepageHeader';
import MRZScannerDemoWrapper from '@site/src/components/MRZScannerDemo';
import CooperationForm from '@site/src/components/ServicePage/CooperationForm';

import demoContent from '@site/src/data/demoContent';
import featuredProjectsData from '@site/src/data/featuredProjectsData';
import i18nMessages from '@site/src/data/indexContent';
import mrzdemoContent from '@site/src/data/mrzdemoContent';
import testimonialsData from '@site/src/data/testimonialsData';

import styles from './index.module.css';

const getTranslation = (id, locale) => {
  return i18nMessages[id]?.[locale] || i18nMessages[id]?.['zh-hant'] || '';
};

// -- Timeline 交錯進入 variants --
const containerVariants = {
  hidden: {},
  show: {
    transition: {
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

// -- Timeline 區塊：交錯進入 --
function StaggeredTimeline({ recentUpdates, visibleCount, setVisibleCount, convertMdLinkToRoute }) {
  const { i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
  return (
    <motion.section
      className={styles.sectionBox}
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      <motion.div variants={itemVariants}>
        <h2 className={styles.sectionTitle}>
          {getTranslation('homepage.recentUpdatesTitle', currentLocale)}
        </h2>
      </motion.div>

      {/* 給 Timeline 加 key，確保每次載入更多都會重繪 */}
      <motion.div variants={containerVariants}>
        <Timeline mode="alternate" key={visibleCount}>
          {recentUpdates.slice(0, visibleCount).map((item, idx) => {
            const finalRoute = convertMdLinkToRoute(item.link);
            return (
              <Timeline.Item key={idx} label={item.date}>
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
            {getTranslation('homepage.loadMore', currentLocale)}
          </button>
        </motion.div>
      )}
    </motion.section>
  );
}

// -- Testimonials 區塊：交錯進入 --
function StaggeredTestimonials({ testimonialsData }) {
  const { i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
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
          {getTranslation('homepage.testimonialsTitle', currentLocale)}
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

// -- 核心：有自動橫向捲動、Hover 停止、左右箭頭控制 --
export function AutoScrollingProjects({ projects }) {
  const { i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
  // 卡片資料重複一次，形成「無縫銜接」
  const scrollingItems = [...projects, ...projects];

  // Framer Motion 控制 & 追蹤實際位移 x
  const controls = useAnimation();
  const xRef = useRef(0);

  // 追蹤整個 track 寬度 (2倍內容的總寬)
  const trackRef = useRef(null);
  const [trackWidth, setTrackWidth] = useState(0);

  // 可自行調整自動捲動速度：數值越大，捲動越慢
  const AUTO_SCROLL_DURATION = 80;

  // 幫助把 offset 限制在 [-half, 0) 之間
  function clampOffset(offset, half) {
    let r = offset % half;
    if (r > 0) {
      r = r - half;
    }
    return r;
  }

  // 圖片載入完後，量 trackWidth
  useEffect(() => {
    if (scrollingItems.length === 0) return;
    let loadedCount = 0;
    scrollingItems.forEach((proj) => {
      const img = new Image();
      img.src = proj.image;
      img.onload = () => {
        loadedCount++;
        if (loadedCount === scrollingItems.length && trackRef.current) {
          setTrackWidth(trackRef.current.scrollWidth);
        }
      };
    });
  }, [scrollingItems]);

  // trackWidth > 0後，啟動自動捲動
  useEffect(() => {
    if (trackWidth > 0) {
      startAutoScroll();
    }
  }, [trackWidth]);

  // 開始自動捲動 (無縫、無限)
  const startAutoScroll = () => {
    if (!trackWidth) return;
    const half = trackWidth / 2;

    // 1) 先把 xRef.current 校正到 [-half, 0) 之間
    xRef.current = clampOffset(xRef.current, half);

    // 2) 先用 controls.set() 讓畫面瞬間跳到校正後的位置
    controls.set({ x: xRef.current });

    // 3) 接著從該位置開始動畫，往左移動 half 的距離
    controls.start({
      x: [xRef.current, xRef.current - half],
      transition: {
        duration: AUTO_SCROLL_DURATION,
        ease: 'linear',
        repeat: Infinity,
        repeatType: 'loop',
      },
    });
  };

  // 停止自動捲動
  const stopAutoScroll = () => {
    controls.stop();
  };

  // 左右按鈕暫停 & 移動 300px
  const handleArrowLeft = () => {
    stopAutoScroll();
    controls.start({
      x: xRef.current + 300,
      transition: { duration: 0.6, ease: 'easeOut' },
    });
  };

  const handleArrowRight = () => {
    stopAutoScroll();
    controls.start({
      x: xRef.current - 300,
      transition: { duration: 0.6, ease: 'easeOut' },
    });
  };

  return (
    <section className={styles.sectionBox}>
      <h2 className={styles.sectionTitle}>
        {getTranslation('homepage.featuredProjectsTitle', currentLocale)}
      </h2>

      <div
        className={styles.projectsMarqueeOuter}
        onMouseLeave={startAutoScroll}
      >
        {/* 左箭頭 */}
        <button
          className={`${styles.arrowButton} ${styles.arrowLeft}`}
          onClick={handleArrowLeft}
        >
          ‹
        </button>

        {/* 內層 track：onUpdate 能即時拿到最新 x */}
        <motion.div
          className={styles.projectsMarqueeTrack}
          ref={trackRef}
          animate={controls}
          onUpdate={(latest) => {
            xRef.current = latest.x;
          }}
        >
          {scrollingItems.map((proj, idx) => (
            <div key={idx} className={styles.projectCardMarquee}>
              <img
                src={proj.image}
                alt={proj.title}
                className={styles.projectImageMarquee}
              />
              <div className={styles.projectContentMarquee}>
                <h3>{proj.title}</h3>
                <p>{proj.description}</p>
                <Link className={styles.projectLink} to={proj.link}>
                  {getTranslation('homepage.learnMore', currentLocale)}
                </Link>
              </div>
            </div>
          ))}
        </motion.div>

        {/* 右箭頭 */}
        <button
          className={`${styles.arrowButton} ${styles.arrowRight}`}
          onClick={handleArrowRight}
        >
          ›
        </button>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [visibleCount, setVisibleCount] = useState(10);

  // 依照語系載入 featuredProjects
  const currentProjects = featuredProjectsData[currentLocale] || featuredProjectsData['en'];
  const localeMrz = mrzdemoContent[currentLocale] || mrzdemoContent['en'];

  // 依照語系載入 recentUpdates
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

  // 回到頂端按鈕
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
      <HomepageHeader siteTitle={siteConfig.title} siteTagline={siteConfig.tagline} />

      <main className={styles.mainWrapper}>
        {/* Features */}
        <section className={styles.sectionBox}>
          <HomepageFeatures />
        </section>

        {/* 精選作品：自動橫向捲動 */}
        <AutoScrollingProjects projects={currentProjects} />

        {/* Timeline */}
        <section className={styles.sectionBox}>
          <Row gutter={[32, 32]}>
            {/* 左欄：StaggeredTimeline */}
            <Col xs={24} md={24}>
              <StaggeredTimeline
                recentUpdates={recentUpdates}
                visibleCount={visibleCount}
                setVisibleCount={setVisibleCount}
                convertMdLinkToRoute={convertMdLinkToRoute}
              />
            </Col>
          </Row>
        </section>

        {/* AI 顧問與技術服務：精簡後的關鍵內容 */}
        <motion.section
          className={styles.sectionBox}
          variants={containerVariants}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <Row gutter={[32, 32]}>
            {/* 左欄：重點說明與服務清單 */}
            <Col xs={24} md={currentLocale === 'zh-hant' ? 12 : 16}>
              <motion.div variants={itemVariants} className="consultingLeft">

                {/* 1) 取得標題 */}
                <h2 className="consultingTitle">
                  {getTranslation('homepage.consultingTitle', currentLocale)}
                </h2>

                {/* 2) 介紹段落 */}
                <p className="consultingIntro">
                  {getTranslation('homepage.consultingIntro', currentLocale)}
                </p>

                {/* 3) 服務項目卡片：用 .map() 根據語系動態產生 */}
                <Row gutter={[16, 16]} style={{ marginTop: '1rem' }}>
                  {getTranslation('homepage.consultingCards', currentLocale).map((card, idx) => (
                    <Col xs={24} sm={12} key={idx}>
                      <Card hoverable>
                        <h3>{card.title}</h3>
                        <p>{card.desc}</p>
                      </Card>
                    </Col>
                  ))}
                </Row>

                {/* 3.5) 更多資訊提示 */}
                <p style={{ fontSize: '0.95rem', color: '#555', marginTop: '1.25rem' }}>
                  📚{' '}
                  <strong>
                    {getTranslation('homepage.consultingMoreInfo', currentLocale)}
                  </strong>{' '}
                  <a
                    href={getTranslation('homepage.consultingMoreInfoLinkUrl', currentLocale)}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ textDecoration: 'underline' }}
                  >
                    <strong>{getTranslation('homepage.consultingMoreInfoLinkText', currentLocale)}</strong>
                  </a>
                </p>

                {/* 4) 注意事項 */}
                <div style={{ marginTop: '1rem' }}>
                  <div
                    style={{
                      border: '1px solid #ffe58f',
                      padding: '1rem',
                      borderRadius: 6,
                      background: '#fffbe6'
                    }}
                  >
                    <strong>
                      {getTranslation('homepage.consultingNoticeTitle', currentLocale)}
                    </strong>
                    <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem' }}>
                      {getTranslation('homepage.consultingNoticeList', currentLocale).map((note, idx) => (
                        <li key={idx}>{note}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            </Col>

            {/* 右欄：合作需求表單 */}
            <Col xs={24} md={currentLocale === 'zh-hant' ? 12 : 8}>
              <motion.div variants={itemVariants} className="consultingRight">
                <div className="cooperationFormContainer">
                  <h2 className="cooperationFormTitle">
                    {getTranslation('homepage.cooperationFormTitle', currentLocale)}
                  </h2>
                  <CooperationForm />
                </div>
              </motion.div>
            </Col>
          </Row>
        </motion.section>

        {/* DocAligner & MRZScanner Demo 區段 */}
        <motion.section
          className={styles.sectionBox}
          variants={containerVariants}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          <motion.div variants={itemVariants}>
            <h2 className={styles.sectionTitle}>
              {getTranslation('homepage.demoTitle', currentLocale)}
            </h2>
          </motion.div>

          <motion.div variants={itemVariants} className="demoSectionRow">
            <p>
              {getTranslation('homepage.demoIntro', currentLocale)}
            </p>
          </motion.div>

          <motion.div variants={itemVariants} className="demoSectionRow">
            <Row gutter={[24, 24]}>
              {/* 左欄：DocAligner */}
              <Col md={12} xs={24}>
                <Card className={styles.demoCard}>
                  <h3 className="demoCardTitle">
                    {getTranslation('homepage.docAlignerDemoTitle', currentLocale)}
                  </h3>
                  <p className="demoCardDesc">
                    {getTranslation('homepage.docAlignerDemoDesc', currentLocale)}
                  </p>
                  <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />
                </Card>
              </Col>

              {/* 右欄：MRZScanner */}
              <Col md={12} xs={24}>
                <Card className={styles.demoCard}>
                  <h3 className="demoCardTitle">
                    {getTranslation('homepage.mrzScannerDemoTitle', currentLocale)}
                  </h3>
                  <p className="demoCardDesc">
                    {getTranslation('homepage.mrzScannerDemoDesc', currentLocale)}
                  </p>
                  <MRZScannerDemoWrapper {...localeMrz.mrzScannerProps} />
                </Card>
              </Col>
            </Row>
          </motion.div>
        </motion.section>

        {/* Testimonials */}
        <StaggeredTestimonials testimonialsData={testimonialsData} />

      </main>

      {showBackToTop && (
        <button className={styles.backToTopBtn} onClick={scrollToTop}>
          ⬆
        </button>
      )}
    </Layout>
  );
}
