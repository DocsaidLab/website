// index.js
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { Timeline } from 'antd';
import { motion, useAnimation } from 'framer-motion';
import React, { useEffect, useRef, useState } from 'react';

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo/DocAlignerDemoWrapper';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomepageHeader from '@site/src/components/HomepageHeader';

import demoContent from '@site/src/data/demoContent';
import featuredProjectsData from '@site/src/data/featuredProjectsData';
import testimonialsData from '@site/src/data/testimonialsData';

import styles from './index.module.css';

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

// -- Testimonials 區塊：交錯進入 --
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

// -- 核心：有自動橫向捲動、Hover 停止、左右箭頭控制 --
export default function AutoScrollingProjects({ projects }) {
  // 卡片資料重複一次，形成「無縫銜接」
  const scrollingItems = [...projects, ...projects];

  // Framer Motion 控制 & 追蹤實際位移 x
  const controls = useAnimation();
  const xRef = useRef(0);

  // 追蹤整個 track 寬度 (2倍內容的總寬)
  const trackRef = useRef(null);
  const [trackWidth, setTrackWidth] = useState(0);

  // 可自行調整自動捲動速度：數值越大，捲動越慢
  const AUTO_SCROLL_DURATION = 40;

  // 幫助把 offset 限制在 [-half, 0) 之間
  // 例如：x = -2500 時，若 half=2000，則 -2500 % 2000 = -500；剛好在 [-2000, 0) 之間
  // 再進一步若為正，則再減掉 half，確保最終落在 [-half, 0)
  function clampOffset(offset, half) {
    let r = offset % half; // JS % 會有正負之分
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

    // 1) 先把 xRef.current 校正到 [-half, 0)
    xRef.current = clampOffset(xRef.current, half);

    // 2) 先用 controls.set() 讓畫面瞬間跳到校正後的位置
    controls.set({ x: xRef.current });

    // 3) 接著從該位置開始動畫，往左移動 half 的距離
    //    repeat: Infinity 表示無限次數，repeatType: 'loop' 會循環往同一方向跑
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
        <Translate id="homepage.featuredProjectsTitle">精選作品</Translate>
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
            xRef.current = latest.x; // 同步記錄當前 x
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
                  <Translate id="homepage.learnMore">了解更多 →</Translate>
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
  const [visibleCount, setVisibleCount] = useState(5);

  // 依照語系載入 featuredProjects
  const currentProjects = featuredProjectsData[currentLocale] || featuredProjectsData['en'];

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

        {/* 改為帶箭頭 & Hover 暫停的自動橫向捲動 */}
        <AutoScrollingProjects projects={currentProjects} />

        {/* Timeline: 交錯進入 */}
        <StaggeredTimeline
          recentUpdates={recentUpdates}
          visibleCount={visibleCount}
          setVisibleCount={setVisibleCount}
          convertMdLinkToRoute={convertMdLinkToRoute}
        />

        {/* DocAligner Demo：保留交錯進入 */}
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

          <motion.div variants={itemVariants} className={styles.demoDescription}>
            {localeContent.description.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </motion.div>

          <motion.div variants={itemVariants}>
            <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />
          </motion.div>
        </motion.section>

        {/* Testimonials: 交錯進入 */}
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
