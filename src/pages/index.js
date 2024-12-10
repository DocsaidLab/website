import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo/DocAlignerDemoWrapper';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import { Timeline } from 'antd';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import React from 'react';
import styles from './index.module.css';

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
            <div className="container">
                <Heading as="h1" className="hero__title">
                    {siteConfig.title}
                </Heading>
                <p className="hero__subtitle">
                <Translate id="homepage.tagline" description="Title for homepage tagline">{siteConfig.tagline}</Translate>
                </p>
                <div className={styles.buttons}>
                    <Link
                        className="button button--secondary button--lg"
                        to="/docs">
                        <Translate id="homepage.button1" description="Title for homepage butten1">開始探索</Translate>
                    </Link>
                </div>
            </div>
        </header>
    );
}

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
};

// 將多語系內容以物件定義
const demoContent = {
  'zh-hant': {
    title: 'DocAligner Demo',
    description: `你可以從檔案系統中選幾張帶有文件的圖片來測試這個功能。\n也可以用我們提供的測試圖片：`,
    docAlignerProps: {
      titleStage1: "測試圖片",
      titleStage2: "模型展示",
      chooseFileLabel: "選擇檔案",
      uploadButtonLabel: "上傳並預測",
      downloadButtonLabel: "下載預測結果",
      clearButtonLabel: "清除結果",
      processingMessage: "正在處理，請稍候...",
      errorMessage: {
        chooseFile: "請選擇一個檔案",
        invalidFileType: "僅支援 JPG、PNG、Webp 格式的圖片",
        networkError: "網路錯誤，請稍後再試。",
        uploadError: "發生錯誤，請稍後再試。"
      },
      warningMessage: {
        noPolygon: "沒有檢測到四個角點，模型可能不認識這種文件類型。",
        imageTooLarge: "圖片太大，可能會導致瀏覽器故障。"
      },
      imageInfoTitle: "圖像資訊",
      inferenceInfoTitle: "模型推論資訊",
      polygonInfoTitle: "偵測結果",
      inferenceTimeLabel: "推論時間",
      timestampLabel: "時間戳",
      fileNameLabel: "檔案名稱",
      fileSizeLabel: "檔案大小",
      fileTypeLabel: "檔案類型",
      imageSizeLabel: "圖像尺寸",
      TransformedTitle: "攤平圖像",
      TransformedWidthLabel: "輸出寬度",
      TransformedHeightLabel: "輸出高度",
      TransformedButtonLabel: "下載攤平圖像",
      defaultImages: [
        { src: '/img/docalign-demo/000025.jpg', description: '文字干擾' },
        { src: '/img/docalign-demo/000121.jpg', description: '部分遮擋' },
        { src: '/img/docalign-demo/000139.jpg', description: '強烈反光' },
        { src: '/img/docalign-demo/000169.jpg', description: '昏暗場景' },
        { src: '/img/docalign-demo/000175.jpg', description: '高度歪斜' }
      ]
    }
  },
  'en': {
    title: 'DocAligner Demo',
    description: `You can select a few images with documents from your file system to test this feature.
...（貼上英文版的介紹文字）`,
    docAlignerProps: {
      titleStage1: "Test Images",
      titleStage2: "Demo",
      chooseFileLabel: "Select File",
      uploadButtonLabel: "Upload and Predict",
      downloadButtonLabel: "Download Prediction Results",
      clearButtonLabel: "Clear Results",
      processingMessage: "Processing, please wait...",
      errorMessage: {
        chooseFile: "Please select a file",
        invalidFileType: "Only JPG, PNG, Webp images are supported",
        networkError: "Network error, please try again later.",
        uploadError: "An error occurred, please try again later."
      },
      warningMessage: {
        noPolygon: "No four corners detected. The model might not recognize this document type.",
        imageTooLarge: "The image is too large and may cause the browser to crash."
      },
      imageInfoTitle: "Image Information",
      inferenceInfoTitle: "Model Inference Information",
      polygonInfoTitle: "Detection Results",
      inferenceTimeLabel: "Inference Time",
      timestampLabel: "Timestamp",
      fileNameLabel: "File Name",
      fileSizeLabel: "File Size",
      fileTypeLabel: "File Type",
      imageSizeLabel: "Image Size",
      TransformedTitle: "Transformed Image",
      TransformedWidthLabel: "Output Width",
      TransformedHeightLabel: "Output Height",
      TransformedButtonLabel: "Download Transformed Image",
      defaultImages: [
        { src: '/en/img/docalign-demo/000025.jpg', description: 'Text Interference' },
        { src: '/en/img/docalign-demo/000121.jpg', description: 'Partial Occlusion' },
        { src: '/en/img/docalign-demo/000139.jpg', description: 'Strong Reflection' },
        { src: '/en/img/docalign-demo/000169.jpg', description: 'Low Light Scene' },
        { src: '/en/img/docalign-demo/000175.jpg', description: 'Highly Skewed' },
      ]
    }
  },
  'ja': {
    title: 'DocAligner デモ',
    description: `ファイルシステムからいくつかの文書画像を選んで、この機能をテストしてみてください。
...（貼上日文版的介紹文字）`,
    docAlignerProps: {
      titleStage1: "テスト画像",
      titleStage2: "モデル展示",
      chooseFileLabel: "ファイルを選択",
      uploadButtonLabel: "アップロードして予測",
      downloadButtonLabel: "予測結果をダウンロード",
      clearButtonLabel: "結果をクリア",
      processingMessage: "処理中です。しばらくお待ちください...",
      errorMessage: {
        chooseFile: "ファイルを選択してください",
        invalidFileType: "JPG、PNG、Webp形式の画像のみ対応しています",
        networkError: "ネットワークエラーです。後でもう一度お試しください。",
        uploadError: "エラーが発生しました。後でもう一度お試しください。"
      },
      warningMessage: {
        noPolygon: "4つのコーナーが検出されませんでした。モデルはこの文書タイプを認識していない可能性があります。",
        imageTooLarge: "画像が大きすぎます。ブラウザがクラッシュする可能性があります。"
      },
      imageInfoTitle: "画像情報",
      inferenceInfoTitle: "モデル推論情報",
      polygonInfoTitle: "検出結果",
      inferenceTimeLabel: "推論時間",
      timestampLabel: "タイムスタンプ",
      fileNameLabel: "ファイル名",
      fileSizeLabel: "ファイルサイズ",
      fileTypeLabel: "ファイルタイプ",
      imageSizeLabel: "画像サイズ",
      TransformedTitle: "平坦化画像",
      TransformedWidthLabel: "出力幅",
      TransformedHeightLabel: "出力高さ",
      TransformedButtonLabel: "平坦化画像をダウンロード",
      defaultImages: [
        { src: '/ja/img/docalign-demo/000025.jpg', description: '文字干渉' },
        { src: '/ja/img/docalign-demo/000121.jpg', description: '部分的な隠れ' },
        { src: '/ja/img/docalign-demo/000139.jpg', description: '強い反射' },
        { src: '/ja/img/docalign-demo/000169.jpg', description: '暗いシーン' },
        { src: '/ja/img/docalign-demo/000175.jpg', description: '強い歪み' },
      ]
    }
  }
};

export default function Home() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  // 根據語系載入對應 recent_updates_data.json
  let recentUpdates;
  if (currentLocale === 'zh-hant') {
    recentUpdates = require('@site/papers/recent_updates_data.json');
  } else if (currentLocale === 'en') {
    recentUpdates = require('@site/i18n/en/docusaurus-plugin-content-docs-papers/current/recent_updates_data.json');
  } else if (currentLocale === 'ja') {
    recentUpdates = require('@site/i18n/ja/docusaurus-plugin-content-docs-papers/current/recent_updates_data.json');
  } else {
    // 預設英文或任何一個預設語系
    recentUpdates = require('@site/papers/recent_updates_data.json');
  }

  const localeContent = demoContent[currentLocale] || demoContent['en'];

  const convertMdLinkToRoute = (mdLink) => {
    return mdLink
      .replace(/^.\//, '/papers/')
      .replace(/\.md$/, '')
      .replace(/\/index$/, '')
      .replace(/\/(\d{4}-)/, '/');
  };

    return (
        <Layout
            title={`Hello from ${siteConfig.title}`}
      description="Description"
    >
            <HomepageHeader />
            <main>
                <HomepageFeatures />
        <div className={styles.twoColumnLayout}>
          {/* 左欄：Timeline */}
          <div className={styles.leftColumn}>
            <section className={styles.timelineSection}>
              {/* 使用 <Translate> 包裝標題 */}
              <Heading as="h2" className={styles.timelineTitle}>
                <Translate id="homepage.recentUpdatesTitle">
                  論文筆記近期更新
                </Translate>
              </Heading>
              <motion.div
                variants={containerVariants}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.1 }}
              >
                <Timeline mode="alternate">
                  {recentUpdates.map((item, idx) => {
                    const finalRoute = convertMdLinkToRoute(item.link);
                    return (
                      <Timeline.Item key={idx} label={item.date}>
                        <motion.div variants={itemVariants} className={styles.timelineItemContent}>
                          <Link to={finalRoute}>{item.combinedTitle}</Link>
                        </motion.div>
                      </Timeline.Item>
                    );
                  })}
                </Timeline>
              </motion.div>
            </section>
          </div>

          {/* 右欄：DocAligner Demo */}
          <div className={styles.rightColumn}>
            <section className={styles.demoSection}>
              <Heading as="h2">{localeContent.title}</Heading>
              <div className={styles.demoDescription}>
                {localeContent.description.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
              <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />
            </section>
          </div>
        </div>
            </main>
        </Layout>
    );
}