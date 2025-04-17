// src/components/MultiCardsCTA/index.js
import { Col, Row } from 'antd';
import React from 'react';
import useLayoutType from '../../hooks/useLayoutType';
import i18nContentData from './i18nContext';
import styles from './index.module.css';
import ServiceCard from './ServiceCard';
import SimpleCTA from './SimpleCTA';

export default function MultiCardsCTA(props) {
  const { locale = 'zh-hant', showServiceCards = true } = props;
  const data = i18nContentData[locale] || i18nContentData['zh-hant'];

  const layoutType = useLayoutType();
  const cardsData = data?.[layoutType] || [];

  return (
    <section className={styles.ctaSection}>
      <CoffeeIntro coffeeData={data.coffeeCTA} />
      {showServiceCards && <ServiceCards cardsData={cardsData} />}
      <Outro outroData={data.outroCTA} />
    </section>
  );
}

function CoffeeIntro({ coffeeData }) {
  if (!coffeeData) return null;
  return (
    <SimpleCTA
      variant="coffee"
      title={coffeeData.title}
      subtitle={coffeeData.subtitle}
      buttonLink={coffeeData.buttonLink}
      buttonText={coffeeData.buttonText}
      buttonImg={coffeeData.buttonImg}
    />
  );
}

function ServiceCards({ cardsData }) {
  if (!cardsData.length) {
    return <p className={styles.emptyState}>目前沒有可顯示的卡片資料</p>;
  }

  // 使用 antd 的 Row / Col 做 RWD 排版
  return (
    <Row className={styles.cardsSection} gutter={[16, 16]}>
      {cardsData.map((card) => {
        const keyValue = card.title;
        return (
          <Col
            xs={{ span: 24 }}
            sm={{ span: 12 }}
            lg={{ span: 8 }}
            key={keyValue}
          >
            <ServiceCard cardData={card} />
          </Col>
        );
      })}
    </Row>
  );
}

function Outro({ outroData }) {
  if (!outroData) return null;
  return (
    <SimpleCTA
      variant="outro"
      title={outroData.title}
      subtitle={outroData.subtitle}
      buttonLink={outroData.buttonLink}
      buttonText={outroData.buttonText}
    />
  );
}
