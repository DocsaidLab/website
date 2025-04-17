// src/components/MultiCardsCTA/index.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Col, Row } from 'antd';
import React from 'react';
import useLayoutType from '../../hooks/useLayoutType';
import i18nContentData from './i18nContext';
import styles from './index.module.css';
import ServiceCard from './ServiceCard';
import SimpleCTA from './SimpleCTA';


export default function MultiCardsCTA(props) {

  const {
    showServiceCards = true,
    maxColumns = 3,  // 預設允許三欄，可在 docs 裡呼叫時設為 2
  } = props;

  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const data = i18nContentData[currentLocale] || i18nContentData['zh-hant'];

  // 依照 maxColumns 得到實際的 layoutType ('oneCard', 'twoCards', 'threeCards')
  const layoutType = useLayoutType({ maxColumns });
  const cardsData = data?.[layoutType] || [];

  return (
    <section className={styles.ctaSection}>
      <CoffeeIntro coffeeData={data.coffeeCTA} />
      {showServiceCards && (
        <ServiceCards
          cardsData={cardsData}
          layoutType={layoutType}
        />
      )}
      <Outro outroData={data.outroCTA} />
    </section>
  );
}

function CoffeeIntro({ coffeeData }) {
  if (!coffeeData) return null;
  return (
    <SimpleCTA
      variant="coffee"
      iconSrc={coffeeData.icon}
      title={coffeeData.title}
      subtitle={coffeeData.subtitle}
      buttonLink={coffeeData.buttonLink}
      buttonImg={coffeeData.buttonImg}
    />
  );
}

function ServiceCards({ cardsData, layoutType }) {
  if (!cardsData.length) {
    return <p className={styles.emptyState}>目前沒有可顯示的卡片資料</p>;
  }

  return (
    <Row
      className={styles.cardsSection}
      gutter={[16, 16]}
      align="stretch"
    >
      {cardsData.map((card, idx) => {
        const keyValue = card.id ? card.id : `card-${idx}`;

        // 依照 layoutType 決定 Col 的屬性
        let colProps = {};
        if (layoutType === 'oneCard') {
          // 只顯示一欄
          colProps = { xs: 24 };
        } else if (layoutType === 'twoCards') {
          // 最多兩欄: 576px 以上就會切兩欄
          colProps = { xs: 24, sm: 12 };
        } else if (layoutType === 'threeCards') {
          // 三欄: 在 lg(≥992px) 才會切 3 欄
          colProps = { xs: 24, sm: 12, lg: 8 };
        }

        return (
          <Col
            key={keyValue}
            {...colProps}
            style={{ display: 'flex' }}
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
    />
  );
}
