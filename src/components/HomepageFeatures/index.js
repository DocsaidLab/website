import Translate from '@docusaurus/Translate';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: (
      <Translate id="homepage.feature.title1" description="Title for the open source projects feature">
        開源專案
      </Translate>
    ),
    Svg: require('@site/static/img/docsaid_draw_1.svg').default,
    description: (
      <Translate id="homepage.feature.description1" description="Description for the open source projects feature">
        探索我們的開源專案，與我們共同推動人工智慧技術的進步與創新，透過社群的力量共同成長。
      </Translate>
    ),
  },
  {
    title: (
      <Translate id="homepage.feature.title2" description="Title for the technical documentation feature">
        技術文件
      </Translate>
    ),
    Svg: require('@site/static/img/docsaid_draw_2.svg').default,
    description: (
      <Translate id="homepage.feature.description2" description="Description for the technical documentation feature">
        我們提供針對自家開源專案的詳細心得，幫助開發者理解我們的想法，提高開發過程的流暢性。
      </Translate>
    ),
  },
  {
    title: (
      <Translate id="homepage.feature.title3" description="Title for the paper notes feature">
        論文筆記
      </Translate>
    ),
    Svg: require('@site/static/img/docsaid_draw_3.svg').default,
    description: (
      <Translate id="homepage.feature.description3" description="Description for the paper notes feature">
        記錄我們在閱讀論文時的心得與見解，透過這些筆記，可以分享我們的思考過程與重點摘錄。
      </Translate>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
