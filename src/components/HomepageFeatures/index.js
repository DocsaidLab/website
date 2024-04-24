import Heading from '@theme/Heading';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Open Source Projects',
    Svg: require('@site/static/img/docsaid_draw_1.svg').default,
    description: (
      <>
        Explore our open-source projects and contribute to advancements in AI technology with our developer.
      </>
    ),
  },
  {
    title: 'Technical Documentation',
    Svg: require('@site/static/img/docsaid_draw_2.svg').default,
    description: (
      <>
        Access our in-depth guides and documentation designed to help developers effectively utilize our AI tools.
      </>
    ),
  },
  {
    title: 'Research Analysis',
    Svg: require('@site/static/img/docsaid_draw_3.svg').default,
    description: (
      <>
        Stay updated with the latest in AI through our concise, clear analyses of key research findings and trends.
      </>
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
