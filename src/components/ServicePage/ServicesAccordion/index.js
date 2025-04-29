import React from 'react';
import ServicesAccordionSection from './ServicesAccordionSection';
import styles from './index.module.css';

export default function ServicesAccordion() {
  return (
    <section className={styles.servicesSection}>
      <ServicesAccordionSection />
    </section>
  );
}