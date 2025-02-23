import { motion } from 'framer-motion';
import React from 'react';
import { useInView } from 'react-intersection-observer';
import styles from './styles.module.css';

export default function FadeInSection({ children, className }) {
  const { ref, inView } = useInView({
    threshold: 0.2,
    triggerOnce: true,
  });

  const variants = {
    hidden: { opacity: 0, y: 25 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: 'easeOut' },
    },
  };

  return (
    <motion.section
      ref={ref}
      className={`${styles.fadeInWrapper} ${className || ''}`}
      initial="hidden"
      animate={inView ? 'visible' : 'hidden'}
      variants={variants}
    >
      {children}
    </motion.section>
  );
}
