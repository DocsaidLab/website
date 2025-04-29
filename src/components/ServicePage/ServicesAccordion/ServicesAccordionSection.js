// src/components/ServicePage/ServicesAccordionSection.js

import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import React, { useState } from 'react';
import AccordionItem from './AccordionItem';
import styles from './index.module.css';
import { servicesData } from './servicesData';


export default function ServicesAccordionSection() {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleAccordion = (index) => {
    setActiveIndex(index === activeIndex ? null : index);
  };

  const {
      i18n: { currentLocale },
    } = useDocusaurusContext();

    const dataForLocale = servicesData[currentLocale] || servicesData['zh-hant'];

  // 若無資料可顯示
  if (!dataForLocale || dataForLocale.length === 0) {
    return (
      <div className={styles.warning}>
        <p>暫無服務資料，請稍後再試。</p>
      </div>
    );
  }

  return (
    <div className={styles.accordionSection}>
      {dataForLocale.map((service, idx) => {
        // 先把 service.detail 轉成 JSX
        const detailJSX = renderServiceDetail(service.detail);

        return (
          <AccordionItem
            key={`accordion-item-${idx}`}
            index={idx}
            activeIndex={activeIndex}
            toggleAccordion={toggleAccordion}
            service={service}
            detailJSX={detailJSX} // 傳給子元件
          />
        );
      })}
    </div>
  );
}

/** 依你的資料結構動態渲染 JSX */
function renderServiceDetail(detail) {
  if (!detail) return null;

  const { description, bullets, warnings, extraNotes, table } = detail;

  return (
    <div>
      {/* description: 陣列 -> 多段 <p> */}
      {Array.isArray(description) &&
        description.map((desc, i) => <p key={`desc-${i}`}>{desc}</p>)}

      {/* bullets: [{title, items}, ...] */}
      {Array.isArray(bullets) &&
        bullets.map((block, bidx) => (
          <div key={`bullets-${bidx}`} style={{ marginBottom: '1rem' }}>
            <strong>{block.title}</strong>
            <ul>
              {block.items.map((item, ii) => (
                <li key={`item-${ii}`}>{item}</li>
              ))}
            </ul>
          </div>
        ))}

      {/* warnings: 顯示在一個警示區 */}
      {Array.isArray(warnings) && warnings.length > 0 && (
        <div className={styles.warning}>
          {warnings.map((warn, widx) => (
            <p key={`warn-${widx}`}>{warn}</p>
          ))}
        </div>
      )}

      {/* extraNotes: 額外備註 */}
      {Array.isArray(extraNotes) && extraNotes.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          {extraNotes.map((note, nidx) => (
            <p key={`note-${nidx}`}>{note}</p>
          ))}
        </div>
      )}

      {/* 若你的 detail 可能還有 table 或其他欄位，可在這裡依序渲染 */}
      {table && (
        <table className={styles.customTable} style={{ marginTop: '1rem' }}>
          <thead>
            <tr>
              {table.headers.map((hdr, hidx) => (
                <th key={`hdr-${hidx}`}>{hdr}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, ridx) => (
              <tr key={`row-${ridx}`}>
                {row.map((cell, cidx) => (
                  <td key={`cell-${cidx}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
