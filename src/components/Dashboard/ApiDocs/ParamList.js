// src/components/Dashboard/ApiDocs/ParamList.jsx
import { List, Tag, Typography } from "antd";
import React from "react";
import styles from "./index.module.css";

const { Text } = Typography;

export default function ParamList({ data, text }) {
  return (
    <List
      className={styles.paramList} // 可在 CSS 中定義 margin-top 等間距
      itemLayout="vertical"
      dataSource={data}
      renderItem={(item) => {
        const { name, type, required, default: defaultVal, desc } = item;
        return (
          <List.Item key={name} className={styles.paramListItem}>
            <List.Item.Meta
              title={
                <span className={styles.paramTitle}>
                  <Text code>{name}</Text>{" "}
                  {type && <Tag color="blue">{type}</Tag>}
                  {required && <Tag color="red">{text.requiredLabel}</Tag>}
                  {defaultVal && defaultVal !== "-" && (
                    <Tag color="green">
                      {text.defaultLabel}: {defaultVal}
                    </Tag>
                  )}
                </span>
              }
              description={<span className={styles.paramDesc}>{desc}</span>}
            />
          </List.Item>
        );
      }}
    />
  );
}
