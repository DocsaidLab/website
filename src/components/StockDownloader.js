import { DownloadOutlined } from "@ant-design/icons";
import { Button, Select, message } from "antd";
import React, { useState } from "react";

const { Option } = Select;

// 預設多國語系字典
const defaultTranslations = {
  zh: {
    selectJson: "JSON",
    selectCsv: "CSV",
    selectTxt: "TXT",
    downloadButton: "下載股票資料",
    downloadSuccess: (format) => `股票資料已下載 (${format.toUpperCase()})`,
    errorMessage: "無法取得股票資料，請檢查網路或 API 設定！",
    fileName: "TWSE_stocks",
    csvHeader: "代號,名稱,市場別,產業別,上市日期",
  },
  en: {
    selectJson: "JSON",
    selectCsv: "CSV",
    selectTxt: "TXT",
    downloadButton: "Download Stocks",
    downloadSuccess: (format) => `Stock data downloaded (${format.toUpperCase()})`,
    errorMessage: "Failed to fetch stock data. Please check network or API settings!",
    fileName: "TWSE_stocks",
    csvHeader: "StockID,StockName,Market,Industry,ListingDate",
  },
  ja: {
    selectJson: "JSON",
    selectCsv: "CSV",
    selectTxt: "TXT",
    downloadButton: "株式情報をダウンロード",
    downloadSuccess: (format) => `株式情報がダウンロードされました (${format.toUpperCase()})`,
    errorMessage: "株式情報を取得できません。ネットワークまたはAPI設定を確認してください！",
    fileName: "TWSE_stocks",
    csvHeader: "銘柄コード,銘柄名,市場,業種,上場日",
  },
};

const StockDownloader = ({
  lang = "zh",                // 預設使用中文
  customTranslations = {},   // 可額外自訂字典
}) => {
  // 合併預設字典 + 自訂字典
  const mergedTranslations = {
    ...defaultTranslations,
    ...customTranslations,
  };
  // 取用對應語系字串
  const t = mergedTranslations[lang] || defaultTranslations.zh;

  const [format, setFormat] = useState("json");
  const [loading, setLoading] = useState(false);

  const fetchStockData = async () => {
    try {
      setLoading(true);
      const response = await fetch("https://api.docsaid.org/stocks/infos");
      if (!response.ok) {
        throw new Error(`載入失敗: ${response.status} ${response.statusText}`);
      }
      const result = await response.json();
      if (!result.data || !Array.isArray(result.data)) {
        throw new Error("回傳資料格式錯誤！");
      }

      const stockData = {};
      for (const item of result.data) {
        const code = item.stock_id;
        stockData[code] = {
          名稱: item.stock_name,
          代號: item.stock_id,
          市場別: item.type,
          產業別: item.industry_category,
          上市日期: item.date,
        };
      }

      generateFile(stockData);
    } catch (error) {
      message.error(t.errorMessage);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const generateFile = (data) => {
    let fileContent = "";
    let blobType = "";
    let fileName = `${t.fileName}.${format}`;

    if (format === "json") {
      fileContent = JSON.stringify(data, null, 2);
      blobType = "application/json";
    } else if (format === "csv") {
      const lines = [t.csvHeader];
      for (const code in data) {
        const item = data[code];
        lines.push(
          `${item["代號"]},${item["名稱"]},${item["市場別"]},${item["產業別"]},${item["上市日期"]}`
        );
      }
      // 加入 UTF-8 BOM
      fileContent = "\uFEFF" + lines.join("\n");
      blobType = "text/csv;charset=utf-8";
    } else if (format === "txt") {
      const lines = [t.csvHeader];
      for (const code in data) {
        const item = data[code];
        lines.push(
          `${item["代號"]},${item["名稱"]},${item["市場別"]},${item["產業別"]},${item["上市日期"]}`
        );
      }
      fileContent = "\uFEFF" + lines.join("\n");
      blobType = "text/plain;charset=utf-8";
    }

    const blob = new Blob([fileContent], { type: blobType });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    message.success(t.downloadSuccess(format));
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <Select
        value={format}
        onChange={setFormat}
        style={{ width: 120, marginRight: 10 }}
      >
        <Option value="json">{t.selectJson}</Option>
        <Option value="csv">{t.selectCsv}</Option>
        <Option value="txt">{t.selectTxt}</Option>
      </Select>
      <Button
        type="primary"
        icon={<DownloadOutlined />}
        onClick={fetchStockData}
        loading={loading}
      >
        {t.downloadButton}
      </Button>
    </div>
  );
};

export default StockDownloader;
