# DocAligner Demo

你可以從檔案系統中選幾張帶有文件的圖片來測試這個功能。

如果一時半刻間找不到圖片，我們也可以先跟 MIDV-2020 借幾張來用：

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemoWrapper';

<DocAlignerDemoWrapper
chooseFileLabel="選擇檔案"
uploadButtonLabel="上傳並預測"
downloadButtonLabel="下載預測結果"
processingMessage="正在處理，請稍候..."
errorMessage={{
    chooseFile: "請選擇一個檔案",
    invalidFileType: "僅支援 JPG、PNG、Webp 格式的圖片",
    networkError: "網路錯誤，請稍後再試。",
    uploadError: "發生錯誤，請稍後再試。"
  }}
warningMessage={{
    noPolygon: "沒有檢測到多邊形，模型不認識這種文件類型。",
    imageTooLarge: "圖片太大，可能會導致瀏覽器故障。"
  }}
imageInfoTitle="圖像資訊"
inferenceInfoTitle="模型推論資訊"
polygonInfoTitle="偵測結果"
inferenceTimeLabel="推論時間"
timestampLabel="時間戳"
fileNameLabel="檔案名稱"
fileSizeLabel="檔案大小"
fileTypeLabel="檔案類型"
imageSizeLabel="圖像尺寸"
defaultImages={[
{ src: '/img/docalign-demo/000025.jpg', description: '文字干擾' },
{ src: '/img/docalign-demo/000139.jpg', description: '強烈反光' },
{ src: '/img/docalign-demo/000175.jpg', description: '高度歪斜' }
]}
/>
