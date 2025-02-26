import React, { useState } from 'react';
import MRZScannerDemo from './MRZScannerDemo';
import styles from './styles.module.css';

const MRZScannerDemoWrapper = ({
  // 可以自訂各種文字 props，以下給一些示範預設值
  chooseFileLabel,
  uploadButtonLabel,
  downloadButtonLabel,
  clearButtonLabel,
  processingMessage,
  errorMessage,
  warningMessage,
  imageInfoTitle,
  inferenceInfoTitle,
  polygonInfoTitle,
  mrzTextsTitle,
  inferenceTimeLabel,
  timestampLabel,
  fileNameLabel,
  fileSizeLabel,
  fileTypeLabel,
  imageSizeLabel,
  mrzOptionsTitle,
  doDocAlignLabel,
  doCenterCropLabel,
  doPostprocessLabel,
  defaultImages,
  titleStage1,
  titleStage2,
}) => {
  // 紀錄使用者點選的預設圖片路徑，傳給 MRZScannerDemo
  const [selectedImage, setSelectedImage] = useState(null);

  // 處理圖片點擊事件，將圖片的路徑存入 state
  const handleImageClick = (imageSrc) => {
    setSelectedImage(imageSrc);
  };

  return (
    <div>
      {/* 範例圖片 (可自行刪除或調整) */}
      <h2>{titleStage1}</h2>
      <div className={styles.imageGrid}>
        {defaultImages.map((image, index) => (
          <div key={index}>
            <img
              src={image.src}
              alt={`example${index + 1}`}
              onClick={() => handleImageClick(image.src)}
            />
            <p>{image.description}</p>
          </div>
        ))}
      </div>

      {/* 主要的 MRZScannerDemo 區塊 */}
      <h2>{titleStage2}</h2>
      <MRZScannerDemo
        // 文字相關 props
        chooseFileLabel={chooseFileLabel}
        uploadButtonLabel={uploadButtonLabel}
        downloadButtonLabel={downloadButtonLabel}
        clearButtonLabel={clearButtonLabel}
        processingMessage={processingMessage}
        errorMessage={errorMessage}
        warningMessage={warningMessage}
        imageInfoTitle={imageInfoTitle}
        inferenceInfoTitle={inferenceInfoTitle}
        polygonInfoTitle={polygonInfoTitle}
        mrzTextsTitle={mrzTextsTitle}
        inferenceTimeLabel={inferenceTimeLabel}
        timestampLabel={timestampLabel}
        fileNameLabel={fileNameLabel}
        fileSizeLabel={fileSizeLabel}
        fileTypeLabel={fileTypeLabel}
        imageSizeLabel={imageSizeLabel}
        mrzOptionsTitle={mrzOptionsTitle}
        doDocAlignLabel={doDocAlignLabel}
        doCenterCropLabel={doCenterCropLabel}
        doPostprocessLabel={doPostprocessLabel}
        // 把使用者點選的範例圖傳進來
        externalImage={selectedImage}
      />
    </div>
  );
};

export default MRZScannerDemoWrapper;
