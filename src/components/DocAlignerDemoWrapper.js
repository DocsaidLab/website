import React, { useState } from 'react';
import DocAlignerDemo from './DocAlignerDemo';


const DocAlignerDemoWrapper = ({
  titleStage1,
  titleStage2,
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
  inferenceTimeLabel,
  timestampLabel,
  fileNameLabel,
  fileSizeLabel,
  fileTypeLabel,
  imageSizeLabel,
  defaultImages
}) => {
  const [selectedImage, setSelectedImage] = useState(null);

  // 處理圖片點擊事件，將圖片的路徑存入 state
  const handleImageClick = (imageSrc) => {
    setSelectedImage(imageSrc);
  };

  return (
    <div>
      <h2>{titleStage1}</h2>
      <div className="image-grid">
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

      {/* 傳遞 externalImage */}
      <h2>{titleStage2}</h2>
      <DocAlignerDemo
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
        inferenceTimeLabel={inferenceTimeLabel}
        timestampLabel={timestampLabel}
        fileNameLabel={fileNameLabel}
        fileSizeLabel={fileSizeLabel}
        fileTypeLabel={fileTypeLabel}
        imageSizeLabel={imageSizeLabel}
        externalImage={selectedImage}
      />
    </div>
  );
};

export default DocAlignerDemoWrapper;
