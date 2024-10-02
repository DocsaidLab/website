# DocAligner Demo

import DocAlignerDemo from '@site/src/components/DocAlignerDemo';

<DocAlignerDemo
chooseFileLabel="Choose File"
uploadButtonLabel="Upload and Predict"
downloadButtonLabel="Download Results"
processingMessage="Processing, please wait..."
errorMessage={{
    chooseFile: "Please choose a file",
    invalidFileType: "Only JPG, PNG, and Webp image formats are supported",
    networkError: "Network error, please try again later.",
    uploadError: "An error occurred, please try again later."
  }}
warningMessage={{
    noPolygon: "No polygon detected, maybe model does not recognize this doc type.",
    imageTooLarge: "Image is too large and may cause browser crash."
  }}
imageInfoTitle="Image Information"
inferenceInfoTitle="Model Inference Information"
polygonInfoTitle="Detection Results"
inferenceTimeLabel="Inference Time"
timestampLabel="Timestamp"
fileNameLabel="File Name"
fileSizeLabel="File Size"
fileTypeLabel="File Type"
imageSizeLabel="Image Size"
/>
