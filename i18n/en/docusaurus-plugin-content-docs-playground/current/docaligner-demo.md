# DocAligner Demo

You can test this feature by selecting a few images with documents from the file system.

If you can't find any images for a while, you can borrow some from MIDV-2020:

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemoWrapper';

<DocAlignerDemoWrapper
chooseFileLabel="Choose File"
uploadButtonLabel="Upload and Predict"
downloadButtonLabel="Download Prediction"
processingMessage="Processing, please wait..."
errorMessage={{
    chooseFile: "Please choose a file",
    invalidFileType: "Only JPG, PNG, and Webp image formats are supported",
    networkError: "Network error, please try again later.",
    uploadError: "An error occurred, please try again later."
  }}
warningMessage={{
    noPolygon: "No polygon detected, the model does not recognize this document type.",
    imageTooLarge: "The image is too large and may cause the browser to crash."
  }}
imageInfoTitle="Image Information"
inferenceInfoTitle="Model Inference Information"
polygonInfoTitle="Detection Result"
inferenceTimeLabel="Inference Time"
timestampLabel="Timestamp"
fileNameLabel="File Name"
fileSizeLabel="File Size"
fileTypeLabel="File Type"
imageSizeLabel="Image Size"
defaultImages={[
{ src: '/en/img/docalign-demo/000025.jpg', description: 'Text interference' },
{ src: '/en/img/docalign-demo/000139.jpg', description: 'Strong reflection' },
{ src: '/en/img/docalign-demo/000175.jpg', description: 'Highly skewed' }
]}
/>
