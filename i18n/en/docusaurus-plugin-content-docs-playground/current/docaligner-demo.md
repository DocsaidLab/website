# DocAligner Demo

You can select a few images with documents from your file system to test this feature.

If you can't find images immediately, we can borrow a few from the MIDV-2020 dataset for now:

:::tip
MIDV-2020 is an open-source dataset containing many document images, perfect for testing document analysis models.

If needed, you can download it here: [**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

:::info
**Clicking on the images below will allow you to use them directly in the Demo.**

These images perform well since they are from the training dataset, which the model has already seen!

However, in real-world applications, you might encounter a wider range of scenarios. So we recommend testing with a variety of images to better understand the model's performance.
:::

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemoWrapper';

<DocAlignerDemoWrapper
titleStage1="Test Images"
titleStage2="Demo"
chooseFileLabel="Select File"
uploadButtonLabel="Upload and Predict"
downloadButtonLabel="Download Prediction Results"
clearButtonLabel="Clear Results"
processingMessage="Processing, please wait..."
errorMessage={{
    chooseFile: "Please select a file",
    invalidFileType: "Only JPG, PNG, Webp images are supported",
    networkError: "Network error, please try again later.",
    uploadError: "An error occurred, please try again later."
  }}
warningMessage={{
    noPolygon: "No four corners detected. The model might not recognize this document type.",
    imageTooLarge: "The image is too large and may cause the browser to crash."
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
defaultImages={[
{ src: '/en/img/docalign-demo/000025.jpg', description: 'Text Interference' },
{ src: '/en/img/docalign-demo/000121.jpg', description: 'Partial Occlusion' },
{ src: '/en/img/docalign-demo/000139.jpg', description: 'Strong Reflection' },
{ src: '/en/img/docalign-demo/000169.jpg', description: 'Low Light Scene' },
{ src: '/en/img/docalign-demo/000175.jpg', description: 'Highly Skewed' },
]}
/>
