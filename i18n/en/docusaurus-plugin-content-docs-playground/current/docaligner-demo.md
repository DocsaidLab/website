# DocAligner Demo

You can select a few images with documents from your file system to test this feature.

If you can't find images immediately, we can borrow a few from the MIDV-2020 dataset for now:

:::info
**Clicking on the images below will allow you to use them directly in the Demo.**

These images perform well since they are from the training dataset, which the model has already seen!

However, in real-world applications, you might encounter a wider range of scenarios. So we recommend testing with a variety of images to better understand the model's performance.

When selecting images, please note the following:

1. If the document's corners are outside the image, the model will not be able to find all four corners and will return an error message.
   - We have made efforts to allow the model to extrapolate to unknown areas, but it may still fail.
2. If multiple documents are present in the image, the model may randomly select four corners from the many available.

With these reminders, we wish you a pleasant experience!
:::

If you want to use it in your own program, you can refer to the inference program example we used:

```python title='python demo code'
from docaligner import DocAligner
import docsaidkit as D

model = DocAligner(model_cfg='fastvit_sa24')

# padding for find unknown corner points in the image
input_img = D.pad(input_img, 100)

polygon = model(
    img=input_img,
    do_center_crop=False,
    return_document_obj=False
)

# Remove padding
polygon -= 100

return polygon
```

:::tip
MIDV-2020 is an open-source dataset containing many document images, perfect for testing document analysis models.

If needed, you can download it here: [**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
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
TransformedTitle="Transformed Image"
TransformedWidthLabel="Output Width"
TransformedHeightLabel="Output Height"
TransformedButtonLabel="Download Transformed Image"
defaultImages={[
{ src: '/en/img/docalign-demo/000025.jpg', description: 'Text Interference' },
{ src: '/en/img/docalign-demo/000121.jpg', description: 'Partial Occlusion' },
{ src: '/en/img/docalign-demo/000139.jpg', description: 'Strong Reflection' },
{ src: '/en/img/docalign-demo/000169.jpg', description: 'Low Light Scene' },
{ src: '/en/img/docalign-demo/000175.jpg', description: 'Highly Skewed' },
]}
/>
