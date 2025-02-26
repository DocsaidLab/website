# MRZScanner Demo

You can test this feature by selecting a few images with MRZ from your file system.

However, unless you have a passport, it may be hard to find such images. (😀 😀 😀)

No worries! As usual, let's borrow a few images from MIDV-2020 to use!

:::info
**Click on the images below to load them directly into the Demo for testing.**

Since MIDV-2020 lacks MRZ region annotations, the model hasn't seen these images before.

In real-world applications, photos taken with a mobile phone may vary more. We recommend testing with different images to better understand the model's performance.

A few important things to note when using this web feature:

1. **If the MRZ region is incomplete or missing, the model will just randomly select a region to highlight.**
2. **If multiple MRZ regions are present in the image, the model will randomly pick four points.**
3. **Due to the limitations of the web page, we must compress the images, which may result in reduced quality.**
   - Without this, the browser will crash.

Finally, we have integrated the `DocAligner Demo` functionality in the backend. By enabling `do_doc_align`, you can seamlessly integrate it.

Enjoy testing, and have fun!
:::

If you'd like to call it in your program, here's a simple Python example:

```python title='python demo code'
from mrzscanner import MRZScanner, ModelType

model = MRZScanner(
   model_type=ModelType.two_stage,
   detection_cfg='20250222',
   recognition_cfg='20250221'
)

result = model(
    img=input_img,
    do_center_crop=False,   # Whether to perform center cropping first
    do_postprocess=True     # Whether to apply post-processing (fix MRZ characters)
)

return result
```

:::tip
MIDV-2020 is an open-source dataset containing many document images that can be used to test document analysis models.

If needed, you can download it here: [**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import MRZScannerDemoWrapper from '@site/src/components/MRZScannerDemo';
import mrzdemoContent from '@site/src/data/mrzdemoContent';

export function Demo() {
const currentLocale = 'en';
const localeContent = mrzdemoContent[currentLocale];
return <MRZScannerDemoWrapper {...localeContent.mrzScannerProps} />;
}

<Demo />
