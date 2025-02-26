# DocAligner Demo

You can select a few images with documents from your file system to test this feature.

If you can't find images immediately, we can borrow a few from the MIDV-2020 dataset for now:

:::info
**Clicking on the images below will allow you to use them directly in the Demo.**

These images perform well since they are from the training dataset, which the model has already seen!

However, in real-world applications, you might encounter a wider range of scenarios. So we recommend testing with a variety of images to better understand the model's performance.

When using this webpage's functionality, please note the following:

1. If the document's corners are outside the image, the model will not be able to find all four corners and will return an error message.
   - We have made efforts to allow the model to extrapolate to unknown areas, but it may still fail.
2. If multiple documents are present in the image, the model may randomly select four corners from the many available.
3. Due to the limitations of webpage load capacity, we need to compress images, which may result in a decrease in image quality.
   - Without this compression, the browser would crash.
4. We download the OpenCV module asynchronously. If the final cropped image is empty, it is because the download has not completed.
   - OpenCV is large (about 8MB) and takes some time to download.
   - This feature is not essential, and you can ignore this part.

With these reminders, we wish you a pleasant experience!
:::

If you want to use it in your own program, you can refer to the inference program example we used:

```python title='python demo code'
from docaligner import DocAligner
from capybara import pad

model = DocAligner(model_cfg='fastvit_sa24')

# padding for find unknown corner points in the image
input_img = pad(input_img, 100)

polygon = model(
   img=input_img,
   do_center_crop=False
)

# Remove padding
polygon -= 100

return polygon
```

:::tip
MIDV-2020 is an open-source dataset containing many document images, perfect for testing document analysis models.

If needed, you can download it here: [**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo';
import demoContent from '@site/src/data/demoContent';

export function Demo() {
const currentLocale = 'en';
const localeContent = demoContent[currentLocale];
return <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />;
}

<Demo />
