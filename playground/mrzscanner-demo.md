# MRZScanner Demo

你可以從檔案系統中選幾張帶有 MRZ 的圖像來測試這個功能。

但一般來說，除非你有護照，不然應該找不到。(😀 😀 😀)

沒關係，我們還是老樣子，先跟 MIDV-2020 借幾張來用吧！

:::info
**直接點擊下方圖片，可以直接載入到 Demo 中測試。**

由於 MIDV-2020 缺乏對應 MRZ 的區域標註，所以這些資料模型都沒看過。

在實際應用中，用手機拍攝的照片可能會有更多情況，我們建議你可以找一些不同的圖片來測試，比較能夠了解模型的效果。

使用本網頁的功能時，有幾個注意事項：

1. **MRZ 的區域不完整或不存在，模型會隨便找個地方框給你。**
2. **若影像中同時存在多個 MRZ，模型會隨意找四個點。**
3. **受限於網頁負荷能力，我們必須對圖片壓縮，因此圖片品質會有所下降。**
   - 不這麼做的話，瀏覽器會直接崩潰。

最後，我們直接從後台串接了 `DocAligner Demo` 的功能，你只要啟用 `do_doc_align`，就可以無痛銜接。

以上提醒，祝你玩得愉快！
:::

如果想要在程式裡呼叫，這裡有個簡單的 Python 示範：

```python title='python demo code'
from mrzscanner import MRZScanner, ModelType

model = MRZScanner(
   model_type=ModelType.two_stage,
   detection_cfg='20250222',
   recognition_cfg='20250221'
)

result = model(
    img=input_img,
    do_center_crop=False,   # 是否先中心裁切
    do_postprocess=True     # 是否做後處理 (修正 MRZ 字元)
)

return result
```

:::tip
MIDV-2020 是個開源資料集，裡面有許多文件圖片，可以用來測試文件分析的模型。

如果你有需要，可以從這裡下載：[**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import MRZScannerDemoWrapper from '@site/src/components/MRZScannerDemo';
import mrzdemoContent from '@site/src/data/mrzdemoContent';

export function Demo() {
const currentLocale = 'zh-hant';
const localeContent = mrzdemoContent[currentLocale];
return <MRZScannerDemoWrapper {...localeContent.mrzScannerProps} />;
}

<Demo />
