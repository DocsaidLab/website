# DocAligner Demo

你可以從檔案系統中選幾張帶有文件的圖片來測試這個功能。

如果一時半刻間找不到圖片，我們也可以先跟 MIDV-2020 借幾張來用：

:::info
**直接點擊下方圖片，可以直接代入 Demo 圖片中使用。**

這些圖片的效果都不錯，因為這是訓練資料，模型都看過啦！

在實際應用中，可能會遇到更多不同的情況，我們建議你可以找一些不同的圖片來測試，比較能夠了解模型的效果。

使用本網頁的功能時，有幾個注意事項：

1. **文件的角點在圖片之外，模型無法找齊四個角點，則會回傳錯誤訊息。**
   - 我們有努力讓模型可以外推到未知區域，但還是可能會失敗。
2. **若影像中同時存在多個文件，模型可能會在眾多角點中，隨意找四個點。**
3. **受限於網頁負荷能力，我們必須對圖片壓縮，因此圖片品質會有所下降。**
   - 不這麼做的話，瀏覽器會直接崩潰。
4. **我們用非同步方式下載 OpenCV 模組，如果你最後的裁切畫面顯示為空，那是因為下載未完成。**
   - OpenCV 很肥（約 8MB），要等一陣子。
   - 這個功能不是必要的，你可以忽略這個部分。

以上提醒，祝你玩得愉快！
:::

如果你想要在自己的程式中使用，可以參考我們使用的推論程式範例：

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
MIDV-2020 是個開源資料集，裡面有許多文件圖片，可以用來測試文件分析的模型。

如果你有需要，可以從這裡下載：[**MIDV-2020 Download**](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
:::

import DocAlignerDemoWrapper from '@site/src/components/DocAlignerDemo';
import demoContent from '@site/src/data/demoContent';

export function Demo() {
const currentLocale = 'zh-hant';
const localeContent = demoContent[currentLocale];
return <DocAlignerDemoWrapper {...localeContent.docAlignerProps} />;
}

<Demo />
