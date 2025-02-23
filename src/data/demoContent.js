const demoContent = {
    'zh-hant': {
      title: 'DocAligner Demo',
      description: `你可以從檔案系統中選幾張帶有文件的圖片來測試這個功能。\n也可以用我們提供的測試圖片：`,
      docAlignerProps: {
        titleStage1: "測試圖片",
        titleStage2: "模型展示",
        chooseFileLabel: "選擇檔案",
        uploadButtonLabel: "上傳並預測",
        downloadButtonLabel: "下載預測結果",
        clearButtonLabel: "清除結果",
        processingMessage: "正在處理，請稍候...",
        errorMessage: {
          chooseFile: "請選擇一個檔案",
          invalidFileType: "僅支援 JPG、PNG、Webp 格式的圖片",
          networkError: "網路錯誤，請稍後再試。",
          uploadError: "發生錯誤，請稍後再試。"
        },
        warningMessage: {
          noPolygon: "沒有檢測到四個角點，模型可能不認識這種文件類型。",
          imageTooLarge: "圖片太大，可能會導致瀏覽器故障。"
        },
        imageInfoTitle: "圖像資訊",
        inferenceInfoTitle: "模型推論資訊",
        polygonInfoTitle: "偵測結果",
        inferenceTimeLabel: "推論時間",
        timestampLabel: "時間戳",
        fileNameLabel: "檔案名稱",
        fileSizeLabel: "檔案大小",
        fileTypeLabel: "檔案類型",
        imageSizeLabel: "圖像尺寸",
        TransformedTitle: "攤平圖像",
        TransformedWidthLabel: "輸出寬度",
        TransformedHeightLabel: "輸出高度",
        TransformedButtonLabel: "下載攤平圖像",
        defaultImages: [
          { src: '/img/docalign-demo/000025.jpg', description: '文字干擾' },
          { src: '/img/docalign-demo/000121.jpg', description: '部分遮擋' },
          { src: '/img/docalign-demo/000139.jpg', description: '強烈反光' },
          { src: '/img/docalign-demo/000169.jpg', description: '昏暗場景' },
          { src: '/img/docalign-demo/000175.jpg', description: '高度歪斜' }
        ]
      }
    },
    'en': {
      title: 'DocAligner Demo',
      description: `You can test this feature by selecting a few images with files from the file system. \nYou can also use the test pictures we provide:`,
      docAlignerProps: {
        titleStage1: "Test Images",
        titleStage2: "Demo",
        chooseFileLabel: "Select File",
        uploadButtonLabel: "Upload and Predict",
        downloadButtonLabel: "Download Prediction Results",
        clearButtonLabel: "Clear Results",
        processingMessage: "Processing, please wait...",
        errorMessage: {
          chooseFile: "Please select a file",
          invalidFileType: "Only JPG, PNG, Webp images are supported",
          networkError: "Network error, please try again later.",
          uploadError: "An error occurred, please try again later."
        },
        warningMessage: {
          noPolygon: "No four corners detected. The model might not recognize this document type.",
          imageTooLarge: "The image is too large and may cause the browser to crash."
        },
        imageInfoTitle: "Image Information",
        inferenceInfoTitle: "Model Inference Information",
        polygonInfoTitle: "Detection Results",
        inferenceTimeLabel: "Inference Time",
        timestampLabel: "Timestamp",
        fileNameLabel: "File Name",
        fileSizeLabel: "File Size",
        fileTypeLabel: "File Type",
        imageSizeLabel: "Image Size",
        TransformedTitle: "Transformed Image",
        TransformedWidthLabel: "Output Width",
        TransformedHeightLabel: "Output Height",
        TransformedButtonLabel: "Download Transformed Image",
        defaultImages: [
          { src: '/en/img/docalign-demo/000025.jpg', description: 'Text Interference' },
          { src: '/en/img/docalign-demo/000121.jpg', description: 'Partial Occlusion' },
          { src: '/en/img/docalign-demo/000139.jpg', description: 'Strong Reflection' },
          { src: '/en/img/docalign-demo/000169.jpg', description: 'Low Light Scene' },
          { src: '/en/img/docalign-demo/000175.jpg', description: 'Highly Skewed' },
        ]
      }
    },
    'ja': {
      title: 'DocAligner デモ',
      description: `ファイル システムからファイルを含む写真をいくつか選択することで、この機能をテストできます。 \n弊社が提供するテスト画像を使用することもできます：`,
      docAlignerProps: {
        titleStage1: "テスト画像",
        titleStage2: "モデル展示",
        chooseFileLabel: "ファイルを選択",
        uploadButtonLabel: "アップロードして予測",
        downloadButtonLabel: "予測結果をダウンロード",
        clearButtonLabel: "結果をクリア",
        processingMessage: "処理中です。しばらくお待ちください...",
        errorMessage: {
          chooseFile: "ファイルを選択してください",
          invalidFileType: "JPG、PNG、Webp形式の画像のみ対応しています",
          networkError: "ネットワークエラーです。後でもう一度お試しください。",
          uploadError: "エラーが発生しました。後でもう一度お試しください。"
        },
        warningMessage: {
          noPolygon: "4つのコーナーが検出されませんでした。モデルはこの文書タイプを認識していない可能性があります。",
          imageTooLarge: "画像が大きすぎます。ブラウザがクラッシュする可能性があります。"
        },
        imageInfoTitle: "画像情報",
        inferenceInfoTitle: "モデル推論情報",
        polygonInfoTitle: "検出結果",
        inferenceTimeLabel: "推論時間",
        timestampLabel: "タイムスタンプ",
        fileNameLabel: "ファイル名",
        fileSizeLabel: "ファイルサイズ",
        fileTypeLabel: "ファイルタイプ",
        imageSizeLabel: "画像サイズ",
        TransformedTitle: "平坦化画像",
        TransformedWidthLabel: "出力幅",
        TransformedHeightLabel: "出力高さ",
        TransformedButtonLabel: "平坦化画像をダウンロード",
        defaultImages: [
          { src: '/ja/img/docalign-demo/000025.jpg', description: '文字干渉' },
          { src: '/ja/img/docalign-demo/000121.jpg', description: '部分的な隠れ' },
          { src: '/ja/img/docalign-demo/000139.jpg', description: '強い反射' },
          { src: '/ja/img/docalign-demo/000169.jpg', description: '暗いシーン' },
          { src: '/ja/img/docalign-demo/000175.jpg', description: '強い歪み' },
        ]
      }
    }
  };

export default demoContent;
