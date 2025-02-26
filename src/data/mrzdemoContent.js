const mrzdemoContent = {
    'zh-hant': {
        title: 'MRZScanner Demo',
        description: `你可以從檔案系統中選幾張包含 MRZ（機器可判讀區域）的圖片進行測試。\n也可以使用我們提供的測試圖片：`,
        mrzScannerProps: {
            titleStage1: "測試圖片",
            titleStage2: "模型展示",
            chooseFileLabel: "選擇檔案",
            uploadButtonLabel: "上傳並預測",
            downloadButtonLabel: "下載結果",
            clearButtonLabel: "清除",
            processingMessage: "正在處理，請稍候...",
            errorMessage: {
                chooseFile: "請先選擇一個檔案",
                invalidFileType: "只支援 JPG、PNG、Webp",
                networkError: "網路錯誤，請稍後重試",
                uploadError: "上傳發生錯誤，請稍後再試"
            },
            warningMessage: {
                noPolygon: "未偵測到多邊形，模型可能無法辨識此圖像。",
                imageTooLarge: "圖片太大，可能影響瀏覽器效能。"
            },
            imageInfoTitle: "圖像資訊",
            inferenceInfoTitle: "推論資訊",
            polygonInfoTitle: "偵測結果 (doc_polygon / mrz_polygon)",
            mrzTextsTitle: "MRZ 文字",
            inferenceTimeLabel: "推論時間",
            timestampLabel: "時間戳",
            fileNameLabel: "檔案名稱",
            fileSizeLabel: "檔案大小",
            fileTypeLabel: "檔案類型",
            imageSizeLabel: "圖像尺寸",
            mrzOptionsTitle: "MRZ Scanner 參數",
            doDocAlignLabel: "文件校正",
            doCenterCropLabel: "中心裁切",
            doPostprocessLabel: "後處理",
            defaultImages: [
                { src: '/img/mrz-demo/000055.jpg', description: '光線昏暗' },
                { src: '/img/mrz-demo/000151.jpg', description: '辦公桌面' },
                { src: '/img/mrz-demo/000235.jpg', description: '雜亂野外' },
                { src: '/img/mrz-demo/000301.jpg', description: '文字干擾' },
                { src: '/img/mrz-demo/000155.jpg', description: '高度歪斜' },
            ]
        }
    },

    'en': {
        title: 'MRZScanner Demo',
        description: `You can select a few images containing an MRZ (Machine Readable Zone) from your file system for testing.\nYou can also use the test images we provide:`,
        mrzScannerProps: {
            titleStage1: "Test Images",
            titleStage2: "Demo",
            chooseFileLabel: "Select File",
            uploadButtonLabel: "Upload and Predict",
            downloadButtonLabel: "Download Results",
            clearButtonLabel: "Clear Results",
            processingMessage: "Processing, please wait...",
            errorMessage: {
                chooseFile: "Please select a file first",
                invalidFileType: "Only JPG, PNG, and Webp are supported",
                networkError: "Network error, please try again later",
                uploadError: "An upload error occurred, please try again later"
            },
            warningMessage: {
                noPolygon: "No polygon detected. The model may not recognize this image.",
                imageTooLarge: "Image is too large, which may affect browser performance."
            },
            imageInfoTitle: "Image Info",
            inferenceInfoTitle: "Inference Info",
            polygonInfoTitle: "Detection Results (doc_polygon / mrz_polygon)",
            mrzTextsTitle: "MRZ Text",
            inferenceTimeLabel: "Inference Time",
            timestampLabel: "Timestamp",
            fileNameLabel: "File Name",
            fileSizeLabel: "File Size",
            fileTypeLabel: "File Type",
            imageSizeLabel: "Image Size",
            mrzOptionsTitle: "MRZ Scanner Parameters",
            doDocAlignLabel: "Document Alignment",
            doCenterCropLabel: "Center Crop",
            doPostprocessLabel: "Post-Processing",
            defaultImages: [
                { src: '/en/img/mrz-demo/000055.jpg', description: 'Dim Lighting' },
                { src: '/en/img/mrz-demo/000151.jpg', description: 'Office Desk' },
                { src: '/en/img/mrz-demo/000235.jpg', description: 'Outdoors' },
                { src: '/en/img/mrz-demo/000301.jpg', description: 'Interference' },
                { src: '/en/img/mrz-demo/000155.jpg', description: 'Highly Skewed' },
            ]
        }
    },

    'ja': {
        title: 'MRZ スキャナー デモ',
        description: `ファイルシステムから MRZ（機械可読領域）を含む画像を数枚選択してテストできます。\n提供されているテスト画像を使用することも可能です:`,
        mrzScannerProps: {
            titleStage1: "テスト画像",
            titleStage2: "モデル展示",
            chooseFileLabel: "ファイルを選択",
            uploadButtonLabel: "アップロードして予測",
            downloadButtonLabel: "予測結果をダウンロード",
            clearButtonLabel: "結果をクリア",
            processingMessage: "処理中です。しばらくお待ちください...",
            errorMessage: {
                chooseFile: "ファイルを先に選択してください",
                invalidFileType: "JPG、PNG、Webp のみ対応しています",
                networkError: "ネットワークエラーです。後でもう一度お試しください",
                uploadError: "アップロード中にエラーが発生しました。後でもう一度お試しください"
            },
            warningMessage: {
                noPolygon: "多角形が検出されませんでした。この画像をモデルが認識できない可能性があります。",
                imageTooLarge: "画像が大きすぎるため、ブラウザのパフォーマンスに影響する可能性があります。"
            },
            imageInfoTitle: "画像情報",
            inferenceInfoTitle: "推論情報",
            polygonInfoTitle: "検出結果 (doc_polygon / mrz_polygon)",
            mrzTextsTitle: "MRZ テキスト",
            inferenceTimeLabel: "推論時間",
            timestampLabel: "タイムスタンプ",
            fileNameLabel: "ファイル名",
            fileSizeLabel: "ファイルサイズ",
            fileTypeLabel: "ファイルタイプ",
            imageSizeLabel: "画像サイズ",
            mrzOptionsTitle: "MRZ スキャナーパラメータ",
            doDocAlignLabel: "傾き補正",
            doCenterCropLabel: "中心切",
            doPostprocessLabel: "後処理",
            defaultImages: [
                { src: '/ja/img/mrz-demo/000055.jpg', description: '暗い照明' },
                { src: '/ja/img/mrz-demo/000151.jpg', description: 'オフィスデスク' },
                { src: '/ja/img/mrz-demo/000235.jpg', description: '雑然とした屋外' },
                { src: '/ja/img/mrz-demo/000301.jpg', description: '文字の干渉' },
                { src: '/ja/img/mrz-demo/000155.jpg', description: 'ひどく歪んだ' },
            ]
        }
    }
};

export default mrzdemoContent;
