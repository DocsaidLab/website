const i18nMessages = {

  'homepage.recentUpdatesTitle': {
    en: 'Recent Updates',
    'zh-hant': '論文筆記近期更新',
    ja: '論文ノート最近の更新'
  },
  'homepage.loadMore': {
    en: 'Load More',
    'zh-hant': '載入更多',
    ja: 'もっと読み込む'
  },
  'homepage.testimonialsTitle': {
    en: 'Testimonials',
    'zh-hant': '讀者回饋',
    ja: '読者の声'
  },
  'homepage.featuredProjectsTitle': {
    en: 'Featured Projects',
    'zh-hant': '精選作品',
    ja: '注目プロジェクト'
  },
  'homepage.learnMore': {
    en: 'Learn More →',
    'zh-hant': '了解更多 →',
    ja: 'もっと詳しく →'
  },
  'homepage.demoTitle': {
    en: 'Features',
    'zh-hant': '功能展示',
    ja: '機能展示'
  },
  'homepage.demoIntro': {
    en: 'Below are demos of two model modules: DocAligner for document alignment and MRZScanner for MRZ recognition.',
    'zh-hant': '以下展示兩個模型模組的運行示範：DocAligner（文件對齊）與 MRZScanner（MRZ 辨識）。',
    ja: '下記は2つのモデルモジュールのデモです：DocAligner（文書整列）と MRZScanner（MRZ認識）。'
  },
  'homepage.docAlignerDemoTitle': {
    en: 'DocAligner Demo',
    'zh-hant': 'DocAligner Demo',
    ja: 'DocAligner デモ'
  },
  'homepage.docAlignerDemoDesc': {
    en: 'Upload an image containing a document to detect key points and perform perspective correction.',
    'zh-hant': '上傳含有文件的圖片，偵測關鍵點並進行透視校正。',
    ja: '文書を含む画像をアップロードし、キーポイント検出と透視補正を行います。'
  },
  'homepage.mrzScannerDemoTitle': {
    en: 'MRZScanner Demo',
    'zh-hant': 'MRZScanner Demo',
    ja: 'MRZScanner デモ'
  },
  'homepage.mrzScannerDemoDesc': {
    en: 'Upload an image with MRZ (Machine Readable Zone) to detect and parse the text content.',
    'zh-hant': '上傳含 MRZ（機器可判讀區）的圖片，偵測並解析其中的文字內容。',
    ja: 'MRZ（機械可読ゾーン）を含む画像をアップロードし、テキストを検出・解析します。'
  },
  'homepage.followUs': {
    en: '👍 Follow our Facebook Page',
    'zh-hant': '👍 追蹤我們的粉絲專頁',
    ja: '👍 フェイスブックページをフォロー'
  },

  // ===== Consulting & Services (re-aligned to model-centric, small studio, collaboration) =====
  'homepage.consultingTitle': {
    en: '🤝 Model-Centric Consulting & Services',
    'zh-hant': '🤝 以模型為核心的顧問與技術服務',
    ja: '🤝 モデル中心のコンサル＆技術サービス'
  },
  'homepage.consultingIntro': {
    en: 'A small model-focused studio. We turn real needs into maintainable, deployable, and evolvable model modules, working embedded with your team. Frontend/backend are lightweight—only to evaluate, showcase, and integrate models.',
    'zh-hant': '這是一個以模型研發為核心的小型工作室。我們把真實需求轉成可維護、可部署、可演進的模型模組，並「內嵌式」與你方團隊協作。前後端僅作為輕量配套，用於評測、展示與接入。',
    ja: '小規模なモデル特化スタジオです。実ニーズを、保守・デプロイ・進化が可能なモデルモジュールに変換し、あなたのチームと内製協働します。フロント/バックエンドは評価・デモ・接続の軽量補助のみです。'
  },

  // Cards updated to match your three service lines
  'homepage.consultingCards': {
    en: [
      {
        title: '🧩 Module Dev & Maintenance',
        desc: 'Productize one model module and maintain it long-term with versioned data & benchmarks'
      },
      {
        title: '🗓️ Consulting',
        desc: '1–2 days/week embedded collaboration: experiment design, data governance, evaluation'
      },
      {
        title: '🚀 MVP from Zero',
        desc: 'Build a minimal, demonstrable product around the model: selection, API, lightweight UI'
      },
      {
        title: '⚡ Inference Optimization & Deployment',
        desc: 'ONNX/TensorRT, quantization, latency budgeting; SDK/REST, batch/stream, private deploy'
      }
    ],
    'zh-hant': [
      {
        title: '🧩 模組開發與維護',
        desc: '把一個模型模組產品化並長期維護，含資料版本化與 Benchmark 報表'
      },
      {
        title: '🗓️ 顧問合作',
        desc: '每週 1～2 天協作：實驗設計、資料治理建議、評測體系與報表'
      },
      {
        title: '🚀 MVP 原型',
        desc: '以模型為核心打造最小可展示產品：選型、API 封裝、輕量介面'
      },
      {
        title: '⚡ 推論最佳化與部署',
        desc: 'ONNX/TensorRT、量化、延遲預算；SDK/REST、批次/串流、私有環境部署'
      }
    ],
    ja: [
      {
        title: '🧩 モジュール開発と保守',
        desc: '1つのモデルモジュールを製品化し、データ版管理とベンチマークで長期保守'
      },
      {
        title: '🗓️ コンサル',
        desc: '週1〜2日で協働：実験設計、データガバナンス助言、評価とレポート'
      },
      {
        title: '🚀 MVP',
        desc: 'モデル中心の最小デモ製品：選定、API化、軽量UIで価値を可視化'
      },
      {
        title: '⚡ 推論最適化とデプロイ',
        desc: 'ONNX/TensorRT、量子化、レイテンシ管理；SDK/REST、バッチ/ストリーム、プライベート環境'
      }
    ]
  },

  'homepage.consultingNoticeTitle': {
    en: '⚠️ Important Notes',
    'zh-hant': '⚠️ 注意事項',
    ja: '⚠️ 注意事項'
  },
  // Replace coffee/USD with TWD + retainer info, model-first scope, international scheduling
  'homepage.consultingNoticeList': {
    en: [
      '🧠 Model-first scope: FE/BE are lightweight for evaluation/showcase/integration only',
      '🗓️ Ongoing work often uses a monthly retainer; final quotes follow scoping',
      '💵 Billing in TWD (Taiwan).',
      '🌍 Cross-timezone/international work is welcome—please schedule in advance',
      '❌ We don’t offer LLM self-training (consultation and system evaluation are OK)',
      '📜 NDA supported; change logs and rollback strategy provided'
    ],
    'zh-hant': [
      '🧠 以模型為核心：前/後端為評測、展示與接入的輕量配套',
      '🗓️ 長期合作多採月保（Retainer）；實際報價以需求釐清為準',
      '💵 以新台幣（TWD）計價',
      '🌍 歡迎跨時區/海外合作，請先預約以便排程',
      '❌ 不提供 LLM 自訓（可提供諮詢與系統評估）',
      '📜 可簽 NDA，提供變更日誌與回滾策略'
    ],
    ja: [
      '🧠 モデル中心：FE/BEは評価・デモ・接続の軽量補助',
      '🗓️ 継続案件は月額リテイナーが一般的。見積は要件整理後に提示',
      '💵 請求は台湾元（TWD）。',
      '🌍 時差のある海外協業歓迎。事前にスケジュール調整をお願いします',
      '❌ LLMの独自学習は非対応（相談・評価は可）',
      '📜 NDA対応。変更履歴とロールバック戦略を提供'
    ]
  },

  'homepage.cooperationFormTitle': {
    en: 'Cooperation Form',
    'zh-hant': '合作需求表單',
    ja: '協力依頼フォーム'
  },
  'homepage.consultingMoreInfo': {
    en: 'For full service details, deliverables, and budget ranges, please visit:',
    'zh-hant': '完整服務內容、交付項目與預算區間，請參考：👉',
    ja: 'サービス内容、納品物、予算レンジの詳細は以下をご覧ください：'
  },
  'homepage.consultingMoreInfoLinkText': {
    en: 'Full Service Overview',
    'zh-hant': '技術服務',
    ja: '技術サービス'
  },
  'homepage.consultingMoreInfoLinkUrl': {
    en: 'https://docsaid.org/en/services/',
    'zh-hant': 'https://docsaid.org/services/',
    ja: 'https://docsaid.org/ja/services/'
  },
};

export default i18nMessages;
