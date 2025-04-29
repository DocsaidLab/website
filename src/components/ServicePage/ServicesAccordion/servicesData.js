// src/components/ServicePage/servicesData.js

export const servicesData = {
  'zh-hant': [
    // =============================
    // 1. 顧問合作
    // =============================
    {
      brief: {
        title: '顧問合作',
        scenario: '想長期顧問支援，但暫無大型專案需求；以每週固定天數獲取穩定支援',
        deliverables: '每週固定天數顧問 / 技術支援、階段性成果與進度紀錄',
        timeline: '按週期彈性協作',
        note: '可遠端或定期會議，適合持續優化與多次小幅開發',
      },
      detail: {
        description: [
          '若你需要長期技術顧問支援，但尚未達到全職或大型專案規模，可採用「週工時切分」模式：',
        ],
        bullets: [
          {
            title: '適合情境',
            items: [
              '需要固定技術顧問，但專案規模暫不大',
              '每週想保留固定天數，用於規劃或功能迭代',
              '希望能彈性分配預算與開發量',
            ],
          },
          {
            title: '服務模式',
            items: [
              '每週 1～2 天（或協商天數）進行專案投入',
              '可遠端或定期會議協作，時程可彈性安排',
              '內容包含程式開發、架構諮詢、文件撰寫等',
            ],
          },
          {
            title: '交付與時程',
            items: [
              '以週為單位進行，持續累積成果',
              '每週提供進度紀錄或完成項目',
              '如有大型功能，可轉為其他專案模式',
            ],
          },
        ],
        extraNotes: [
          '此模式適合開發步調尚不明確、需求會持續變動，或想穩定累積技術深度的人。',
          '你能在每週獲得專業建議並靈活調整開發方向。',
        ],
      },
    },

    // =============================
    // 2. 技術論文導讀與應用策略建議
    // =============================
    {
      brief: {
        title: '技術論文導讀與應用策略建議',
        scenario: '有想研究的論文，需判斷應用價值並快速選型，或擬定技術路線',
        deliverables: '策略建議書 (PDF) + 線上討論',
        timeline: '約 1 ～ 3 天',
        note: '可簽 NDA；預留 1～2 天溝通時間',
      },
      detail: {
        description: [
          '若你發現了一篇值得關注的技術論文，不確定它是否適合投入開發或能否與你的應用場景契合，此服務能協助快速分析並提出應用策略：',
        ],
        bullets: [
          {
            title: '重點',
            items: [
              '萃取論文的核心技術與應用特色',
              '評估其與你需求的相容性與落地風險',
              '提供後續行動建議（測試、小型實驗等）',
            ],
          },
          {
            title: '適合需求',
            items: [
              '想判斷論文方法的開發價值，並選擇投入時機',
              '對比多篇模型並做更精準的選型',
              '用報告說服團隊、主管或投資人',
            ],
          },
        ],
        references: [
          {
            label: '範例參考',
            linkText: 'Face Anti-Spoofing 技術地圖',
            linkHref: 'https://docsaid.org/blog/fas-paper-roadmap',
          },
        ],
        extraNotes: [
          '我更注重「轉譯」成適用於你專案的決策與實作方針。',
        ],
        warnings: [
          '建議先提供論文與背景資訊；若論文屬未公開或機密資料，請事先說明。',
        ],
      },
    },

    // =============================
    // 3. AI 模型 API 化與部署服務
    // =============================
    {
      brief: {
        title: 'AI 模型 API 化與部署服務',
        scenario: '已有模型，需要封裝成 API，並穩定部署到本地或私有環境',
        deliverables: 'FastAPI + Docker Demo、部署文件，基本保固',
        timeline: '約 3 ～ 14 天',
        note: '部署完成後提供短期保固',
      },
      detail: {
        description: [
          '若你已擁有一個模型，但不知道如何封裝成 API，或需要在本地或私有環境中穩定運行，我能透過 FastAPI 與 Docker 技術為你的模型打造服務端點。',
        ],
        bullets: [
          {
            title: '內容範圍',
            items: [
              '使用 FastAPI + Uvicorn 進行 API 封裝',
              '提供部署腳本（支援 Docker / 本地 / Nginx）',
              '包含 Swagger 文件、錯誤處理與基本安全設定',
            ],
          },
          {
            title: '延伸項目',
            items: [
              'API 測試頁設計',
              '部署錄影教學',
              'HTTPS 配置',
            ],
          },
        ],
        warnings: [
          '僅針對「已有模型」的部署，不含模型訓練或微調。',
          '若涉多容器協同或特定平台，請事先溝通。',
        ],
        extraNotes: [
          '完成後提供簡易保固，若需功能擴充或長期維運可再討論。',
        ],
      },
    },

    // =============================
    // 4. 教學型網站建置服務
    // =============================
    {
      brief: {
        title: '教學型網站建置服務',
        scenario: '以 Docusaurus 建置技術教學網站、團隊文件或部落格',
        deliverables: '基礎網站框架 (首頁、Blog、文件模組)、部署說明',
        timeline: '約 3 ～ 5 天',
        note: '上線後提供基本保固；可再討論延伸功能',
      },
      detail: {
        description: [
          '若你想將技術筆記或開源文件整理成一個易於擴充、結構分明且有設計感的網站，我能使用 Docusaurus 帶給你一個輕量且功能豐富的站點。',
        ],
        bullets: [
          {
            title: '可用範圍',
            items: [
              '個人技術筆記或部落格',
              '開源專案文件 / README 展示頁',
              '團隊內部教學平台或知識庫',
            ],
          },
          {
            title: '服務內容',
            items: [
              '網站框架設定與模組安裝（Blog、多語系、Navbar、Sidebar 等）',
              '客製化首頁與主視覺設計',
              '自訂網域、SEO 設定與部署教學',
            ],
          },
        ],
        warnings: [
          '若需更多功能（線上課程付費、會員系統、客製化後台）或大量排版調整，需另行討論。',
        ],
        extraNotes: [
          '網站上線後提供基本保固；若有進一步維運或功能擴充需求，可再談合作。',
        ],
      },
    },

    // =============================
    // 5. 從零打造 AI 模型產品（進階專案）
    // =============================
    {
      brief: {
        title: '從零打造 AI 模型產品（進階專案）',
        scenario: '需從 0 開發 MVP，整合 AI 應用、模型訓練、部署、前後端串接',
        deliverables: 'MVP 系統 (模型、API、介面整合)、專案說明與操作指南',
        timeline: '約 1 ～ 2 個月起',
        note: '分階段里程碑交付，可簽 NDA / 合約',
      },
      detail: {
        description: [
          '若你已有明確 AI 應用構想，但缺乏技術團隊或原型，我可協助「從 0 到 1」打造 MVP 系統，提供模型選型與評估，並進行 API 封裝、前後端整合。',
        ],
        bullets: [
          {
            title: '適用場景',
            items: [
              '新創或團隊需要快速驗證 PoC 可行性',
              '已有場景與資料，想開發專屬 AI 功能（如 OCR、人臉辨識）',
              '需要可展示的 demo，向決策者證明價值',
            ],
          },
          {
            title: '服務範圍',
            items: [
              '需求釐清與資料設計規劃',
              '模型訓練流程與評估策略',
              'API 打包與部署（FastAPI / Docker / 私有環境）',
              '前後端串接（或提供可視化 demo）',
            ],
          },
        ],
        warnings: [
          '此服務屬進階定製專案，需明確應用方向與決策者參與，並階段性討論功能範疇。',
        ],
        extraNotes: [
          '可依需求分階段交付（Prototype、Beta、正式 MVP 等），保障交付與保密。',
        ],
      },
    },
  ], // end zh-hant

  // -----------------------------------------
  //                 EN
  // -----------------------------------------
  en: [
    // 1. Consulting Partnership
    {
      brief: {
        title: 'Consulting Partnership',
        scenario:
          'Seeking ongoing technical consulting without a full-scale project; allocate fixed days each week for stable support',
        deliverables: 'Weekly consulting / technical support, staged deliverables & progress logs',
        timeline: 'Flexible weekly collaboration',
        note: 'Remote or scheduled meetings, suitable for continuous improvements with small increments',
      },
      detail: {
        description: [
          'If you need long-term technical consulting but are not ready for a full-time or large project, consider a “weekly time-slice” model:',
        ],
        bullets: [
          {
            title: 'Suitable Scenarios',
            items: [
              'Need stable consulting support but project scale is still small',
              'Want to reserve fixed days each week for planning or feature iteration',
              'Prefer flexible ways to allocate development resources',
            ],
          },
          {
            title: 'Service Model',
            items: [
              'Dedicate 1–2 days per week (or agreed upon) for project involvement',
              'Remote or scheduled collaboration, flexible scheduling',
              'Includes coding, architecture consulting, documentation, etc.',
            ],
          },
          {
            title: 'Deliverables & Timeline',
            items: [
              'Operate on a weekly cycle, continuously accumulating progress',
              'Provide progress updates or completed items each week',
              'For larger features, we can switch to other project modes',
            ],
          },
        ],
        extraNotes: [
          'Ideal for teams with uncertain development pace or evolving requirements, aiming to steadily grow technical depth.',
          'You will receive professional advice weekly and can adjust direction dynamically.',
        ],
      },
    },

    // 2. Technical Paper Reading & Strategy
    {
      brief: {
        title: 'Technical Paper Insight & Strategy',
        scenario:
          'Have a paper to study? Need to assess its practical value, choose implementation paths, or plan a technical route',
        deliverables: 'Strategy report (PDF) + online discussion',
        timeline: 'Around 1–3 days',
        note: 'NDA possible; reserve at least 1–2 days for communication',
      },
      detail: {
        description: [
          'If you find a noteworthy technical paper but are unsure whether it suits your use case, this service provides a quick analysis and strategic suggestions:',
        ],
        bullets: [
          {
            title: 'Key Points',
            items: [
              'Extract the core techniques and application highlights from the paper',
              'Evaluate compatibility with your requirements and potential risks',
              'Offer next-step suggestions (testing, small-scale experiments, etc.)',
            ],
          },
          {
            title: 'Ideal For',
            items: [
              'Deciding whether a paper’s method is worth developing and when to invest',
              'Comparing multiple models for more precise selection',
              'Using a concise report to convince your team, executives, or investors',
            ],
          },
        ],
        references: [
          {
            label: 'Sample Reference',
            linkText: 'Face Anti-Spoofing Tech Roadmap',
            linkHref: 'https://docsaid.org/blog/fas-paper-roadmap',
          },
        ],
        extraNotes: [
          'Beyond simple “translation,” the focus is on turning the paper into actionable strategies for your project.',
        ],
        warnings: [
          'Recommended to provide the paper and background info in advance; if the paper is confidential, please mention beforehand.',
        ],
      },
    },

    // 3. AI Model API & Deployment
    {
      brief: {
        title: 'AI Model API & Deployment',
        scenario: 'Have a trained model, need to expose it as an API, deploy locally or in a private environment',
        deliverables: 'FastAPI + Docker demo, documentation, basic support',
        timeline: 'Approx. 3–14 days',
        note: 'Short-term support after deployment',
      },
      detail: {
        description: [
          'If you already have a trained model but are unsure how to wrap it as an API or run it stably on your own infrastructure, I use FastAPI & Docker to build a service endpoint.',
        ],
        bullets: [
          {
            title: 'Scope',
            items: [
              'Wrap the model with FastAPI + Uvicorn',
              'Provide deployment scripts (Docker / local / Nginx)',
              'Include Swagger docs, error handling, basic security settings',
            ],
          },
          {
            title: 'Extended Options',
            items: [
              'API test page design',
              'Deployment video tutorial',
              'HTTPS setup',
            ],
          },
        ],
        warnings: [
          'For existing models only—no training or fine-tuning included.',
          'If you require multi-container orchestration or specialized platforms, please discuss first.',
        ],
        extraNotes: [
          'Short-term support post-deployment. Further expansions or long-term maintenance can be discussed.',
        ],
      },
    },

    // 4. Educational Website with Docusaurus
    {
      brief: {
        title: 'Educational Website Setup',
        scenario: 'Use Docusaurus to build a documentation or blogging platform for technical content',
        deliverables: 'Basic website structure (home, blog, doc modules), deployment guides',
        timeline: 'Approx. 3–5 days',
        note: 'Basic support after launch; additional features can be discussed',
      },
      detail: {
        description: [
          'If you want to organize your technical notes or open-source docs into a well-structured, design-oriented site, I can help you build a lightweight yet feature-rich website using Docusaurus.',
        ],
        bullets: [
          {
            title: 'Use Cases',
            items: [
              'Personal tech blogs or note collections',
              'Open-source project documentation / README showcase',
              'Team internal knowledge base or learning portal',
            ],
          },
          {
            title: 'Service Outline',
            items: [
              'Framework setup and modules (Blog, i18n, Navbar, Sidebar, etc.)',
              'Customized homepage and visual design',
              'Custom domain, SEO setup, and deployment instructions',
            ],
          },
        ],
        warnings: [
          'For advanced features (paid courses, membership systems, custom backends), further discussion is required.',
        ],
        extraNotes: [
          'After launch, basic support is provided. For extended maintenance or feature expansions, let’s talk further.',
        ],
      },
    },

    // 5. Full AI Product Development (Advanced)
    {
      brief: {
        title: 'Full AI Product Development (Advanced)',
        scenario: 'Need to build an MVP from scratch, integrating AI, model training, deployment, frontend & backend',
        deliverables: 'MVP system (model, API, interface integration), project docs & user guide',
        timeline: 'Approx. 1–2 months+',
        note: 'Milestone-based delivery, NDA or contract possible',
      },
      detail: {
        description: [
          'If you have a clear AI application idea but lack a dedicated tech team or prototype, I can assist in building an MVP from the ground up, including model selection & evaluation, API packaging, and front/backend integration.',
        ],
        bullets: [
          {
            title: 'Ideal Scenarios',
            items: [
              'Startups or teams needing a quick PoC feasibility check',
              'Have data & use cases, want custom AI features (OCR, face recognition, etc.)',
              'Need a functional demo to convince decision-makers or investors',
            ],
          },
          {
            title: 'Service Scope',
            items: [
              'Requirement analysis and data design',
              'Model training process & evaluation strategy',
              'API packaging & deployment (FastAPI / Docker / private environment)',
              'Frontend-backend integration (or a visual demo interface)',
            ],
          },
        ],
        warnings: [
          'This is a tailored, advanced project; it requires clear objectives and iterative discussions about feature scope.',
        ],
        extraNotes: [
          'We can deliver in phases (Prototype, Beta, final MVP) with confidentiality and IP protection.',
        ],
      },
    },
  ], // end en

  // -----------------------------------------
  //                 JA
  // -----------------------------------------
  ja: [
    // 1. コンサルティングパートナーシップ
    {
      brief: {
        title: 'コンサルティングパートナーシップ',
        scenario:
          '大規模なプロジェクトではなく、定期的な技術支援を求めている。週単位で固定日を確保して安定したサポートを受けたい',
        deliverables: '毎週固定のコンサル / 技術支援、進捗ログや成果物',
        timeline: '週単位で柔軟に対応',
        note: 'リモートまたは定期ミーティング可能。小規模開発を継続的に改善するのに最適',
      },
      detail: {
        description: [
          'フルタイムや大規模プロジェクトまでは必要ないが、長期的な技術コンサルを望む場合、「週単位の時間切り分け」モデルを検討できます：',
        ],
        bullets: [
          {
            title: '想定シチュエーション',
            items: [
              '専任の技術アドバイザーが必要だが、まだ大規模ではない',
              '週ごとに固定日を設け、計画・機能追加を継続したい',
              '予算と開発リソースを柔軟に配分したい',
            ],
          },
          {
            title: 'サービスモデル',
            items: [
              '週1～2日（または相談）でプロジェクト支援',
              'リモートや定期ミーティングでの協力、スケジュールは柔軟',
              'プログラミング、アーキテクチャ相談、ドキュメント作成など含む',
            ],
          },
          {
            title: '納品と期間',
            items: [
              '週単位で進行し、継続的に成果を蓄積',
              '毎週、進捗レポートや完了項目を提示',
              '大きな機能要件がある場合は別のプロジェクト形態に移行可能',
            ],
          },
        ],
        extraNotes: [
          '開発ペースが不確定、要件が頻繁に変動するチームや、技術力を徐々に蓄えたい方に適しています。',
          '週ごとに専門的なアドバイスを得ながら柔軟に方向修正が可能です。',
        ],
      },
    },

    // 2. 技術論文リーディング＆戦略提案
    {
      brief: {
        title: '技術論文リーディング＆戦略提案',
        scenario:
          '気になる論文があるが、実用価値や導入可能性を判断したい、または最適なモデル選定や手法を知りたい',
        deliverables: '戦略レポート（PDF）+ オンラインディスカッション',
        timeline: '約1～3日',
        note: 'NDA対応可。1～2日ほどのコミュニケーション期間を確保してください',
      },
      detail: {
        description: [
          '注目している技術論文があり、それが本当に利用シーンに合うか不明な場合に、このサービスで短期間に分析し、実装戦略を提案します：',
        ],
        bullets: [
          {
            title: '主なポイント',
            items: [
              '論文のコア技術・応用要点を抽出',
              'ニーズとの適合度とリスクを評価',
              '次のアクション（小規模テスト、実験など）を提案',
            ],
          },
          {
            title: '対象となる方',
            items: [
              '論文手法が投資する価値があるかどうかを判断したい',
              '複数のモデルを比較検討したい',
              '要約レポートでチーム・上司・投資家を説得したい',
            ],
          },
        ],
        references: [
          {
            label: '参考例',
            linkText: 'Face Anti-Spoofing 技術マップ',
            linkHref: 'https://docsaid.org/blog/fas-paper-roadmap',
          },
        ],
        extraNotes: [
          '単なる論文「翻訳」ではなく、あなたのプロジェクトに活かせる「方針策定」を重視します。',
        ],
        warnings: [
          '論文と背景情報を事前に共有いただくとスムーズです。未公開や機密論文の場合はご相談ください。',
        ],
      },
    },

    // 3. AIモデルのAPI化とデプロイ
    {
      brief: {
        title: 'AIモデルのAPI化とデプロイ',
        scenario: '既にモデルはあるが、APIとして公開またはローカル・社内環境で安定稼働させたい',
        deliverables: 'FastAPI + Docker デモ、ドキュメント、基本的なサポート',
        timeline: '3～14日ほど',
        note: 'デプロイ後、短期的なサポートあり',
      },
      detail: {
        description: [
          '学習済みモデルはあるが、APIとしてどのように構築・運用すればいいかわからない場合、FastAPIとDockerを用いてサービスエンドポイントを作成します。',
        ],
        bullets: [
          {
            title: '対応範囲',
            items: [
              'FastAPI + Uvicorn を用いたモデルのAPI化',
              'Docker / ローカル / Nginx などのデプロイスクリプト提供',
              'Swaggerドキュメント、エラーハンドリング、基本的なセキュリティ設定',
            ],
          },
          {
            title: '拡張オプション',
            items: [
              'APIテストページ設計',
              'デプロイ方法の録画チュートリアル',
              'HTTPS設定',
            ],
          },
        ],
        warnings: [
          '既存モデルのデプロイのみを対象（学習やファインチューニングは含まず）',
          'マルチコンテナ構成や特定クラウド環境などは事前にご相談ください',
        ],
        extraNotes: [
          'デプロイ後は短期間のサポートを提供。長期の運用や機能拡張は別途ご相談ください。',
        ],
      },
    },

    // 4. 教学向けサイト構築（Docusaurus）
    {
      brief: {
        title: 'ドキュメント系サイト構築サービス',
        scenario: 'Docusaurusを使った技術ドキュメント、ブログやチーム内向け資料サイトの構築',
        deliverables: '基本的なサイトフレーム（ホーム、ブログ、ドキュメントモジュール）とデプロイ手順',
        timeline: '3～5日程度',
        note: 'リリース後の基本サポートあり。追加機能は別途要相談',
      },
      detail: {
        description: [
          '技術メモやオープンソース資料を整理し、拡張性とデザイン性を備えたサイトを作成したい場合、Docusaurusを用いて軽量かつ機能豊富なドキュメントサイトを構築します。',
        ],
        bullets: [
          {
            title: '想定ユースケース',
            items: [
              '個人の技術ブログやノート整理',
              'オープンソースプロジェクトのドキュメント/README',
              'チーム内ナレッジベースや教育用ポータル',
            ],
          },
          {
            title: 'サービス内容',
            items: [
              'サイトフレームの設定、ブログ、多言語対応、Navbar、Sidebarなど',
              'ホームページや主要ビジュアルのカスタマイズ',
              '独自ドメイン、SEO対応、デプロイ手順の提供',
            ],
          },
        ],
        warnings: [
          'オンラインコース課金、会員システム、独自バックエンドなど高度機能が必要な場合は追加検討が必要。',
        ],
        extraNotes: [
          'サイト公開後は基本的なサポートを行います。追加機能や長期メンテは別途ご相談ください。',
        ],
      },
    },

    // 5. ゼロからAIモデル製品を構築（上級案件）
    {
      brief: {
        title: 'ゼロからAIモデル製品開発（上級プロジェクト）',
        scenario: 'MVPを0から作り、AIモデルの学習、デプロイ、フロント・バックエンド統合まで行いたい',
        deliverables: 'MVPシステム（モデル、API、UI統合）、プロジェクトドキュメントと操作ガイド',
        timeline: '1～2ヶ月以上',
        note: 'マイルストーン単位で納品。NDAや契約にも対応可',
      },
      detail: {
        description: [
          '明確なAI活用アイデアがあるが、専任チームやプロトタイプがない場合、モデル選定・評価からAPI化、フロント／バックエンドの接続まで「0→1」でMVPを構築します。',
        ],
        bullets: [
          {
            title: '適した状況',
            items: [
              'スタートアップやチームでPoC実験を急ぎたい',
              'データやシーンはあるがOCRや顔認証など独自AI機能を実装したい',
              '投資家や上層部へのデモが必要',
            ],
          },
          {
            title: 'サービス範囲',
            items: [
              '要件定義とデータ設計支援',
              'モデル学習・評価戦略',
              'API化とデプロイ（FastAPI / Docker / プライベート環境）',
              'フロントエンドとの連携（またはデモUI）',
            ],
          },
        ],
        warnings: [
          '本サービスは上級のカスタムプロジェクトです。明確なゴール設定と機能範囲のすり合わせが必要です。',
        ],
        extraNotes: [
          'プロトタイプ、ベータ、正式MVPなど段階的に納品し、知的財産や機密を保護します。',
        ],
      },
    },
  ], // end ja
};
