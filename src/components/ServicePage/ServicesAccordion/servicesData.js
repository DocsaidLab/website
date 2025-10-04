// src/components/ServicePage/servicesData.js

export const servicesData = {
  'zh-hant': [
    // =============================
    // 1) 顧問合作（週工時切分, 協力不取代）
    // =============================
    {
      brief: {
        title: '顧問合作（週工時切分）',
        scenario: '需要穩定的模型研發/評估建議，但暫無大型專案；以每週固定天數取得支持，與你方團隊協作而非取代',
        deliverables: '每週顧問/技術支援、模型與資料決策建議、階段性成果與進度紀錄',
        timeline: '按週期彈性協作',
        note: '遠端或定期會議皆可；適合持續優化與多次小幅迭代',
      },
      detail: {
        description: [
          '當你希望持續累積模型能力、但節奏尚未到專案制時，建議採用「週工時切分」模式：我們以小型團隊內嵌協作，聚焦模型品質與可靠性，沿用你的工具與流程。',
        ],
        bullets: [
          {
            title: '適合情境',
            items: [
              '需要固定技術顧問，但專案規模暫不大',
              '每週保留固定天數，處理資料/模型/評測的排程',
              '希望彈性配置預算與開發量，逐步拉升模型表現',
            ],
          },
          {
            title: '服務模式',
            items: [
              '每週 1～2 天（或協商）投入；與你方工程/資料/產品團隊協作',
              '可遠端或定期會議；沿用你方既有 repo、CI、看板與溝通工具',
              '內容包含：模型調研/實驗設計、資料治理建議、評測體系與報表',
            ],
          },
          {
            title: '交付與時程',
            items: [
              '以週為單位滾動累積，維持穩定節奏',
              '提供每週進度紀錄與決策備忘、下週優先級',
              '若需求擴大，可平滑切換至專案模式',
            ],
          },
        ],
        extraNotes: [
          '重點在「協力補位」，不取代你方團隊的主導權。',
          '如需小型前/後端介面，僅作為模型展示與評測之配套。',
        ],
      },
    },

    // =============================
    // 2) 單一模型模組：開發與長期維護（模組制）
    // =============================
    {
      brief: {
        title: '單一模型模組：開發與長期維護（模組制）',
        scenario:
          '聚焦一個 AI 模型模組，長期迭代與維護（典型如 DocAligner：文件對齊/版面理解），我們負責把模型做穩、做準、可持續演進',
        deliverables:
          '訓練模組 + 推論模組 + Benchmark 前端（多模型/多資料比較），版本化與維運紀錄',
        timeline: '長期維運；以迭代週期交付',
        note: '新增能力（如在 DocAligner 基礎上擴充版面語義/鍵值抽取/ReID）視為新模組，另行估價',
      },
      detail: {
        description: [
          '我們將「單一模型模組」產品化與長期維護。以 DocAligner（文件對齊/版面理解）為例：從資料治理→訓練與評估→推論與部署→持續版本化，確保每次改版皆可量化、可回溯、可解釋。',
        ],
        bullets: [
          {
            title: '服務構成（核心模組）',
            items: [
              '訓練模組：資料規劃/版本、標註流程、增強策略、訓練與微調、實驗追蹤（mAP/F1/Latency/TPR@FPR）',
              '推論模組：SDK/REST、批次與串流、效能最佳化（ONNX/TensorRT/量化）、資源監控與延遲預算管理',
              'Benchmark 模組：Kaggle 式排行榜頁，PR 曲線/混淆矩陣/速度–精度曲面；多模型與多資料版本比較、報表匯出',
            ],
          },
          {
            title: '展示範例（以 DocAligner 為例）',
            items: [
              '多輪訓練/多資料版本的模型比較，對齊精度、關鍵點誤差與延遲統一呈現',
              '前端可切換模型版本，檢視在多份公版/匿名化文件的對齊品質與熱區可視化',
              '一鍵匯出審查報表，支援里程碑回顧',
            ],
          },
          {
            title: '協作與交付方式',
            items: [
              '以迭代週期協作：每一週期鎖定明確目標（資料更新/指標提升/效能優化）與交付物',
              '提供技術紀錄與變更日誌；維持可復現的環境/腳本',
              '可與既有標註/MLOps/CI-CD 流程整合；你方保有系統主導權',
            ],
          },
          {
            title: '模組邊界與擴充',
            items: [
              '新能力（版面語義分割、鍵值抽取、表格結構化、跨文件匹配）視為「新模組」',
              '硬體/平台切換（Jetson/Edge TPU/雲 GPU）如需顯著調整，將另估',
              '可簽 NDA；依你方規範執行資料治理與權限管控',
            ],
          },
        ],
        references: [
          {
            label: '經典專案：DocAligner（GitHub）',
            linkText: 'DocAligner 專案連結',
            linkHref: 'https://github.com/DocsaidLab/DocAligner',
          },
        ],
        warnings: [
          '本服務專注單一模型模組；大型平台建設或跨多系統整合不在本套餐範圍。',
          '第三方/商用模型授權費用不含在內。',
        ],
        extraNotes: [
          '可採 PoC → Pilot → Production 分階段路線；也可接手既有模型維護。',
          '交付包含基準資料集與固定頻率之 Benchmark 報告（週/雙週/月，依協議）。',
        ],
      },
    },

    // =============================
    // 3) MVP 原型：從 0 打造可展示的模型產品（進階專案）
    // =============================
    {
      brief: {
        title: 'MVP 原型：從 0 打造可展示的模型產品（進階專案）',
        scenario: '需要從 0 開發 MVP：模型選型/訓練、推論部署與輕量前後端整合，用於驗證價值與內部展示',
        deliverables: 'MVP（模型 + API + 輕量介面），專案說明與操作指南',
        timeline: '約 1 ～ 2 個月起（依範疇）',
        note: '分階段里程碑交付，可簽 NDA / 合約；前後端以「模型展示/評測」為主，不承諾大型平台開發',
      },
      detail: {
        description: [
          '當你已有明確應用構想但缺少技術原型時，我們能自模型核心出發，建立最小可行產品：包含模型選型與評估、API 封裝與部署、以及可展示的輕量介面以便溝通價值。',
        ],
        bullets: [
          {
            title: '適用場景',
            items: [
              '新創/團隊需要快速驗證 PoC 可行性',
              '以 DocAligner 概念為靈感的文件理解原型（對齊/版面/鍵值抽取等）',
              '需要「可操作的 demo」來說服決策者',
            ],
          },
          {
            title: '服務範圍',
            items: [
              '需求釐清與資料設計',
              '模型訓練流程與評估策略',
              'API 打包與部署（FastAPI / Docker / 私有環境）',
              '輕量前端/後端整合（以評測與展示為目的）',
            ],
          },
        ],
        warnings: [
          '屬進階客製專案；需與決策者就目標與範疇進行階段性對齊，避免變更失控。',
        ],
        extraNotes: [
          '可依需求分階段交付（Prototype → Beta → MVP），並確保保密與權限控管。',
        ],
      },
    },
  ],

  // -----------------------------------------
  //                 EN
  // -----------------------------------------
  en: [
    // 1) Consulting (time-sliced, embedded collaboration)
    {
      brief: {
        title: 'Consulting (Time-Sliced, Embedded)',
        scenario:
          'Need steady model R&D/evaluation guidance without a large project; fixed weekly days with embedded collaboration (we augment, not replace)',
        deliverables: 'Weekly consulting/tech support, model/data decisions, staged notes & progress logs',
        timeline: 'Flexible, week-based cadence',
        note: 'Remote or scheduled sessions; great for continuous improvements and small iterations',
      },
      detail: {
        description: [
          'When you want to compound model capability but are not ready for project mode, use a “weekly time-slice” model: we embed into your process, focus on model quality/reliability, and use your toolchain.',
        ],
        bullets: [
          {
            title: 'Suitable Scenarios',
            items: [
              'Stable consulting needed but current scope is small',
              'Reserve fixed weekly days for data/model/evaluation work',
              'Prefer flexible budget/bandwidth allocation while lifting metrics',
            ],
          },
          {
            title: 'Service Model',
            items: [
              '1–2 days/week (or agreed) with your eng/data/product teams',
              'Remote or scheduled; we adopt your repos, CI, boards, comms',
              'Model research/experiment design, data governance advice, evaluation & reports',
            ],
          },
          {
            title: 'Deliverables & Timeline',
            items: [
              'Weekly rhythm with compounding progress',
              'Weekly update notes: decisions, diffs, next priorities',
              'Seamless upgrade to project mode when scope grows',
            ],
          },
        ],
        extraNotes: [
          'We augment your team; you retain ownership.',
          'Any frontend/backend is lightweight for showcasing/evaluation only.',
        ],
      },
    },

    // 2) Single Model Module: Development & Long-Term Maintenance (Modular)
    {
      brief: {
        title: 'Single Model Module: Development & Long-Term Maintenance',
        scenario:
          'Focus on one AI model module for iterative development and maintenance (e.g., DocAligner for alignment/layout). We make the model accurate, stable, and evolvable.',
        deliverables:
          'Training module + Inference module + Benchmark web (multi-model/dataset comparisons), versioned ops logs',
        timeline: 'Long-term; delivered in iterative cycles',
        note: 'New capabilities (e.g., extending DocAligner with semantic layout/key-value/ReID) are separate modules and quoted separately',
      },
      detail: {
        description: [
          'We productize and maintain a single model module end-to-end. Using DocAligner as the example: data governance → training & evaluation → inference & deployment → ongoing versioning, making every release measurable, traceable, and explainable.',
        ],
        bullets: [
          {
            title: 'Included (Core)',
            items: [
              'Training: data planning/versioning, labeling workflow, augmentation, training & finetune, experiment tracking (mAP/F1/Latency/TPR@FPR)',
              'Inference: SDK/REST, batch & streaming, perf optimization (ONNX/TensorRT/quantization), resource monitoring, latency budgeting',
              'Benchmark web: Kaggle-like leaderboard with PR/CM/speed–accuracy surfaces; compare across models & dataset versions; export reports',
            ],
          },
          {
            title: 'Showcase (DocAligner)',
            items: [
              'Compare models over rounds/dataset versions: alignment accuracy, keypoint error, latency',
              'Switch model versions on a web page and visualize hot zones on public/anon docs',
              'One-click report export for internal reviews & milestones',
            ],
          },
          {
            title: 'Collaboration & Delivery',
            items: [
              'Iteration-based: each cycle pins targets (data refresh/metric lift/perf tuning) and deliverables',
              'Technical notes/changelogs; reproducible env/scripts',
              'Integrates with your labeling/MLOps/CI-CD; you keep ownership/control',
            ],
          },
          {
            title: 'Boundaries & Extensions',
            items: [
              'New capabilities (layout semantics, key-value extraction, table structuring, cross-doc linking) are new modules',
              'Platform shifts (Jetson/Edge TPU/cloud GPU) may require re-estimation',
              'NDA and data governance aligned to your policies',
            ],
          },
        ],
        references: [
          {
            label: 'Classic Project: DocAligner (GitHub)',
            linkText: 'DocAligner Repository',
            linkHref: 'https://github.com/DocsaidLab/DocAligner',
          },
        ],
        warnings: [
          'Scope is a single model module; large platform build-outs/cross-system integration are out of scope.',
          'Third-party/commercial model licensing is excluded.',
        ],
        extraNotes: [
          'Supports PoC → Pilot → Production; we can also take over existing models.',
          'Baseline datasets and periodic benchmark reports (weekly/biweekly/monthly) included by agreement.',
        ],
      },
    },

    // 3) MVP from Zero: Demonstrable Model Product (Advanced)
    {
      brief: {
        title: 'MVP from Zero: Demonstrable Model Product (Advanced)',
        scenario:
          'Build an MVP from scratch: model selection/training, inference deployment, and lightweight FE/BE for showcasing value',
        deliverables: 'MVP (model + API + lightweight UI), docs & user guide',
        timeline: 'Approx. 1–2 months+ depending on scope',
        note: 'Milestone-based; NDA/contract ready. FE/BE is for model demo/evaluation, not for large platform dev.',
      },
      detail: {
        description: [
          'When you have a clear application idea but lack a prototype, we build the minimal viable product around the model: selection/evaluation, API packaging/deploy, and a small UI to communicate value to stakeholders.',
        ],
        bullets: [
          {
            title: 'Ideal Scenarios',
            items: [
              'Startups/teams needing fast PoC validation',
              'DocAligner-inspired document understanding prototype (alignment/layout/key-value, etc.)',
              'A working demo to persuade decision-makers',
            ],
          },
          {
            title: 'Service Scope',
            items: [
              'Requirements analysis & data design',
              'Training pipeline & evaluation strategy',
              'API packaging & deployment (FastAPI / Docker / private env)',
              'Lightweight FE/BE integration for evaluation/showcase',
            ],
          },
        ],
        warnings: [
          'Tailored and advanced; needs iterative scope alignment with decision-makers.',
        ],
        extraNotes: [
          'Phased delivery (Prototype → Beta → MVP) with confidentiality and access control.',
        ],
      },
    },
  ],

  // -----------------------------------------
  //                 JA
  // -----------------------------------------
  ja: [
    // 1) コンサル（週単位タイムスライス、内製チームと協働）
    {
      brief: {
        title: 'コンサルティング（週タイムスライス・内製協働）',
        scenario:
          '大規模案件ではないが、モデルR&D/評価の継続支援が必要。週の固定日を確保し、あなたのチームを増強（置き換えではなく協働）',
        deliverables: '毎週のコンサル/技術支援、モデル/データの意思決定メモ、進捗ログ',
        timeline: '週単位で柔軟に対応',
        note: 'リモート/定期ミーティング可。小さな反復で着実に前進',
      },
      detail: {
        description: [
          'プロジェクト化する前段階でもモデル力を積み上げたい場合、「週タイムスライス」がおすすめです。あなたのツールチェーンに合わせ、品質/信頼性を重視して内製チームと協働します。',
        ],
        bullets: [
          {
            title: '想定シチュエーション',
            items: [
              '安定した助言が必要だが現状スコープは小さい',
              'データ/モデル/評価作業のため週の固定日を確保したい',
              '予算/開発リソースを柔軟に配分しつつ指標を伸ばしたい',
            ],
          },
          {
            title: 'サービスモデル',
            items: [
              '週1〜2日（応相談）で参画、あなたの Eng/データ/PM と協働',
              'リモート/定期ミーティング。既存のRepo/CI/ボード/コミュニケーションを使用',
              'モデル調査/実験設計、データガバナンス助言、評価とレポート',
            ],
          },
          {
            title: '納品と期間',
            items: [
              '週次リズムで積み上げ',
              '意思決定メモ/差分/次の優先度を毎週共有',
              'スコープ拡大時はプロジェクトモードへシームレス移行',
            ],
          },
        ],
        extraNotes: [
          '増強（augmentation）が目的で、置き換えではありません。',
          'FE/BE は評価/デモのための軽量実装に留めます。',
        ],
      },
    },

    // 2) 単一モデルモジュール：開発と長期保守（モジュール制）
    {
      brief: {
        title: '単一モデルモジュール：開発と長期保守（モジュール制）',
        scenario:
          '1つのAIモデルモジュールに集中してイテレーション/保守（例：DocAlignerのアライメント/レイアウト）。正確・安定・進化可能な状態に保ちます。',
        deliverables:
          '学習モジュール + 推論モジュール + ベンチマークWeb（複数モデル/データ比較）、バージョン化された運用ログ',
        timeline: '長期運用。イテレーション単位で納品',
        note: 'DocAligner拡張（レイアウト意味/キー値/ReID など）は新モジュールとして別見積もり',
      },
      detail: {
        description: [
          '単一モデルモジュールを製品化し、データ設計→学習/評価→推論/デプロイ→継続バージョニングまで一貫対応。各リリースを測定可能・追跡可能・説明可能に保ちます。',
        ],
        bullets: [
          {
            title: '含まれる範囲（コア）',
            items: [
              '学習：データ設計/バージョン、アノテーション、拡張、学習/微調整、実験トラッキング（mAP/F1/Latency/TPR@FPR）',
              '推論：SDK/REST、バッチ/ストリーム、性能最適化（ONNX/TensorRT/量子化）、リソース監視・レイテンシ予算管理',
              'ベンチマークWeb：Kaggle風ダッシュボード（PR/CM/速度–精度）、モデル/データ版比較、レポート出力',
            ],
          },
          {
            title: 'DocAligner での例',
            items: [
              'ラウンド/データ版を跨いだ比較：アライメント精度、キーポイント誤差、レイテンシ',
              'Webでモデル版を切替、公開/匿名文書の品質とホットゾーン可視化',
              'ワンクリックでレビュー/マイルストーン向けレポート出力',
            ],
          },
          {
            title: '協業と納品',
            items: [
              'イテレーションごとに目標（データ更新/指標向上/性能調整）と成果物を定義',
              '技術ノート/変更履歴、再現可能な環境スクリプト',
              '既存のラベリング/MLOps/CI-CD と統合。所有権はあなた側に保持',
            ],
          },
          {
            title: '境界と拡張',
            items: [
              'レイアウト意味分割/キー値抽出/表構造化/文書間リンクなどは新モジュール扱い',
              'Jetson/EdgeTPU/クラウドGPUなどプラットフォーム変更は再見積もりの可能性',
              'NDAとデータガバナンスは貴社ポリシーに準拠',
            ],
          },
        ],
        references: [
          {
            label: 'クラシック：DocAligner（GitHub）',
            linkText: 'DocAligner リポジトリ',
            linkHref: 'https://github.com/DocsaidLab/DocAligner',
          },
        ],
        warnings: [
          '対象は単一モデル。大規模プラットフォーム構築や複雑なシステム連携はスコープ外です。',
          'サードパーティ/商用モデルのライセンス費用は含みません。',
        ],
        extraNotes: [
          'PoC → Pilot → 本番の段階導入に対応。既存モデルの引継ぎも可能。',
          'ベースラインデータセットと定期ベンチマークレポート（週/隔週/月）は合意の上で提供。',
        ],
      },
    },

    // 3) MVPをゼロから：デモ可能なモデル製品（上級）
    {
      brief: {
        title: 'MVPをゼロから：デモ可能なモデル製品（上級）',
        scenario:
          'モデル選定/学習、推論デプロイに加え、価値を伝えるための軽量FE/BEを統合したMVPを0→1で構築',
        deliverables: 'MVP（モデル + API + 軽量UI）、ドキュメント/ユーザーガイド',
        timeline: '目安 1〜2ヶ月以上（スコープ依存）',
        note: 'マイルストーンベース。NDA/契約可。FE/BEは評価/デモ目的であり、大規模プラットフォーム開発は対象外',
      },
      detail: {
        description: [
          '明確なユースケースがありつつプロトタイプが無い場合、モデルを中心に最小製品を構築。選定/評価、API化・デプロイ、小さなUIでステークホルダーに価値を伝えます。',
        ],
        bullets: [
          {
            title: '適した状況',
            items: [
              'PoCの迅速検証が必要',
              'DocAligner発想の文書理解プロトタイプ（アライメント/レイアウト/キー値など）',
              '意思決定者に見せる「動くデモ」が要る',
            ],
          },
          {
            title: 'サービス範囲',
            items: [
              '要件分析とデータ設計',
              '学習パイプラインと評価戦略',
              'API化とデプロイ（FastAPI / Docker / プライベート環境）',
              '軽量なFE/BE統合（評価・デモ用途）',
            ],
          },
        ],
        warnings: [
          'カスタム性が高いため、意思決定者と段階的にスコープをすり合わせる必要があります。',
        ],
        extraNotes: [
          'Prototype → Beta → MVP の段階納品。機密/アクセス管理に対応。',
        ],
      },
    },
  ],
};
