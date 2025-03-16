# Worklog

---

:::info
This page is solely for recording our work activities.
:::

---

## 2025

### March

- Completed the second-generation backend of the website
  - [x] User login/registration system.
  - [x] Email system.
  - [x] API management system.

### February

- Refactored demo-related code for better functionality sharing.
- Redesigned the website homepage layout and added more sections.
- Added `CookieBanner` feature to comply with relevant regulations.
- **MRZScanner**: Completed the second-phase MRZ model and released version 1.0.6.
- Completed MRZScanner Demo functionality: [**mrzscanner-demo**](https://docsaid.org/en/playground/mrzscanner-demo)
- Wrote research paper notes, totaling 165 entries so far.

### January

- Shut down GCP, change the model file storage location, and update all download links.
  - Update the model files for Capybara, DocAligner, and MRZScanner.
- **DocsaidKit**: Completed the split, retired the project.

## 2024

### December

- Invited four authors to share their daily development experiences.
- Completed the **StockAnalysis Demo** feature: [**stock-demo**](https://docsaid.org/en/playground/stock-demo)
- Added an "Author Field" to the paper notes and also beautified the homepage, paper notes, and blog styles.
- Started building the website's backend management system, database system, and member registration system.
- Wrote paper notes, with a current total of 150 entries.

### November

- Added i18n: [**日本語**](https://docsaid.org/ja/)
- Updated **@docusaurus/core@3.6.1** and found out it's not backward compatible...
  - Spent time updating the problematic code sections.
- Wrote paper reviews, totaling 135 papers.
- **DocumentAI**: Continued development.
- **TextRecognizer**: Continued development from October.

### October

- Completed the model demo functionality: [**docaligner-demo**](https://docsaid.org/en/playground/docaligner-demo)
- Moved NextCloud from our own host to GCP and updated all download links.

### September

- **MRZScanner**: Project completed and made open-source. 🎉 🎉 🎉
- **TextDetector**: Continued development, following progress made in March.
- Came across a beautifully designed website and had to note it down:
  - [**Hello Algorithm**](https://www.hello-algo.com/)
- Wrote paper reviews, totaling 100 papers.

### August

- **MRZScanner**: Deployment testing and rework.
- Updated **@docusaurus/core@3.5.2** and found out it's not backward compatible...
  - Spent time updating the problematic code sections.
- Investigated OpenCV dependency issues and discovered we weren’t alone:
  - [**修复 OpenCV 依赖错误的小工具：OpenCV Fixer**](https://soulteary.com/2024/01/07/fix-opencv-dependency-errors-opencv-fixer.html)
  - Open-source project: [**soulteary/opencv-fixer**](https://github.com/soulteary/opencv-fixer/tree/main)
  - Thanks to [**蘇洋博客**](https://soulteary.com/) for sharing and saving us a lot of time.
- Wrote paper reviews, with 90 papers reviewed so far.

### July

- Wrote paper reviews, with around 80 papers in total so far.
- **MRZScanner**: Began development.

### June

- **AutoTraderX**: Completed the API integration for Yuanta Securities and made it open-source. 🎉 🎉 🎉
- Ran out of funds for OpenAI services, so we suspended the daily news push from GmailSummary.
- Continued writing paper reviews, totaling 50 papers by this point.

### May

- Finished developing the **Text Recognizer** model.
  - Final evaluation results were promising, but we think it’s still an "overfitted model pretending not to be overfitted." (???)
  - Since it doesn’t meet our ideal standards yet, we’ve decided not to release it for now.
- Explored **Docusaurus**' Search feature, tested and integrated Algolia search service.
  - Thanks to [**WeiWei**](https://github.com/WeiYun0912) for the tutorial:
    - [**[docusaurus] 在 Docusaurus 中使用 Algolia 實作搜尋功能**](https://wei-docusaurus-vercel.vercel.app/docs/Docusaurus/Algolia)
- Continued working on the **Text Recognizer** model, adjusting parameters and training.
- **AutoTraderX**: Development started.

### April

- Learned how to configure CSS styles to tweak the blog’s appearance.
  - Thanks to [**朝八晚八**](https://from8to8.com/) for the helpful article:[**部落格首頁**](https://from8to8.com/docs/Website/blog/blog_homepage/)
- **TextRecognizer**: Continued development from **WordCanvas** and made further progress on the text recognition project.
- **GmailSummary**: Modified functionality to push daily news to the tech documentation page.
- Completed technical documentation for all ongoing projects.
- Explored **Docusaurus**’ i18n functionality and started writing English documentation.
- Investigated **Docusaurus**’ documentation features and began migrating content from GitHub to the platform.
- **WordCanvas**: Project completed and made open-source. 🎉 🎉 🎉

### March

One day, we found that the Google Drive download feature broke—what was once accessible through `gen_download_cmd` became a garbled mess of HTML. 👻 👻 👻

After considering several options...

We decided to use [**NextCloud**](https://github.com/nextcloud) to set up a private cloud for storing data and updated our previous download links accordingly.

- **GmailSummary**: Completed development and made it open-source. 🎉 🎉 🎉
- **DocClassifier**: Discovered that stacking multiple normalization layers significantly improved model performance (a surprising discovery...).
- **TextRecognizer**: Early-stage project planning.
- **WordCanvas**: Development started.
- **TextDetector**: Ran into several issues and decided to put it on hold for now.

### February

- **TextDetector**: Collected public datasets.
- **DocClassifier**: Introduced **CLIP** into the model and applied knowledge distillation with excellent results!
- Explored **Docusaurus**’ commenting functionality and integrated **giscus** for comments.
  - Thanks to [**不務正業的架構師**](https://ouch1978.github.io/) for the insightful guide:
    - [**在文件庫和部落格的文章下方加上 giscus 留言區**](https://ouch1978.github.io/docs/docusaurus/customization/add-giscus-to-docusaurus)

### January

- **TextDetector**: Early-stage project planning.
- **DocClassifier**: Project completed and made open-source. 🎉 🎉 🎉

## 2023

### December

- **DocClassifier**: Development started.
- **DocAligner**: Completed development and made it open-source. 🎉 🎉 🎉
- **Website**: Discovered Meta’s interesting open-source project [**Docusaurus**](https://github.com/facebook/docusaurus). It provides a simple way to build a static website using Markdown for content creation, so I decided to use it to write a blog.
- Abandoned and deleted the WordPress-built website, migrating all content to the GitHub `website` project.

### November

- **DocClassifier**: Early-stage project planning.
- **DocsaidKit**: Completed development and made it open-source. 🎉 🎉 🎉
- Wrote paper reviews, totaling 20 papers.

### October

- **WordCanvas**: Early-stage project planning.
- **DocGenerator**: Completed phase two of development, splitting the text synthesis module into the **WordCanvas** project.

### September

- **DocAligner**: Development started.
- **DocGenerator**: Phase one of development completed.
- Wrote paper reviews, totaling 5 papers.

### August

- **DocAligner**: Early-stage project planning.
- **DocsaidKit**: Organized commonly used tools and started development.
- Explored [**WordPress**](https://wordpress.org/) functionality, experimented with building a personal blog.
  - Thanks to [**諾特斯網站**](https://notesstartup.com/) for the generous knowledge sharing.
- Created a **DOCSAID** GitHub account and started planning various projects.

### Before This

We drifted between various jobs, day after day, year after year. Listening to different bosses tell the same dreams, chewing on those tasteless promises.

Countless projects rushed to completion through sleepless nights, intertwining passionate ideals, only to sway between the capital market’s affection and indifference.

When the love fades, it all falls apart.

Before our youth completely slips away, we still want to leave something behind.

Anything will do—just to mark that we were here.
