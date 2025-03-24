---
slug: build-a-resume
title: 用 JS 來寫一份履歷吧！
authors: Z. Yuan
image: /img/2025/0324.webp
tags: [JavaScript, React, Resume]
description: 履歷模板不求人，自己寫 Code！
---

這次我需要寫份履歷。

於是我照慣例先去找一些線上工具，不外乎就是幾種老樣子：不是模板設計太過陽春，就是要付費才能下載。

看了一圈後，心中只浮出一個念頭：

> **這一天天的，怎麼又要收費？**

<!-- truncate -->

## 那就自己上！

有寫過履歷的朋友應該都知道，內容要寫得好是一回事，排版要漂亮也是另一門學問。

字要對、排版要整齊、風格還要乾淨專業，這些需求加起來，根本就跟我們平常在切網頁沒兩樣。

那既然本來就會寫網頁，何不乾脆直接用 JS 把履歷寫出來？

而且這樣一來，不但可以完全客製化排版風格、自由組裝模組，也方便版本管理。加上 React 組件化的結構特性，維護起來也更加清晰。

## 配置文件

履歷的內容資料我統一集中在一個配置檔案 `resumeData.js` 裡。

這樣一來可以清楚地分離資料與呈現邏輯，未來若要改資料、換語言、甚至加上 API 取得，也會更彈性。

這份資料主要包含個人資訊、技能列表、關於我、工作經驗、個人成就與學歷等欄位。

以下是範例內容，我直接填了一些示意資料：

```javascript title="src/data/resumeData.js"
const resumeData = {
  name: "Z. Yuan",
  title: "Senior CV/ML Engineer",
  contact: {
    phone: "09xx-xxx-xxx",
    email: "xxx@gmail.com",
    location: "XXX City, Taiwan",
    linkedin: "https://www.linkedin.com/in/ze-yuan-sh7/",
    github: "https://github.com/zephyr-sh",
    website: "https://docsaid.org"
  },
  skills: [
    { skillName: "Python", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "PyTorch", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "Deep Learning", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "Computer Vision", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "ONNX Runtime", levelLabel: "Proficient", levelWidth: "85%" }
  ],
  aboutMe: `
    Senior CV/ML Engineer with strong expertise in deep learning, MLOps, and document processing.
  `,
  workExperience: [
    {
      role: "Senior AI Engineer",
      company: "CompanyA, Taipei",
      date: "Aug 2020 - Present",
      highlights: [
        "Developed OCR and facial recognition solutions.",
        "Optimized deployment using Docker and ONNX Runtime."
      ]
    },
    {
      role: "ML Engineer",
      company: "CompanyB, Taipei",
      date: "Feb 2020 - Jun 2020",
      highlights: [
        "Built threat detection models.",
        "Improved data pipeline efficiency."
      ]
    }
  ],
  personalAchievements: [
    {
      title: "Web Design",
      description: "Created a multilingual technical blog."
    },
    {
      title: "Open Source",
      description: "Contributed to deep learning projects."
    }
  ],
  education: [
    {
      degree: "Master's Degree, XXXX",
      date: "XXXX - XXXX",
      desc: "Dept. of XXXX Engineering"
    }
  ]
};

export default resumeData;
```

## 左右兩欄

我們這次選擇使用左右欄的版型來呈現履歷。

這種設計算是履歷版型中的經典款，資訊清楚分類，視覺上也不會太壓迫。

- 左邊主要放靜態基本資料，例如姓名、職稱、聯絡方式與技能
- 右邊則負責放經歷類動態內容，如自我介紹、工作經驗、成就與學歷等。

### 聯繫資訊

聯繫資訊的呈現方式上，透過 icon 搭配文字列出各項聯絡方式，並包含跳轉連結（如 LinkedIn、GitHub、個人網站），讓用人單位點一下就能馬上找到你：

```javascript title="src/components/Resume/ContactInfo.js"
import React from "react";
import resumeData from "../../data/resumeData";

function ContactInfo() {
  const { phone, email, location, linkedin, github, website } = resumeData.contact;
  return (
    <div className="section">
      <h3>
        <i className="fa-solid fa-address-book"></i> Contact Information
      </h3>
      <ul className="contact-list">
        <li>
          <i className="fa-solid fa-phone"></i> {phone}
        </li>
        <li>
          <i className="fa-solid fa-envelope"></i> {email}
        </li>
        <li>
          <i className="fa-solid fa-location-dot"></i> {location}
        </li>
        <li>
          <a href={linkedin} target="_blank" rel="noreferrer">
            <i className="fa-brands fa-linkedin"></i> LinkedIn
          </a>
        </li>
        <li>
          <a href={github} target="_blank" rel="noreferrer">
            <i className="fa-brands fa-github"></i> GitHub
          </a>
        </li>
      </ul>
      <a className="website-button" href={website} target="_blank" rel="noreferrer">
        <i className="fa-solid fa-globe"></i> My Website
      </a>
    </div>
  );
}

export default ContactInfo;
```

### 技能樹

技能區塊是個視覺與資訊兼具的小亮點。

每項技能都有名稱、熟練度標籤與對應的進度條，讓人能一眼看出擅長領域與程度。

這裡我們直接從 `resumeData.skills` 陣列中進行渲染，每筆資料包含技能名稱與熟練度百分比：

```javascript title="src/components/Resume/Skills.js"
import React from "react";
import resumeData from "../../data/resumeData";

function Skills() {
  return (
    <div className="section">
      <h3>Skills</h3>
      <ul className="skills-list">
        {resumeData.skills.map((item, idx) => (
          <li key={idx}>
            <div className="skill-item">
              <div className="skill-label">
                <span>{item.skillName}</span>
                <span>{item.levelLabel}</span>
              </div>
              <div className="skill-bar">
                <div className="skill-level" style={{ width: item.levelWidth }}></div>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Skills;
```

### 關於自己

這區塊通常用來放一句簡短但有力的自我介紹。

避免落入空泛模板文句，不妨強調個人專業、技術主軸與過去累積的經驗核心：

```javascript title="src/components/Resume/AboutMe.js"
import React from "react";
import resumeData from "../../data/resumeData";

function AboutMe() {
  return (
    <div className="section-right">
      <h3>About Me</h3>
      <p className="about-me">{resumeData.aboutMe}</p>
    </div>
  );
}

export default AboutMe;
```

### 工作經驗

這一區是履歷的重頭戲，格式上建議用卡片式結構來組織：職稱、公司名稱、在職時間，以及幾條具體描述的工作重點或成就，讓 HR 或主管快速掌握你做過什麼、有什麼成果。

```javascript title="src/components/Resume/WorkExperience.js"
import React from "react";
import resumeData from "../../data/resumeData";

function WorkExperience() {
  return (
    <div className="section-right">
      <h3>Work Experience</h3>
      <div className="card-container">
        {resumeData.workExperience.map((exp, idx) => (
          <div className="card" key={idx}>
            <h4>
              {exp.role}
              <span className="company">{exp.company}</span>
            </h4>
            <div className="date">{exp.date}</div>
            <ul>
              {exp.highlights.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default WorkExperience;
```

### 個人成就

:::tip
有些公司會拒絕工程師寫部落格，請斟酌使用。
:::

除了正職工作之外，若你有經營部落格、開源專案、參與比賽等，也非常值得寫進來。

這些資訊有時比正式職務更能突顯一個工程師的熱情與持續投入的態度。

```javascript title="src/components/Resume/PersonalAchievements.js"
import React from "react";
import resumeData from "../../data/resumeData";

function PersonalAchievements() {
  return (
    <div className="section-right">
      <h3>Personal Achievements</h3>
      <div className="card-container">
        {resumeData.personalAchievements.map((item, idx) => (
          <div className="card" key={idx}>
            <strong>{item.title}</strong>
            <p>{item.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PersonalAchievements;
```

### 教育程度

這裡的格式相對簡單，但仍需注意排版清楚、有條理。

若你是新鮮人或轉職者，學歷可能佔比較大比重，就更要好好呈現。

```javascript title="src/components/Resume/Education.js"
import React from "react";
import resumeData from "../../data/resumeData";

function Education() {
  return (
    <div className="section-right" style={{ padding: "15px 20px" }}>
      <h3>Education</h3>
      <div className="education-block">
        {resumeData.education.map((edu, idx) => (
          <div className="education-item" key={idx}>
            <h4>{edu.degree}</h4>
            <div className="date">{edu.date}</div>
            <p>{edu.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Education;
```

### 左欄

我們把聯絡資訊與技能放在左側欄，搭配頂部的姓名與職稱，整體呈現結構清楚，資訊一目了然。

```javascript title="src/components/Resume/LeftColumn.js"
import React from "react";
import resumeData from "../../data/resumeData";
import ContactInfo from "./ContactInfo";
import Skills from "./Skills";

function LeftColumn() {
  return (
    <div className="left-column">
      <div className="name-title">
        <h1>{resumeData.name}</h1>
        <h2>
          <i className="fa-solid fa-robot"></i> {resumeData.title}
        </h2>
      </div>
      <ContactInfo />
      <Skills />
    </div>
  );
}

export default LeftColumn;
```

### 右欄

右側欄用來顯示動態的履歷內容，像是自我介紹、經驗與教育背景，這樣的配置也比較符合用人單位閱讀的習慣：

```javascript title="src/components/Resume/RightColumn.js"
import React from "react";
import AboutMe from "./AboutMe";
import Education from "./Education";
import PersonalAchievements from "./PersonalAchievements";
import WorkExperience from "./WorkExperience";

function RightColumn() {
  return (
    <div className="right-column">
      <AboutMe />
      <WorkExperience />
      <PersonalAchievements />
      <Education />
    </div>
  );
}

export default RightColumn;
```

## 整合顯示

最後，將左右欄組件包進統一的容器元件中，整份履歷就算是組合完成。

```javascript title="src/components/Resume/ResumeContainer.js"
import React from "react";
import LeftColumn from "./LeftColumn";
import RightColumn from "./RightColumn";
import "./resume.css";

function ResumeContainer() {
  return (
    <div className="resume-container">
      <LeftColumn />
      <RightColumn />
    </div>
  );
}

export default ResumeContainer;
```

CSS 的部分我有單獨拉出一份樣式檔，方便維護與客製，這邊附上樣式檔連結，樣式全放這份檔案裡，風格要再怎麼改就隨你高興：

- [**src/components/Resume/resume.css**](https://github.com/DocsaidLab/website/blob/main/src/components/Resume/resume.css)

## 成品

執行後的成果如下圖所示。

整體看起來乾淨、簡潔、有條理，比起外面那些要收費的模板其實不差。而且這份履歷 100% 客製、100% 可控，該換內容、加模組、甚至變換語言都超級輕鬆。

若你也想打造一份屬於自己的技術履歷，這種用 JS 自製履歷的方式，也許會是個有趣的嘗試。

import ResumeContainer from '@site/src/components/Resume/ResumeContainer';
import { Helmet } from "react-helmet";

<Helmet>
  <style>
    {`
      header, footer { display: none; }
      body { background-color: #f9f9f9; }
    `}
  </style>
  <link
    rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
  />
</Helmet>

<ResumeContainer />

## 最後

如果你想要把 HTML 檔案轉成 PDF 檔，可以使用 `puppeteer` 套件。

這個套件是由 Chrome 團隊開發的 headless browser 自動化工具，能讓你用程式控制瀏覽器的行為，例如開網頁、點按鈕、截圖、甚至直接將整個頁面轉成 PDF。

以下是一個簡單的範例，說明如何用 Puppeteer 把履歷的 HTML 輸出成 PDF：

```js title="html2pdf.js"
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // 載入本地 HTML 或遠端網址
  await page.goto('http://localhost:3000/resume', {
    waitUntil: 'networkidle0',
  });

  await page.pdf({
    path: 'resume.pdf',
    format: 'A4',
    printBackground: true,
  });

  await browser.close();
})();
```

注意到 `goto()` 裡的網址可以是本地開發伺服器（如 localhost）或部署後的履歷頁面網址。

設定 `printBackground: true` 可以保留你設定的 CSS 背景與色彩。如果你的頁面有動態資料渲染（例如 React 的 SPA），記得確保內容已經載入完成再匯出，`waitUntil: 'networkidle0'` 是比較保險的做法。

這樣就能輕鬆將你用 JS 打造的漂亮履歷，匯出成高品質 PDF，寄出去、印出來都沒問題！
