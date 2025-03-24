---
slug: build-a-resume
title: JSを使って履歴書を書いてみよう！
authors: Z. Yuan
image: /ja/img/2025/0324.webp
tags: [JavaScript, React, Resume]
description: テンプレートに頼らず、自分でコードを書く！
---

今回は履歴書を書く必要がある。

そこで、いつものようにオンラインツールを探してみたが、どれもこれも昔ながらのパターンだった。テンプレートがあまり洗練されていないか、あるいは有料でしかダウンロードできないものばかりだった。

一通り見渡した後、心に一つの考えが浮かんだ：

> **毎日毎日、どうしてまた有料にするの？**

<!-- truncate -->

## それなら自分でやってみよう！

履歴書を書いたことがある友人なら、内容が良く書けるのは一つのことだが、レイアウトが美しく仕上がるのもまた別の技だということはよく分かるはずだ。

文字が正確で、レイアウトが整然としており、スタイルが清潔でプロフェッショナルでなければならず、これらの要求は、普段ウェブサイトを作るのと同じくらいの手間がかかる。

そもそもウェブサイトが作れるなら、なぜJSを使って直接履歴書を書いてみないのだろうか？

さらに、この方法ならレイアウトのスタイルを完全にカスタマイズでき、自由にモジュールを組み合わせることができ、バージョン管理も容易になる。加えて、Reactのコンポーネント化された構造特性により、保守もより明確になる。

## 設定ファイル

履歴書の内容データは、すべてひとつの設定ファイル `resumeData.js` に集約している。

これにより、データと表示ロジックを明確に分離でき、将来的にデータを変更したり、言語を切り替えたり、さらにはAPIから取得することも可能になる。

このデータには主に個人情報、スキルリスト、「私について」、職務経験、個人的な成果、学歴などの項目が含まれている。

以下はサンプル内容で、例としていくつかのデータを入力している：

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

今回は左右カラムのレイアウトを使用して履歴書を表現します。

このデザインは履歴書レイアウトのクラシックなスタイルであり、情報が明確に分類され、視覚的にも圧迫感が少ないです。

- 左側には主に静的な基本情報、例えば氏名、職位、連絡先、スキルなどを配置します
- 右側は経歴などの動的な内容、例えば自己紹介、職務経験、成果、学歴などを配置します

### 聯繫資訊

連絡情報の表示方法としては、アイコンとテキストを組み合わせて各種連絡手段を列挙し、リンク先（例：LinkedIn、GitHub、個人ウェブサイト）を含むことで、採用担当者がワンクリックであなたの情報にアクセスできるようにします：

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

### スキルツリー

スキルセクションは、視覚と情報の両面を兼ね備えた小さな見どころです。

各スキルには名称、熟練度ラベル、そして対応する進捗バーがあり、一目で得意分野とその程度が分かります。

ここでは、`resumeData.skills` 配列から直接レンダリングし、各項目にはスキルの名称と熟練度のパーセンテージが含まれています：

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

### 自己紹介

このセクションは通常、短くて力強い自己紹介文を掲載するために用いられます。

ありきたりなテンプレート文句に陥らず、個人の専門性、技術の主軸、そして過去に積み上げた経験の核となる部分を強調してみてください：

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

### 職務経験

このセクションは履歴書の重要な部分です。フォーマットとしては、カード型の構造を使用することをお勧めします：職種、会社名、勤務期間、そしていくつかの具体的な職務内容や成果を記載することで、HRや上司があなたが行った業務内容や成果を素早く把握できるようにします。

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

### 個人の実績

:::tip
一部の企業ではエンジニアがブログを書くことを禁止している場合がありますので、使用には注意が必要です。
:::

正社員としての仕事以外にも、もしブログを運営していたり、オープンソースプロジェクトに参加したり、コンテストに参加した経験があれば、それらも非常に価値のある情報です。

これらの情報は、時には正式な職務よりもエンジニアとしての情熱や継続的な取り組みの姿勢を強調することができます。

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

### 学歴

こちらのフォーマットは比較的シンプルですが、レイアウトを分かりやすく、整理された形にすることが重要です。

もしあなたが新卒や転職者であれば、学歴が比較的大きな比重を占めることになりますので、特にしっかりと表現することが求められます。

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

### 左側欄

連絡先情報とスキルは左側の欄に配置し、上部には名前と職種を並べることで、全体の構造が分かりやすく、情報が一目で理解できるようにします。

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

### 右側欄

右側の欄は、動的な履歴情報を表示するために使用します。例えば自己紹介、経験、学歴などが含まれます。この配置は、採用担当者が読む際の習慣にもより合ったものです：

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


## 統合表示

最後に、左右の欄を統一のコンテナコンポーネントに包み込むことで、履歴書全体が完成します。

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

CSSの部分については、別途スタイルファイルを分けており、メンテナンスやカスタマイズがしやすいようにしています。こちらにスタイルファイルのリンクを添付しますので、スタイルはこのファイル内で全て管理されており、デザイン変更もお好みで行えます。

- [**src/components/Resume/resume.css**](https://github.com/DocsaidLab/website/blob/main/src/components/Resume/resume.css)

## 完成品

実行後の成果は以下の図の通りです。

全体的に見て、清潔でシンプル、整然としており、市販の有料テンプレートに引けを取らないクオリティです。また、この履歴書は100%カスタマイズ可能で、100%コントロールできるため、内容の変更やモジュールの追加、さらには言語の変更も非常に簡単に行えます。

もしあなたも自分だけの技術履歴書を作りたいのであれば、JSで自作する方法は、面白い挑戦になるかもしれません。

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

## 最後に

もしHTMLファイルをPDFに変換したい場合は、`puppeteer`パッケージを使用することができます。

このパッケージは、Chromeチームによって開発されたヘッドレスブラウザの自動化ツールで、プログラムでブラウザの動作を制御できるようにします。たとえば、ウェブページを開いたり、ボタンをクリックしたり、スクリーンショットを撮ったり、ページ全体をPDFに変換することができます。

以下は、Puppeteerを使って履歴書のHTMLをPDFに出力する簡単な例です：

```js title="html2pdf.js"
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // ローカルHTMLまたはリモートURLを読み込む
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

`goto()`内のURLは、ローカル開発サーバー（例えばlocalhost）またはデプロイ後の履歴書ページのURLでもかまいません。

`printBackground: true`を設定すると、設定したCSSの背景や色を保持することができます。もしページに動的なデータレンダリング（例えばReactのSPA）が含まれている場合は、内容が完全に読み込まれてからエクスポートするようにしてください。`waitUntil: 'networkidle0'`はより安全な方法です。

これで、JSで作成した素晴らしい履歴書を簡単に高品質のPDFに変換し、送信したり、印刷したりすることができます！