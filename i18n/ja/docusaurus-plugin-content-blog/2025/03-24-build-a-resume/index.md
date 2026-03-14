---
slug: build-a-resume
title: JS で履歴書を書いてみよう！
authors: Z. Yuan
image: /ja/img/2025/0324.webp
tags: [JavaScript, React, Resume]
description: テンプレートに頼らず、自分でコードを書く
---

今回は履歴書を書く必要がありました。

そこでいつものようにオンラインツールを探してみたのですが、だいたい似たようなものばかりでした。テンプレートの見た目がいまひとつだったり、ダウンロードには課金が必要だったり。

一通り見たあと、頭に浮かんだのはこれだけです。

> **毎回毎回、どうしてすぐ課金になるんだろう？**

<!-- truncate -->

## それなら自分で作ろう

履歴書を書いたことがある人なら分かると思いますが、内容をきちんと書くことと、見た目をきれいに整えることは別問題です。

誤字がなく、レイアウトが整っていて、全体の雰囲気も清潔でプロフェッショナルであること。これらを全部そろえようとすると、やっていることはほとんど普段のフロントエンド実装と変わりません。

だったら、もともと Web を書けるのだし、JS でそのまま履歴書を組んでしまえばいいのでは？

この方法なら、レイアウトも完全に自分で制御できますし、モジュールの組み替えも自由です。さらにバージョン管理もしやすく、React のコンポーネント構成のおかげで保守もしやすくなります。

## 設定ファイル

履歴書の内容データは、ひとつの設定ファイル `resumeData.js` にまとめています。

こうしておくと、データと表示ロジックをきれいに分離できます。後から内容を変更したり、言語を切り替えたり、将来的に API からデータを取るようにしたりするのも簡単です。

このデータには、個人情報、スキル一覧、自己紹介、職務経験、個人成果、学歴などを含めています。

以下はサンプルです。説明用にダミーデータを少し入れています。

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

## 左右 2 カラム

今回は左右 2 カラムのレイアウトで履歴書を構成します。

この形式は履歴書では定番で、情報を分類しやすく、見た目にも窮屈になりにくいのが利点です。

- 左側には、氏名、肩書き、連絡先、スキルなどの静的な基本情報を置く
- 右側には、自己紹介、職務経験、実績、学歴といった経歴系の情報を置く

### 連絡先情報

連絡先は、アイコンとテキストを組み合わせて各種手段を並べる形にしました。LinkedIn、GitHub、個人サイトなどのリンクも付けておけば、採用担当者がそのまま確認できます。

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

### スキル

スキル欄は、視覚的にも情報量的にも分かりやすいポイントになります。

各スキルに名称、熟練度ラベル、進捗バーを付けておけば、どの分野が得意で、どの程度できるのかを一目で伝えられます。

ここでは `resumeData.skills` 配列をそのままレンダリングしています。

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

このセクションには、短くても印象に残る自己紹介を書くのが向いています。

ありきたりな定型文ではなく、自分の専門領域、技術の軸、これまで積み上げてきた経験の核をはっきり示した方が効果的です。

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

ここは履歴書の中心になる部分です。カード型の構成にして、職種、会社名、在籍期間、具体的な業務内容や成果をまとめると、採用担当者が内容を素早く把握しやすくなります。

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

### 個人成果

:::tip
会社によっては、エンジニアの個人ブログ運営を歓迎しない場合もあるので、記載するかどうかは状況を見て判断してください。
:::

正職での仕事以外にも、ブログ運営、オープンソース活動、コンテスト参加などがあれば、十分に書く価値があります。

こうした情報は、正式な職歴以上に、エンジニアとしての熱意や継続性を伝えてくれることがあります。

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

ここは比較的シンプルなセクションですが、やはり見やすさと整理の良さは重要です。

新卒や異業種からの転職であれば、学歴の比重が高くなることもあるので、丁寧に見せた方がよいでしょう。

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

### 左カラム

左側には連絡先とスキルを配置し、最上部に氏名と職種を置いて、全体の情報構造がひと目で分かるようにします。

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

### 右カラム

右側には、自己紹介、職務経験、実績、学歴といった動的な履歴情報を配置します。この方が読む側の視線の流れにも自然に合います。

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

最後に、左右のカラムをひとつのコンテナコンポーネントにまとめれば、履歴書全体の構成は完成です。

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

CSS は別ファイルに切り出してあり、保守やカスタマイズがしやすいようにしています。スタイルは以下のファイルでまとめて管理しています。

- [**src/components/Resume/resume.css**](https://github.com/DocsaidLab/website/blob/main/src/components/Resume/resume.css)

## 完成品

実行後の見た目は次のとおりです。

全体として、すっきりしていて見やすく、有料テンプレートと比べても十分戦える仕上がりです。しかも、この履歴書は 100% カスタマイズ可能で、内容の差し替えやモジュール追加、言語切り替えもとても簡単です。

もし自分だけの技術履歴書を作ってみたいなら、JS で自作するのはかなり面白い方法だと思います。

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

HTML ファイルを PDF に変換したい場合は、`puppeteer` パッケージが使えます。

これは Chrome チームが開発している headless browser 自動化ツールで、プログラムからブラウザを操作できます。Web ページを開く、ボタンを押す、スクリーンショットを撮る、ページ全体を PDF にする、といった処理が可能です。

以下は、Puppeteer を使って履歴書の HTML を PDF として出力する簡単な例です。

```js title="html2pdf.js"
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // ローカル HTML またはリモート URL を読み込む
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

`goto()` に渡す URL は、ローカル開発サーバー（localhost）でも、デプロイ後の履歴書ページでも構いません。

`printBackground: true` を指定すると、CSS で設定した背景や色をそのまま保持できます。React の SPA のように動的レンダリングがあるページでは、内容が完全に読み込まれてから出力するようにしてください。`waitUntil: 'networkidle0'` はそのための比較的安全な指定です。

これで、JS で作った履歴書を高品質な PDF として簡単に書き出せます。送付にも印刷にもそのまま使えます。
