---
slug: build-a-resume
title: Write a Resume with JS!
authors: Z. Yuan
image: /en/img/2025/0324.webp
tags: [JavaScript, React, Resume]
description: No need to rely on resume templates, just write the code yourself!
---

This time, I needed to write a resume.

So, as usual, I started looking for some online tools, which mostly turned out to be the same old options: either the templates were too basic or you had to pay to download them.

After taking a look around, one thought kept popping into my mind:

> **Why does everything have to be behind a paywall?**

<!-- truncate -->

## Then I’ll Do It Myself!

Anyone who’s written a resume before knows that writing good content is one thing, but having a beautiful layout is another skill altogether.

The text has to be right, the layout has to be neat, and the style should be clean and professional. With all these requirements, it’s basically no different from creating a webpage.

Since I already know how to build webpages, why not just use JS to write the resume directly?

Not only does this allow for full customization of the layout and style, but it also gives me the freedom to assemble modules as I like. It’s also great for version control. Plus, with React’s component-based structure, maintaining it becomes even clearer.

## Configuration File

I keep all the data for the resume in a single configuration file called `resumeData.js`.

This way, I can clearly separate the data from the presentation logic. If I need to change the data, switch languages, or even add API integration, it becomes much more flexible.

This data mainly includes personal information, a list of skills, about me, work experience, personal achievements, and education.

Here is an example with some placeholder data I filled in:

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

## Two-Column Layout

This time, we chose a two-column layout to present the resume.

This design is considered a classic format for resumes. It clearly categorizes the information and doesn’t feel too overwhelming visually.

- The left column mainly contains static basic information, such as name, title, contact details, and skills.
- The right column is responsible for dynamic content related to experiences, such as the self-introduction, work experience, achievements, and education.

### Contact Information

For the contact information, we use icons paired with text to list various contact methods, including clickable links (such as LinkedIn, GitHub, and personal website), allowing potential employers to find you with just a click:

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

### Skill Tree

The skills section is a small highlight that combines both visual appeal and informative value.

Each skill has a name, proficiency label, and a corresponding progress bar, making it easy to see at a glance the areas of expertise and their levels.

Here, we directly render from the `resumeData.skills` array, with each entry containing the skill name and proficiency percentage:

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

### About Me

This section is typically used for a short but impactful self-introduction.

Avoid falling into vague, template-like phrases. Instead, emphasize your professional skills, technical focus, and the core experiences you’ve accumulated over time:

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

### Work Experience

This section is the centerpiece of the resume. For the format, it's recommended to use a card-based structure to organize the information: job title, company name, employment duration, and a few specific descriptions of key responsibilities or achievements. This allows HR or hiring managers to quickly grasp what you've done and what results you've achieved.

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

### Personal Achievements

:::tip
Some companies may discourage engineers from writing blogs, so use this section with discretion.
:::

In addition to your full-time job, if you have a blog, open-source projects, or have participated in competitions, it’s definitely worth including.

These details can sometimes highlight an engineer's passion and ongoing commitment more than formal job roles.

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

### Education

The format here is relatively simple, but it’s still important to ensure the layout is clear and organized.

If you're a recent graduate or career changer, your education may take up a larger portion of your resume, so it's important to present it well.

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

### Left Column

We place the contact information and skills in the left column, along with the name and job title at the top. This creates a clear structure, making the information easy to read and understand at a glance.

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

### Right Column

The right column is used to display the dynamic resume content, such as the self-introduction, experience, and educational background. This layout also aligns more closely with the typical reading habits of hiring managers.

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

## Final Display

Finally, we wrap the left and right column components into a unified container component, and the entire resume is complete.

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

I have separated the CSS into its own style file for easier maintenance and customization. Here’s the link to the style file, where all the styles are contained, and you can change the design however you like:

- [**src/components/Resume/resume.css**](https://github.com/DocsaidLab/website/blob/main/src/components/Resume/resume.css)

## Final Product

The result after execution is shown in the image below.

Overall, it looks clean, concise, and organized. It’s actually not any worse than the paid templates out there. Moreover, this resume is 100% customizable and 100% controllable, making it super easy to update the content, add modules, or even change languages.

If you're looking to create your own technical resume, using JS to build it yourself might be an interesting approach to try.

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

## Final Step

If you want to convert your HTML file into a PDF, you can use the `puppeteer` package.

This package, developed by the Chrome team, is a headless browser automation tool that allows you to control browser actions programmatically, such as opening webpages, clicking buttons, taking screenshots, and even converting entire pages into PDFs.

Here’s a simple example of how to use Puppeteer to output the resume's HTML as a PDF:

```js title="html2pdf.js"
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Load local HTML or a remote URL
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

Note that the URL in `goto()` can be either a local development server (like localhost) or the deployed resume page URL.

Setting `printBackground: true` will retain the CSS background and colors you’ve defined. If your page has dynamic data rendering (e.g., React's SPA), make sure the content is fully loaded before exporting. Using `waitUntil: 'networkidle0'` is a safer approach.

With this, you can easily export your beautifully crafted JS resume into a high-quality PDF, ready to send or print!