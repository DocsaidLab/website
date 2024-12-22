[中文](./README.md) | **[English](./README_en.md)** | [日本語](./README_ja.md)

# Website

> We think, we record, we persistently move forward.

## Dear Sir/Madam

Thank you for joining our group of authors. Below are some commonly used tools and guidelines that will help you get started more quickly.

## Start the Website

We currently use Docusaurus as the development framework. Below are the basic commands to start the website:

```bash
git clone https://github.com/DocsaidLab/website.git
cd website
nvm use node
yarn start
```

### Detailed Setup Process

Please follow the steps below to check and configure your environment to ensure it runs smoothly:

1. **Check Node.js Version**

   - We recommend using Node.js v22 or above.
   - If you haven't installed `nvm`, please refer to the [official nvm documentation](https://github.com/nvm-sh/nvm#installing-and-updating) for installation.
   - After installation, run the following commands to check and set the Node.js version:
     ```bash
     nvm install 22
     nvm use 22
     ```

2. **Install Yarn**

   - If you haven't installed Yarn, you can install it with the following command:
     ```bash
     npm install -g yarn
     ```

3. **Install Dependencies**

   - After entering the project directory, run the following command to install the required dependencies:
     ```bash
     yarn install
     ```

4. **Check Website Startup Port**

   - By default, Docusaurus runs on `http://localhost:3000`.
   - If the port is already in use, you can modify the start command in `package.json` or set a new port through an environment variable:
     ```bash
     yarn start --port 3001
     ```

5. **Test Website Startup**

   - After running the startup command, ensure that the website can be accessed normally through your browser.
   - If you encounter issues, try clearing the cache and restarting:
     ```bash
     yarn clear
     yarn start
     ```

6. **Additional Testing**

   - If you need to test in production mode, use the following commands:
     ```bash
     yarn build
     yarn serve
     ```
   - The website will be available at `http://localhost:3000` for testing.

Docusaurus allows us to directly write website content in Markdown and customize it using React. For detailed usage, refer to: [**Docusaurus Markdown Features**](https://docusaurus.io/docs/markdown-features)

## Writing Technical Documents

After completing the project development and achieving certain results, you may be eager to share your achievements with everyone. At this point, you can follow these steps to publish your technical documentation on our website:

Here, we use the `DocAligner` project as an example for creating technical documentation:

1. In the `docs` folder, create a new folder, for example, `docs/docaligner`.
2. In the folder, create an `index.md` file with the following content:

   ````markdown
   # DocAligner (Project Name)

   The core feature of this project is called "**Document Localization**". (Project Introduction)

   - [**DocAligner Github (Project's Github)**](https://github.com/DocsaidLab/DocAligner)

   ---

   ![title](./resources/title.jpg) (Project image, you can draw it yourself or ask GPT to generate)

   ---

   (Fixed code to display the project's card)

   ```mdx-code-block
   import DocCardList from '@theme/DocCardList';

   <DocCardList />
   ```
   ````

3. Create a `resources` folder within the folder to store project images.
4. Other technical documents, such as:

   - `docs/docaligner/quickstart.md`: Quick Start Guide
   - `docs/docaligner/installation.md`: Installation Guide
   - `docs/docaligner/advanced.md`: Advanced Usage
   - `docs/docaligner/model_arch`: Model Architecture
   - `docs/docaligner/benchmark`: Performance Evaluation
   - ... (Other content you wish to share)

5. After completion, submit a PR to the `main` branch and wait for review.

## Writing Blogs

During the development process, you may encounter various issues, both big and small.

Your problem is someone else's problem, and your solution is someone else's solution.

Therefore, we encourage you to write a blog about your problems and solutions and share them with others.

Below are the guidelines for writing a blog:

1. In the `blog` folder, find the corresponding year folder, such as `blog/2024`. If it does not exist, please create one.
2. In the year folder, create a folder with the date and title, such as `12-17-flexible-video-conversion-by-python`.
3. In the folder, create an `index.md` file with the following content, using the title from earlier as an example:

   ```markdown
   ---
   slug: flexible-video-conversion-by-python (The URL of the article)
   title: Batch Video Conversion
   authors: Zephyr (Must exist in authors.yml)
   image: /img/2024/1217.webp (Please generate with GPT and place it in the /static/img folder)
   tags: [Media-Processing, Python, ffmpeg]
   description: Using Python and ffmpeg to create a batch conversion process for specified formats.
   ---

   I received a batch of MOV video files, but the system does not support reading them. They need to be converted to MP4.

   So, I had to write some code myself.

    <!-- truncate --> (Summary end marker)

   ## Design Draft (Main content begins)
   ```

4. In the folder, create an `img` folder to store the blog's images.

   For aesthetics, you can use HTML syntax to align and resize images in the markdown document:

   ```html
   <div align="center"> (Center image)
   <figure style={{"width": "90%"}}> (Resize image)
   ![img description](./img/img_name.jpg)
   </figure>
   </div>
   ```

5. Once completed, submit a PR to the `main` branch and wait for review.
6. Finally, ensure your information is written into the `authors.yml` file so that we can correctly display your author details.

   For example, the current file content is as follows:

   ```yaml
   Zephyr: (Name used to locate the author)
     name: Zephyr (Name displayed on the webpage)
     title: Dosaid maintainer, Full-Stack AI Engineer (Author's title)
     url: https://github.com/zephyr-sh (Author's GitHub)
     image_url: https://github.com/zephyr-sh.png (Author's avatar)
     socials:
       github: "zephyr-sh" (Author's GitHub account)
   ```

   For more detailed settings, refer to: [**Docusaurus Blog Authors**](https://docusaurus.io/docs/blog#global-authors)

## Writing Paper Notes

Reading papers is our fate, and writing notes is to remind our future selves, because our memory is so fragile that we can hardly remember what we had for lunch yesterday, let alone the papers we’ve read.

If you also want to take notes, here are the guidelines for writing paper notes:

1. **Paper Selection Guide**:

   1. Choose papers published at top conferences such as CVPR, ICCV, NeurIPS, etc., to ensure the quality of the papers.
   2. If the paper does not meet the first criterion, choose papers with over 100 citations, indicating the paper has certain reference value.
   3. Avoid papers that require payment to access.
   4. Choose papers that are publicly available on ArXiv so that readers can access the full text.

2. **Paper Year**: There are several types of paper years, including the publication date on ArXiv, the conference date, and the paper's release date. For easy reference, we use the public date on ArXiv.
3. In the `papers` folder, find the corresponding branch for the paper, such as `papers/multimodality`. If it doesn't exist, please create one.
4. In the branch folder, create a folder that includes the year and title of the paper, for example, `2408-xgen-mm`.
5. Inside the newly created folder, create an `index.md` file, and place the paper images in the same-level `img` folder.
6. The standard format for writing paper notes is as follows:

   - **Title**: Format as year, month, and paper title.
   - **Authors**: Author names.
   - **Subtitle**: A catchy subtitle of your choice.
   - **Paper Link**: Full paper title and link.
   - **Problem Definition**: Summarize the problem defined by the authors.
   - **Solution**: Explain in detail how the authors solve the problem.
   - **Discussion**: Effectiveness or controversy of the solution or experimental results.
   - **Conclusion**: Summarize the key points of the paper.

7. A basic example is as follows:

   ```markdown
   ---
   title: "[24.08] xGen-MM" (Paper title, defined by industry norms or the authors themselves)
   authors: Zephyr (Author name, related definitions should be written in the `/blog/authors.json` file)
   ---

   ## Also known as BLIP-3 (A catchy subtitle)

   [**xGen-MM (BLIP-3): A Family of Open Large Multimodal Models**](https://arxiv.org/abs/2408.08872) (Full paper title and link)

   ---

   Just some casual chat.

   ## Problem Definition

   Summarize the problem defined by the authors.

   ## Solution

   Explain in detail how the authors solve the problem.

   ## Discussion

   Effectiveness or controversy of the solution or experimental results.

   ## Conclusion

   Summarize the key points of the paper.
   ```

8. Writing Guidelines:

   1. If you think the paper has issues, first suspect that you might have misunderstood or misinterpreted it.
   2. If you still think there are problems, first look for other references, don't make rash comments.
   3. Every paper has its trade-offs; focus on the inspirations from the paper, not its flaws.
   4. Maintain objectivity and neutrality, avoiding excessive criticism or praise.
   5. If you can’t find anything good about the paper, abandon writing the note; you chose the wrong paper.
   6. Be professional and avoid inappropriate language or images.

9. Once completed, submit a PR to the `main` branch and wait for review.

## Writing Code in Articles

We use the MDX syntax based on React, so you can write React code directly within articles.

Here’s a simple example: first, write a `HelloWorld.js` component:

```jsx
import React from "react";

const HelloWorld = () => {
  return <div>Hello, World!</div>;
};

export default HelloWorld;
```

Then, in the markdown article, import it. Even though it’s a `.md` file, it will be parsed as an MDX file, so you can write React code directly:

```mdx
import HelloWorld from "./HelloWorld";

# Title

<HelloWorld />

## Subtitle

Other content
```

## Multilingual Support

After writing the article, you suddenly realize that our website supports multiple languages, but you don't know any other languages!

Don't worry, generally speaking, someone named Zephyr will help you with this, but you can do it yourself:

1. Place your written article in the corresponding `i18n` folder, for example:

   - Articles for `docs` go into the `i18n/en/docusaurus-plugin-content-docs/current` folder,
   - Articles for `blog` go into the `i18n/en/docusaurus-plugin-content-blog/current` folder,
   - Articles for `papers` go into the `i18n/en/docusaurus-plugin-content-papers/current` folder.

   ***

   For Japanese, place the content in the `i18n/ja` folder, and for other languages, follow the same pattern.

2. Then, translate the content of the article in the `i18n` folder to the corresponding language. It is recommended to use GPTs for translation, then remove any unnecessary phrases and obvious mistakes.
3. Finally, submit a PR to the `main` branch and wait for the review.

## Finally

We must remind ourselves that, while many AI tools can now assist in generating articles, a truly captivating piece of writing must carry the author's unique personal style and emotional expression, which is something AI cannot fully replicate or replace.

The core operation of AI models is based on statistical models, generating content through maximum likelihood estimation. This means that the model tends to produce **mainstream, common syntax and sentence structures**, leading to content that often feels bland and similar in style. Therefore, excessive reliance on the model’s output can cause creators to lose their originality and soul, making the article lack depth and impact.

If you find yourself unable to create without AI assistance, or your mind goes completely blank, consider it a warning: you need to first strengthen your own writing skills and master basic creative techniques. Otherwise, you might end up becoming just a mouthpiece for the model.

In our experience, AI models are very suitable for tedious, repetitive tasks, such as spreadsheet data analysis and data aggregation, because these tasks typically have high standardization, requiring precision and efficiency but relatively low demands for creativity and flexibility:

> After all, spreadsheet data won’t become more interesting or change its experimental results just because you are creative.

Take "paper notes" as an example, where AI can assist us with difficult mathematical theories or help summarize experimental results and conclusions in the "## Discussion" section. However, it cannot replace our understanding and critical thinking of the paper, nor can it replace in-depth analysis. Similarly, for "project documentation," AI can quickly generate large amounts of technical documentation for the tedious details of function inputs and outputs, but for the design philosophy of the model, we still rely on our own expertise. As for "blogs," which completely depend on the creator’s style and thinking, AI’s help is even more limited.

Therefore, we should clearly define the scope of AI’s use and adjust our creative strategies accordingly to ensure the quality and uniqueness of our work. AI is a tool that can help us expand our perspectives and improve efficiency, not a tool to limit our thinking or style.

Remember, the truly moving words in writing come from the heart.

AI will only replace those who are unwilling to think. I think, therefore I am.

＊

2024 © Zephyr
