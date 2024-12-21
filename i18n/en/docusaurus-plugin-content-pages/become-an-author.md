# Dear Sir/Madam

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

   - We recommend using Node.js v16 or above.
   - If you haven't installed `nvm`, please refer to the [official nvm documentation](https://github.com/nvm-sh/nvm#installing-and-updating) for installation.
   - After installation, run the following commands to check and set the Node.js version:
     ```bash
     nvm install 16
     nvm use 16
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

## Finally

We want to remind you that although many AI tools can help generate articles, a truly engaging article always carries the author's unique style and emotional expression, which is something AI cannot fully replicate or replace.

AI operates based on statistical models and generates content using maximum likelihood estimation. This means AI tends to produce more generic and common syntax and sentence structures, which often leads to bland and similar content. Therefore, over-relying on AI may cause creators to lose their individuality and soul, making their articles lack depth and impact.

If you find that you cannot create without AI assistance or feel mentally blocked, this is a warning: it indicates that you need to solidify your writing skills and master the basic techniques of creation. Otherwise, you will merely be a mouthpiece for AI models rather than a true author.

In our experience, AI is excellent for tedious and repetitive tasks, such as table data analysis and data aggregation, because these tasks are highly standardized, requiring high accuracy and efficiency but relatively low creativity and flexibility:

> After all, table data will not become more interesting or change its experimental results just because you are creative.

For example, in "Paper Notes," AI can only assist in the "## Discussion" section or help us deal with difficult mathematical theories, but it cannot replace our understanding and thinking of the paper. As for "Project Documentation" and "Blogs," which entirely depend on the creator's style and thinking, AI’s help is relatively limited.

Therefore, we should clearly define the scope of AI usage and adjust our creative strategies accordingly to ensure the quality and uniqueness of the articles. Use AI's power wisely, but more importantly, retain your own creativity and desire for expression.

AI should be a tool that helps expand our vision and improve efficiency, not limit our thinking or style. In the writing process, remember that truly moving words come from the heart of a person, not cold algorithms. Let AI be your capable assistant, but you, as the creator, are the true master of the work.

AI will only replace those who refuse to think. I think, therefore I am.

＊

2024 © Zephyr
