const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整
const RECENT_DAYS = 30;

// 這裡是目標檔案，程式不再直接修改這些檔案
// 而是在同目錄下產生 `recent_updates.mdx` 檔案
const TARGET_FILES = [
  path.join(__dirname, '..', '..', 'papers', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];

async function getAddedArticles(sinceOption) {
  const log = await git.log({'--since': sinceOption});
  if (log.total === 0) {
    console.log('No commits found in the given date range.');
    return [];
  }
  console.log(`Found ${log.total} commits.`);

  const addedArticles = [];
  for (const commit of log.all) {
    if (/\[A\] Add article/i.test(commit.message)) {
      const diff = await git.show(['--name-status', commit.hash]);
      const lines = diff.split('\n');
      for (const line of lines) {
        if (line.startsWith('A\t')) {
          const filePath = line.substring(2);
          if (filePath.endsWith('.md') || filePath.endsWith('.mdx')) {
            const articleFullPath = path.resolve(__dirname, '..', '..', filePath);
            addedArticles.push({ filePath, fullPath: articleFullPath, date: commit.date });
          }
        }
      }
    }
  }

  console.log('Total added articles found:', addedArticles.length);
  return addedArticles;
}

async function extractTitleInfo(article) {
  if (!(await fs.pathExists(article.fullPath))) {
    console.log(`File does not exist: ${article.fullPath}`);
    return null;
  }

  const content = await fs.readFile(article.fullPath, 'utf-8');
  const lines = content.split('\n');

  let mainTitle = '';
  let subTitle = '';

  for (const line of lines) {
    if (line.startsWith('# ')) {
      mainTitle = line.replace('# ', '').trim();
    } else if (line.startsWith('## ')) {
      subTitle = line.replace('## ', '').trim();
    }

    if (mainTitle && subTitle) break;
  }

  // 若未找到主標題則以檔名替代
  if (!mainTitle) {
    mainTitle = path.basename(article.filePath, path.extname(article.filePath));
  }

  const combinedTitle = subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
  return combinedTitle;
}

/**
 * 將文章清單產生的 <Timeline> 內容寫入不被 Git 追蹤的檔案中 (recent_updates.md)
 */
async function writeRecentUpdates(targetDir, articles) {
  if (articles.length === 0) {
    console.log('No articles to write for', targetDir);
    return;
  }

  // 組合要插入的 markdown 片段
  let markdownContent = 'import { Timeline } from "antd";\n\n<Timeline mode="alternate">\n';
  for (const article of articles) {
    markdownContent += `  <Timeline.Item label="${article.date}">\n`;
    markdownContent += `    [${article.combinedTitle}](${article.link})\n`;
    markdownContent += `  </Timeline.Item>\n`;
  }
  markdownContent += '</Timeline>';

  const outputFile = path.join(targetDir, 'recent_updates.mdx');
  await fs.writeFile(outputFile, markdownContent, 'utf-8');
  console.log(`✅ Generated recent updates at: ${outputFile}\nPlease make sure this file is in .gitignore to avoid tracking.`);
}

(async () => {
  try {
    const sinceOption = `${RECENT_DAYS} days ago`;
    console.log('Since option:', sinceOption);

    // 取得最近 N 天 commits 中的已新增文章清單
    const addedArticles = await getAddedArticles(sinceOption);
    if (addedArticles.length === 0) {
      console.log('No added articles found, no updates needed.');
      return;
    }

    // 對每個 TARGET_FILE 產生獨立的 recent_updates.md
    for (const TARGET_FILE of TARGET_FILES) {
      const targetDir = path.dirname(TARGET_FILE);

      const filteredArticles = [];
      for (const article of addedArticles) {
        if (!article.fullPath.startsWith(targetDir)) continue;

        const combinedTitle = await extractTitleInfo(article);
        if (!combinedTitle) continue;

        const relativeLink = './' + path.relative(targetDir, article.fullPath).replace(/\\/g, '/');
        const date = article.date.split('T')[0];
        filteredArticles.push({ combinedTitle, link: relativeLink, date });
      }

      filteredArticles.sort((a, b) => new Date(b.date) - new Date(a.date));
      await writeRecentUpdates(targetDir, filteredArticles);
    }

    console.log('\nAll TARGET_FILES have been processed. The recent updates are now stored in their respective `recent_updates.mdx` files.');
    console.log('Remember to add `recent_updates.mdx` to your `.gitignore` file so it won\'t be tracked by git.');
  } catch (error) {
    console.error('❌ Error：', error);
  }
})();
