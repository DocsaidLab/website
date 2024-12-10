const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整
const RECENT_DAYS = 30;

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
    // 僅針對訊息中包含 "[A] Add article" 的 commit
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

  // 解析 YAML 頭部的 title
  let inYamlBlock = false;
  for (const line of lines) {
    if (line.trim() === '---') {
      inYamlBlock = !inYamlBlock;
      continue;
    }
    if (inYamlBlock && line.startsWith('title:')) {
      mainTitle = line.replace('title:', '').trim().replace(/^["']|["']$/g, ''); // 去掉引號
    }
    if (!inYamlBlock && line.startsWith('## ')) {
      subTitle = line.replace('## ', '').trim();
      break;
    }
  }

  // 若未找到主標題則以檔名替代
  if (!mainTitle) {
    mainTitle = path.basename(article.filePath, path.extname(article.filePath));
  }

  const combinedTitle = subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
  return combinedTitle;
}

async function writeRecentUpdatesData(targetDir, articles) {
  if (articles.length === 0) {
    console.log('No articles to write for', targetDir);
    return;
  }

  const data = articles.map(a => ({
    date: a.date,
    link: a.link,
    combinedTitle: a.combinedTitle,
  }));

  const outputFile = path.join(targetDir, 'recent_updates_data.json');
  await fs.writeJson(outputFile, data, { spaces: 2 });
  console.log(`✅ Generated recent updates data at: ${outputFile}\nPlease make sure this file is in .gitignore if you don't want it tracked.`);
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

    // 對每個 TARGET_FILE 產生獨立的 recent_updates_data.json
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
      await writeRecentUpdatesData(targetDir, filteredArticles);
    }

    console.log('\nAll TARGET_FILES have been processed. The recent updates are now stored in their respective `recent_updates_data.json` files.');
  } catch (error) {
    console.error('❌ Error：', error);
  }
})();
