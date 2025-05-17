// recent-updates.js
// 用途：抓取最近新增的 article (.md/.mdx) 並為各語系 docs 目錄產生 recent_updates_data.json

const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整：抓取最近 N 天的新增檔案
const RECENT_DAYS = 30;

// 對應到各語系的 intro.md 檔案所在資料夾
const TARGET_FILES = [
  path.join(__dirname, '..', '..', 'papers', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];

async function getAddedArticles(sinceOption) {
  // 直接用 git log + diff-filter=A 找出所有新增檔案
  const raw = await git.raw([
    'log',
    `--since=${sinceOption}`,
    '--diff-filter=A',
    '--name-only',
    '--pretty=format:'
  ]);

  const files = raw
    .split('\n')
    .map(f => f.trim())
    .filter(f => f && (f.endsWith('.md') || f.endsWith('.mdx')));

  const uniqueFiles = Array.from(new Set(files));

  const addedArticles = [];
  for (const filePath of uniqueFiles) {
    const fullPath = path.resolve(__dirname, '..', '..', filePath);
    // 查這個檔案最後一次 commit 的日期（也就是新增日期）
    const logForFile = await git.log({
      file: filePath,
      '--max-count': 1,
      '--format': '%cI'
    });
    const date = logForFile.latest ? logForFile.latest.date : null;
    addedArticles.push({ filePath, fullPath, date });
  }

  console.log(`Found ${addedArticles.length} added article(s) in the last ${sinceOption}.`);
  return addedArticles;
}

async function extractTitleInfo(article) {
  if (!(await fs.pathExists(article.fullPath))) {
    console.warn(`File does not exist: ${article.fullPath}`);
    return null;
  }

  const content = await fs.readFile(article.fullPath, 'utf-8');
  const lines = content.split('\n');

  let mainTitle = '';
  let subTitle = '';
  let inYaml = false;

  for (const line of lines) {
    if (line.trim() === '---') {
      inYaml = !inYaml;
      continue;
    }
    if (inYaml && line.startsWith('title:')) {
      mainTitle = line.replace('title:', '').trim().replace(/^["']|["']$/g, '');
    }
    if (!inYaml && line.startsWith('## ')) {
      subTitle = line.replace('## ', '').trim();
      break;
    }
  }

  if (!mainTitle) {
    mainTitle = path.basename(article.filePath, path.extname(article.filePath));
  }

  return subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
}

async function writeRecentUpdatesData(targetDir, articles) {
  if (articles.length === 0) {
    console.log(`No new articles for ${targetDir}, skipping.`);
    return;
  }

  const data = articles.map(a => ({
    date: a.date ? a.date.split('T')[0] : '',
    link: a.link,
    combinedTitle: a.combinedTitle,
  }));

  const outputFile = path.join(targetDir, 'recent_updates_data.json');
  await fs.writeJson(outputFile, data, { spaces: 2 });
  console.log(`✅ Generated recent updates data at: ${outputFile}`);
  console.log('   (請確認該檔案已加入 .gitignore)');
}

(async () => {
  try {
    const sinceOption = `${RECENT_DAYS} days ago`;
    console.log(`🔍 Scanning commits since: ${sinceOption}`);

    const addedArticles = await getAddedArticles(sinceOption);
    if (addedArticles.length === 0) {
      console.log('No added articles found—nothing to update.');
      return;
    }

    for (const targetFile of TARGET_FILES) {
      const targetDir = path.dirname(targetFile);

      // 篩選出屬於此語系資料夾的檔案
      const filtered = [];
      for (const art of addedArticles) {
        if (!art.fullPath.startsWith(targetDir)) continue;

        const combinedTitle = await extractTitleInfo(art);
        if (!combinedTitle) continue;

        const rel = './' + path.relative(targetDir, art.fullPath).replace(/\\/g, '/');
        filtered.push({
          combinedTitle,
          link: rel,
          date: art.date,
        });
      }

      // 按日期由新到舊排序
      filtered.sort((a, b) => new Date(b.date) - new Date(a.date));

      await writeRecentUpdatesData(targetDir, filtered);
    }

    console.log('\n🎉 All TARGET_FILES processed. Done!');
  } catch (err) {
    console.error('❌ Error:', err);
  }
})();
