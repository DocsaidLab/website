// recent-updates.js
// 用途：抓取最近 N 天內，在 main 上最終出現的新文章 (.md / .mdx)。
// 無論是直接在 main 上 commit 還是從 PR merge 到 main，只要檔案出現在 main，就能偵測到。
// 然後為各語系 docs 產生 recent_updates_data.json。

const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整：抓取最近 N 天的新增檔案
const RECENT_DAYS = 30;

// 對應到各語系 (或其他) intro.md 檔案所在資料夾
const TARGET_FILES = [
  path.join(__dirname, '..', '..', 'papers', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];

// 如果你的主要分支不是叫 "main"，請自行改成 "master"、"dev" 或其他名稱
const MAIN_BRANCH = 'main';

/**
 * 第1步：找出「在 main 分支、最近 N 天內產生的所有 commit (含合併)」，
 *        解析每個 commit (或其 merge parent) 的 "A\t" 檔案，以得知有哪些 .md / .mdx 新增了。
 *
 *        這樣就能同時偵測到「直接在 main 上 commit」和「從其他分支 merge 回 main」帶進來的檔案。
 */
async function getAddedArticles(sinceOption) {
  // 使用 --name-status -m：對 merge commit 也會列出來自各 parent 的 diff。
  // --pretty=format:%H|%cI：方便同時抓取 "commit SHA" 與 "commit date (ISO)"。
  const rawLog = await git.raw([
    'log',
    MAIN_BRANCH,
    `--since=${sinceOption}`,
    '--name-status',
    '-m',
    `--pretty=format:%H|%cI`,
  ]);

  // 逐行解析
  const lines = rawLog.split('\n');
  const addedArticlesMap = new Map();
  // 為了避免重複列到同一檔案 (可能出現在多個 merge parent)，用 Map 來去重
  // key = 檔案路徑, value = commit date (最後一次看到它出現時的 date)

  let currentCommit = null;      // SHA
  let currentCommitIsoDate = ''; // ISO string

  for (const line of lines) {
    // 如果符合 commit 的標識(格式: SHA|ISO8601)
    if (/^[0-9a-f]{40}\|/.test(line)) {
      const [sha, iso] = line.split('|');
      currentCommit = sha;
      currentCommitIsoDate = iso; // commit 的 ISO 時間
      continue;
    }

    // 如果是檔案變動資訊 (e.g. "A\tpath/to/file.md")
    if (line.startsWith('A\t')) {
      const filePath = line.substring(2).trim();
      if (filePath.endsWith('.md') || filePath.endsWith('.mdx')) {
        // 先記錄下來 (或覆蓋)
        addedArticlesMap.set(filePath, currentCommitIsoDate);
      }
    }
  }

  // 整理成一個陣列
  const addedArticles = [];
  for (const [filePath, isoDate] of addedArticlesMap.entries()) {
    addedArticles.push({
      filePath,
      fullPath: path.resolve(__dirname, '..', '..', filePath),
      // 這裡先存「被檢測到在 main 的 commit 時間 (ISO)」
      // 稍後我們還會再查真正 "首次" 新增的日期
      foundDate: isoDate,
      date: null,
    });
  }

  console.log(`\n🔍 Found ${addedArticles.length} newly added .md/.mdx in the last ${sinceOption} on branch [${MAIN_BRANCH}].`);
  return addedArticles;
}

/**
 * 第2步：查詢檔案 **真正首次** (earliest) 在整個 repo 被加入 (Add) 的 commit 日期 (只取 YYYY-MM-DD)。
 *        (若你只想紀錄「它何時被 merge 進 main」的時間，可用 addedArticles[i].foundDate 即可。
 *         不過通常都想要知道它在 repo 第一次出現的時間，所以還是做這一步。)
 */
async function getFileFirstAddedDate(filePath) {
  const rawLog = await git.raw([
    'log',
    '--diff-filter=A',
    '--format=%cI',
    '--reverse',
    '--max-count=1',
    filePath
  ]).catch(() => null);

  if (!rawLog) return null;
  const isoDate = rawLog.trim();
  if (!isoDate) return null;

  // 只要 'YYYY-MM-DD'
  return isoDate.split('T')[0];
}

async function extractTitleInfo(fullPath, filePath) {
  if (!(await fs.pathExists(fullPath))) {
    console.warn(`File does not exist: ${fullPath}`);
    return null;
  }

  const content = await fs.readFile(fullPath, 'utf-8');
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

  // 如果 YAML 沒有 title，就用檔名替代
  if (!mainTitle) {
    mainTitle = path.basename(filePath, path.extname(filePath));
  }

  return subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
}

/**
 * 第3步：把最終的資料寫到 recent_updates_data.json
 */
async function writeRecentUpdatesData(targetDir, articles) {
  if (articles.length === 0) {
    console.log(`No new articles for ${targetDir}, skipping.`);
    return;
  }

  console.log(`\n📄 Articles under ${targetDir}:`);
  articles.forEach(a => {
    console.log(`  • ${a.date} → ${a.combinedTitle}`);
  });

  const data = articles.map(a => ({
    date: a.date || '',
    link: a.link,
    combinedTitle: a.combinedTitle,
  }));

  const outputFile = path.join(targetDir, 'recent_updates_data.json');
  await fs.writeJson(outputFile, data, { spaces: 2 });
  console.log(`✅ Generated recent_updates_data.json at: ${outputFile}`);
  console.log('   (請確認該檔案已加入 .gitignore)');
}

(async () => {
  try {
    const sinceOption = `${RECENT_DAYS} days ago`;
    console.log(`\n=== Start scanning commits on [${MAIN_BRANCH}] since: ${sinceOption} ===`);

    // 第1步：先抓出「在 main 分支、最近 N 天內的 commit (含合併)」中，被偵測為 'A' 的檔案
    const addedArticles = await getAddedArticles(sinceOption);
    if (!addedArticles || addedArticles.length === 0) {
      console.log('No added articles found—nothing to update.\n');
      return;
    }

    // 第2步：再到整個 repo 找它真正的首次新增日期
    for (const art of addedArticles) {
      art.date = await getFileFirstAddedDate(art.filePath);
    }

    // 第3步：針對每個 targetDir 寫出資料檔
    for (const targetFile of TARGET_FILES) {
      const targetDir = path.dirname(targetFile);
      const filtered = [];

      for (const art of addedArticles) {
        if (!art.date) continue;
        if (!art.fullPath.startsWith(targetDir)) continue;

        const combinedTitle = await extractTitleInfo(art.fullPath, art.filePath);
        if (!combinedTitle) continue;

        const rel = './' + path.relative(targetDir, art.fullPath).replace(/\\/g, '/');
        filtered.push({
          combinedTitle,
          link: rel,
          date: art.date,
        });
      }

      // 按發佈時間 (YYYY-MM-DD) 由新到舊排序
      filtered.sort((a, b) => new Date(b.date) - new Date(a.date));

      await writeRecentUpdatesData(targetDir, filtered);
    }

    console.log('\n🎉 All TARGET_FILES processed. Done!\n');
  } catch (err) {
    console.error('❌ Error:', err);
  }
})();
