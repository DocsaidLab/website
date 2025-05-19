/**
 * recent-updates.js
 *
 * 功能：
 *  1. 偵測「最近 N 天內，最終合併/出現在 main 分支」的 .md 或 .mdx 檔案（包含從其他分支 PR merge 進來）。
 *  2. 追溯每個檔案「在整個 repo 第一次被加入」的 commit 日期（作為 publishTime）。
 *  3. 幫各語系的 docs 目錄產生 `recent_updates_data.json`（依照檔案所在資料夾）。
 *
 *  - 使用 `git log main -m --name-status --since=... --pretty=format:%H|%cI` 去擴展 merge commit，
 *    以偵測合併帶進來的新檔案。
 *  - 另做第二階段的最早 `Add` commit 查詢：`git log --diff-filter=A --reverse --max-count=1 filePath`
 *    取得檔案在 repo 第一次加入的 ISO 時間，再格式化成 YYYY-MM-DD。
 */

const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整：抓取最近 N 天的新增檔案
const RECENT_DAYS = 30;

// 若主要分支不是 "main"，請自行修改
const MAIN_BRANCH = 'main';

// 對應到各語系 (或其他) intro.md 檔案所在資料夾
const TARGET_FILES = [
  path.join(__dirname, '..', '..', 'papers', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];

/* -------------------------------------------------------------------------- */
/*                     1. 抓「main 分支最近 N 天內的新增檔案」                    */
/* -------------------------------------------------------------------------- */

async function getAddedArticles(sinceOption) {
  /**
   * 這裡的關鍵：
   *  1. `main`：只看 main 分支的歷史 (包含合併 commit)。
   *  2. `-m`：對 merge commit 也會列出來自各個 parent 的差異。
   *  3. `--name-status`：用 "A\t" / "M\t" 等標識檔案操作類型。
   *  4. `--pretty=format:%H|%cI`：顯示 commit SHA | ISO 時間，方便分行解析。
   */
  const rawLog = await git.raw([
    'log',
    MAIN_BRANCH,
    `--since=${sinceOption}`,
    '--name-status',
    '-m',
    '--pretty=format:%H|%cI',
  ]);

  const lines = rawLog.split('\n');

  // 用 Map 來避免重複記錄同一個檔案
  // key = 檔案路徑, value = commit的ISO時間 (最後一次在 log 裡看到它以"A"出現的那個commit)
  const addedMap = new Map();

  let currentCommitIso = ''; // 目前解析到的 commit 時間 (ISO)

  for (const line of lines) {
    // 如果是一行 commit 資訊 (格式: "40位SHA|YYYY-MM-DDTHH:mm:ssZ")
    if (/^[0-9a-f]{40}\|/.test(line)) {
      const [sha, isoDate] = line.split('|');
      currentCommitIso = isoDate.trim();
      continue;
    }

    // 若是檔案變動行 (如 "A\tpath/to/file.md")
    if (line.startsWith('A\t')) {
      const filePath = line.substring(2).trim();
      if (filePath.endsWith('.md') || filePath.endsWith('.mdx')) {
        // 記下 (或覆蓋) → 表示在這個 commit 看到它被新增
        addedMap.set(filePath, currentCommitIso);
      }
    }
  }

  // 整理回陣列
  const addedArticles = [];
  for (const [filePath, isoDate] of addedMap.entries()) {
    const fullPath = path.resolve(__dirname, '..', '..', filePath);
    addedArticles.push({
      filePath,
      fullPath,
      // 先存「在 main 偵測到的 commit」的時間（如需即可使用）
      foundTime: isoDate.split('T')[0],
      // 真正的第一次加入時間（待第二階段查詢）
      publishTime: null,
    });
  }

  console.log(`\n🔍 Detected ${addedArticles.length} newly added MD/MDX under [${MAIN_BRANCH}] since ${sinceOption}.`);
  return addedArticles;
}

/* -------------------------------------------------------------------------- */
/*       2. 查詢檔案「在整個 repo 第一次被加入」的 commit 時間 (ISO) + 去除時區   */
/* -------------------------------------------------------------------------- */

async function getEarliestAddDate(filePath) {
  /**
   * 透過 --diff-filter=A + --reverse + --max-count=1 找到該檔案最早 (earliest) 的 Add commit
   * - %cI：取得 ISO 8601 的提交時間。
   */
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

  // 回傳完整ISO字串 (e.g. "2025-05-18T11:22:33+08:00")
  return isoDate.split('T')[0];
}

/* -------------------------------------------------------------------------- */
/*                              3. 解析文章標題                                */
/* -------------------------------------------------------------------------- */

async function extractTitleInfo(fullPath, filePath) {
  if (!(await fs.pathExists(fullPath))) {
    console.warn(`🚫 File not found: ${fullPath}`);
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

  // 若 YAML 沒有 title，就用檔名 (去掉副檔名)
  if (!mainTitle) {
    mainTitle = path.basename(filePath, path.extname(filePath));
  }

  return subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
}

/* -------------------------------------------------------------------------- */
/*                 4. 寫入 recent_updates_data.json 到各目錄                   */
/* -------------------------------------------------------------------------- */

async function writeRecentUpdatesData(targetDir, articles) {
  if (articles.length === 0) {
    console.log(`ℹ️  No new articles under ${targetDir}. Skipped.`);
    return;
  }

  console.log(`\n📄 Articles under ${targetDir}:`);
  for (const a of articles) {
    // date = YYYY-MM-DD, publishTime = 全部(ISO)
    console.log(`   • ${a.publishTime}  →  ${a.combinedTitle}`);
  }

  // 寫入 JSON
  const data = articles.map(a => ({
    date: a.publishTime.split('T')[0], // 純日期 (YYYY-MM-DD)
    publishTime: a.publishTime,        // 完整 ISO timestamp
    link: a.link,
    combinedTitle: a.combinedTitle,
  }));

  const outputFile = path.join(targetDir, 'recent_updates_data.json');
  await fs.writeJson(outputFile, data, { spaces: 2 });
  console.log(`✅  JSON generated: ${outputFile}`);
}

/* -------------------------------------------------------------------------- */
/*                                   主流程                                    */
/* -------------------------------------------------------------------------- */

(async () => {
  try {
    const sinceOption = `${RECENT_DAYS} days ago`;
    console.log(`\n=== Scanning commits on [${MAIN_BRANCH}] since: ${sinceOption} ===`);

    // 第1步：先從 main 分支 (含 merge commit) 找「最近 N 天的新增(A)」檔案
    const addedArticles = await getAddedArticles(sinceOption);
    if (!addedArticles || addedArticles.length === 0) {
      console.log('No newly added articles found—done.\n');
      return;
    }

    // 第2步：對於每個檔案，找它在整個 repo "第一次被加入" 的 commit 時間
    for (const art of addedArticles) {
      const earliestISO = await getEarliestAddDate(art.filePath);
      // 若找不到就用 foundTime 退而求其次
      art.publishTime = earliestISO || art.foundTime;
    }

    // 第3步：針對每個語系資料夾，歸納出屬於該資料夾的新檔案
    for (const introFile of TARGET_FILES) {
      const targetDir = path.dirname(introFile);
      const articlesForDir = [];

      for (const art of addedArticles) {
        if (!art.publishTime) continue;
        if (!art.fullPath.startsWith(targetDir)) continue;

        const combinedTitle = await extractTitleInfo(art.fullPath, art.filePath);
        if (!combinedTitle) continue;

        // 相對連結
        const relLink = './' + path.relative(targetDir, art.fullPath).replace(/\\/g, '/');

        articlesForDir.push({
          combinedTitle,
          link: relLink,
          publishTime: art.publishTime,
        });
      }

      // 依發佈時間 新→舊 排序
      articlesForDir.sort((a, b) => new Date(b.publishTime) - new Date(a.publishTime));

      // 第4步：輸出到 recent_updates_data.json
      await writeRecentUpdatesData(targetDir, articlesForDir);
    }

    console.log('\n🎉  All done.\n');
  } catch (err) {
    console.error('❌  Error:', err);
  }
})();
