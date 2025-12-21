/**
 * recent-updates.js
 *
 * 功能：
 *  1. 取得各語系 docs 目錄下的所有 .md/.mdx 檔案。
 *  2. 透過 git 歷史查詢每個檔案「第一次被加入 repo」的 commit 時間（作為 publishTime）。
 *  3. 依 publishTime 由新到舊排序，取最新 N 篇，寫入 `recent_updates_data.json`。
 *
 * 註：這裡以「首次加入 repo 的時間」作為文章日期來源，避免依賴 front matter 的 date 欄位。
 */

const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();

// 可根據需求調整：顯示最新 N 篇文章
const MAX_ARTICLES = 10;

const REPO_ROOT = path.join(__dirname, '..', '..');

// 對應到各語系 (或其他) intro.md 檔案所在資料夾
const TARGET_FILES = [
  path.join(REPO_ROOT, 'papers', 'intro.md'),
  path.join(REPO_ROOT, 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(REPO_ROOT, 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];

/* -------------------------------------------------------------------------- */
/*                              1. 取得文章清單                                 */
/* -------------------------------------------------------------------------- */

function isMarkdownFile(filePath) {
  return filePath.endsWith('.md') || filePath.endsWith('.mdx');
}

/* -------------------------------------------------------------------------- */
/*                 2. 查詢檔案「第一次被加入 repo」的 commit 時間 (ISO)          */
/* -------------------------------------------------------------------------- */

async function resolveLogBaseRef() {
  const preferredRef = process.env.RECENT_UPDATES_BASE_REF || 'HEAD';
  try {
    await git.raw(['rev-parse', '--verify', preferredRef]);
    return preferredRef;
  } catch {
    return 'HEAD';
  }
}

async function getEarliestAddDatesByDir(dirRelativeToRepo, baseRef) {
  /**
   * 使用一次 git log 掃描整個資料夾，避免逐檔案呼叫 git。
   * 以 --reverse 由舊到新，第一次看到檔案即為最早 add 時間。
   */
  const rawLog = await git
    .raw([
      'log',
      baseRef,
      '--diff-filter=A',
      '--name-only',
      '--pretty=format:%cI',
      '--reverse',
      '--',
      dirRelativeToRepo,
    ])
    .catch(() => '');

  const lines = rawLog.split('\n');
  const earliestMap = new Map();
  let currentIso = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // --pretty=format:%cI 會輸出 ISO 8601
    if (/^\d{4}-\d{2}-\d{2}T/.test(trimmed)) {
      currentIso = trimmed;
      continue;
    }

    if (!currentIso) continue;
    if (!isMarkdownFile(trimmed)) continue;
    if (!earliestMap.has(trimmed)) {
      earliestMap.set(trimmed, currentIso);
    }
  }

  return earliestMap;
}

async function getEarliestCommitDateISO(filePathRelativeToRepo, baseRef) {
  const rawLog = await git
    .raw([
      'log',
      baseRef,
      '--follow',
      '--format=%cI',
      '--reverse',
      '--max-count=1',
      '--',
      filePathRelativeToRepo,
    ])
    .catch(() => '');

  const isoDate = rawLog.trim();
  return isoDate || null;
}

/* -------------------------------------------------------------------------- */
/*                              3. 解析文章標題                                 */
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
  await fs.ensureDir(targetDir);

  if (articles.length > 0) {
    console.log(`\n📄 Articles under ${targetDir}:`);
    for (const a of articles) {
      console.log(`   • ${a.publishTime}  →  ${a.combinedTitle}`);
    }
  } else {
    console.log(`ℹ️  No new articles under ${targetDir}. Writing empty recent_updates_data.json.`);
  }

  const data = articles.map((a) => ({
    date: a.publishTime.split('T')[0],
    publishTime: a.publishTime,
    link: a.link,
    combinedTitle: a.combinedTitle,
  }));

  const outputFile = path.join(targetDir, 'recent_updates_data.json');
  await fs.writeJson(outputFile, data, { spaces: 2 });
  console.log(`✅  JSON generated: ${outputFile}`);
}

async function listMarkdownFilesRecursively(dirAbs) {
  const results = [];
  const stack = [dirAbs];

  while (stack.length > 0) {
    const current = stack.pop();
    const entries = await fs.readdir(current, { withFileTypes: true }).catch(() => []);

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!isMarkdownFile(entry.name)) continue;
      results.push(fullPath);
    }
  }

  return results;
}

/* -------------------------------------------------------------------------- */
/*                                   主流程                                    */
/* -------------------------------------------------------------------------- */

(async () => {
  try {
    const baseRef = await resolveLogBaseRef();
    console.log(`\n=== Generating recent updates from [${baseRef}] (latest ${MAX_ARTICLES}) ===`);

    // 針對每個語系資料夾：找出最新 N 篇文章並輸出 JSON
    for (const introFile of TARGET_FILES) {
      const targetDir = path.dirname(introFile);
      const dirRelativeToRepo = path.relative(REPO_ROOT, targetDir).replace(/\\/g, '/');
      const introFileRel = path.relative(REPO_ROOT, introFile).replace(/\\/g, '/');

      const existingFilesAbs = await listMarkdownFilesRecursively(targetDir);
      const existingFilesRel = existingFilesAbs
        .map((p) => path.relative(REPO_ROOT, p).replace(/\\/g, '/'))
        .filter((p) => p.startsWith(dirRelativeToRepo + '/'))
        .filter((p) => p !== introFileRel);

      const earliestMap = await getEarliestAddDatesByDir(dirRelativeToRepo, baseRef);

      const candidates = [];
      for (const fileRel of existingFilesRel) {
        let publishTime = earliestMap.get(fileRel) || null;
        if (!publishTime) {
          publishTime = await getEarliestCommitDateISO(fileRel, baseRef);
        }
        if (!publishTime) continue;
        candidates.push({ fileRel, publishTime });
      }

      candidates.sort((a, b) => new Date(b.publishTime) - new Date(a.publishTime));

      const latest = [];
      for (const { fileRel, publishTime } of candidates) {
        if (latest.length >= MAX_ARTICLES) break;

        const fullPath = path.join(REPO_ROOT, fileRel);
        const combinedTitle = await extractTitleInfo(fullPath, fileRel);
        if (!combinedTitle) continue;

        const relLink = './' + path.relative(targetDir, fullPath).replace(/\\/g, '/');
        latest.push({
          combinedTitle,
          link: relLink,
          publishTime,
        });
      }

      await writeRecentUpdatesData(targetDir, latest);
    }

    console.log('\n🎉  All done.\n');
  } catch (err) {
    console.error('❌  Error:', err);
  }
})();
