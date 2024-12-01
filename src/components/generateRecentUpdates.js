const simpleGit = require('simple-git');
const fs = require('fs-extra');
const path = require('path');

const git = simpleGit();
const RECENT_DAYS = 30;

const TARGET_FILES = [
  path.join(__dirname, '..', '..', 'papers', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'en', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
  path.join(__dirname, '..', '..', 'i18n', 'ja', 'docusaurus-plugin-content-docs-papers', 'current', 'intro.md'),
];


(async () => {
  try {
    const sinceOption = `${RECENT_DAYS} days ago`;
    console.log('Since option:', sinceOption);
    const log = await git.log({
      '--since': sinceOption,
    });

    if (log.total === 0) {
      console.log('No commits found in the given date range.');
      return;
    } else {
      console.log(`Found ${log.total} commits.`);
    }

    const addedArticles = [];
    for (const commit of log.all) {
      if (/\[A\] Add article/i.test(commit.message)) {
        const diff = await git.show(['--name-status', commit.hash]);
        const lines = diff.split('\n');
        for (const line of lines) {
          if (line.startsWith('A\t')) {
            const filePath = line.substring(2);
            const articleFullPath = path.resolve(__dirname, '..', '..', filePath);
            if (filePath.endsWith('.md') || filePath.endsWith('.mdx')) {
              addedArticles.push({
                filePath,
                fullPath: articleFullPath,
                date: commit.date,
              });
            }
          }
        }
      }
    }

    console.log('Total added articles found:', addedArticles.length);
    for (const TARGET_FILE of TARGET_FILES) {
      console.log('\nProcessing TARGET_FILE:', TARGET_FILE);

      const targetDir = path.dirname(TARGET_FILE);
      const baseArticleDir = path.resolve(targetDir);
      console.log('Base article directory:', baseArticleDir);

      const articles = [];
      for (const article of addedArticles) {
        const articleFullPath = article.fullPath;

        if (!articleFullPath.startsWith(baseArticleDir)) {
          // console.log(`Skipping file not in target directory: ${articleFullPath}`);
          continue;
        }

        if (await fs.pathExists(articleFullPath)) {
          const content = await fs.readFile(articleFullPath, 'utf-8');
          const lines = content.split('\n');
          let mainTitle = '';
          let subTitle = '';

          for (const line of lines) {
            if (line.startsWith('# ')) {
              mainTitle = line.replace('# ', '').trim();
            } else if (line.startsWith('## ')) {
              subTitle = line.replace('## ', '').trim();
            }

            if (mainTitle && subTitle) {
              break;
            }
          }

          if (!mainTitle) {
            mainTitle = path.basename(article.filePath, path.extname(article.filePath));
          }

          const combinedTitle = subTitle ? `${mainTitle}: ${subTitle}` : mainTitle;
          const relativeLink = './' + path.relative(targetDir, articleFullPath).replace(/\\/g, '/');

          const date = article.date.split('T')[0]; // 使用 commit.date
          articles.push({ combinedTitle, link: relativeLink, date });
        } else {
          console.log(`File does not exist: ${articleFullPath}`);
        }
      }

      console.log('Articles to be updated for this TARGET_FILE:', articles.length);
      articles.sort((a, b) => new Date(b.date) - new Date(a.date));

      if (articles.length === 0) {
        console.log('No articles to update for this TARGET_FILE.');
        continue;
      }

      let markdownContent = '<Timeline mode="alternate">\n';

      articles.forEach((article) => {
        markdownContent += `  <Timeline.Item label="${article.date}">\n`;
        markdownContent += `    [${article.combinedTitle}](${article.link})\n`;
        markdownContent += `  </Timeline.Item>\n`;
      });

      markdownContent += '</Timeline>';

      let targetFileContent = await fs.readFile(TARGET_FILE, 'utf-8');

      const SECTION_MARKER_START = '<!-- RECENT_UPDATES_START -->';
      const SECTION_MARKER_END = '<!-- RECENT_UPDATES_END -->';

      const newContent = `${SECTION_MARKER_START}\n\n${markdownContent}\n\n${SECTION_MARKER_END}`;

      const regex = new RegExp(
        `${SECTION_MARKER_START}[\\s\\S]*?${SECTION_MARKER_END}`,
        'g'
      );
      const match = targetFileContent.match(regex);
      // console.log('Regex match:', match);

      if (match) {
        targetFileContent = targetFileContent.replace(regex, newContent);
        await fs.writeFile(TARGET_FILE, targetFileContent, 'utf-8');
        console.log(`✅ File ${TARGET_FILE} has been updated.`);
      } else {
        console.error(`❌ Failed to find the marker in the target file: ${TARGET_FILE}`);
      }
    }

    console.log('\nAll TARGET_FILES have been processed.');
  } catch (error) {
    console.error('❌ Error：', error);
  }
})();
