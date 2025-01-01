import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  future: {
    experimental_faster: true,
  },
  title: 'DOCSAID',
  tagline: '這是我們的技術遊樂場',
  favicon: 'img/favicon.ico',
  url: 'https://docsaid.org',
  baseUrl: '/',
  organizationName: 'DocsaidLab',
  projectName: 'website',
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'zh-hant',
    locales: ['zh-hant', 'en', 'ja']
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/docs',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: {
          blogTitle: 'Blog',
          blogDescription: 'Docsaid Blog.',
          showReadingTime: true,
          blogSidebarTitle: 'All our Posts',
          blogSidebarCount: 'ALL',
          postsPerPage: 6,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-RDF83L9R4M',
          anonymizeIP: true,
        },
        sitemap: {
          lastmod: 'date',
          changefreq: 'weekly',
          priority: 0.5,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      }),
    ],
  ],
  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'papers',
        path: 'papers',
        routeBasePath: 'papers',
        sidebarPath: require.resolve('./sidebarsPapers.js'),
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'playground',
        path: 'playground',
        routeBasePath: 'playground',
        sidebarPath: require.resolve('./sidebarsPlayground.js'),
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ]
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      announcementBar: {
        id: 'support_us',
        content:
          '檔案伺服器維護中，暫停下載模型功能。',
        backgroundColor: '#fafbfc',
        textColor: '#091E42',
        isCloseable: true,
      },
      image: 'img/docsaid-social-card.jpg',
      navbar: {
        title: '',
        hideOnScroll: true,
        logo: {
          alt: 'Docsaid Logo',
          src: 'img/docsaid_logo.png',
          srcDark: 'img/docsaid_logo_white.png',
        },
        items: [
          {
            label: '開源專案',
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
          },
          {
            label: '論文筆記',
            type: 'docSidebar',
            sidebarId: 'papersSidebar',
            position: 'left',
            docsPluginId: 'papers'
          },
          {
            label: '部落格',
            to: '/blog',
            position: 'left'
          },
          {
            label: '遊樂場',
            type: 'docSidebar',
            sidebarId: 'playgroundSidebar',
            position: 'left',
            docsPluginId: 'playground'
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            label: 'GitHub',
            position: 'right',
            href: 'https://github.com/DocsaidLab',
          },
          {
            label: '支持我們',
            position: 'right',
            href: 'https://buymeacoffee.com/docsaid',
          },
          {
            label: '關於我們',
            href: '/aboutus',
            position: 'right',
          },

        ],
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        }
      },
      papers: {
        sidebar: {
          hideable: false,
          autoCollapseCategories: false,
        }
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
      },
      footer: {
        style: 'dark',
        links: [
          {
            label: '開源專案',
            to: '/docs',
          },
          {
            label: '論文筆記',
            to: '/papers/intro',
          },
          {
            label: '部落格',
            to: '/blog',
          },
          {
            label: '使用條款',
            href: '/terms-of-service',
          },
          {
            label: '隱私政策',
            href: '/privacy-policy',
          },
          {
            label: '成為作者',
            href: '/become-an-author',
          },
          {
            label: '工作日誌',
            href: '/worklog',
          },
          {
            label: '支持我們',
            href: 'https://buymeacoffee.com/docsaid',
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} DOCSAID.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.vsDark,
      },
      algolia: {
        appId: 'S9NC0RYCHF',
        apiKey: '842757a059db8a232231828803688f96', // Public API key: it is safe to commit it
        indexName: 'docusaurus-algolia',
        contextualSearch: true,
        externalUrlRegex: 'external\\.com|domain\\.com',
        searchPagePath: 'search',
      },
    }),
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
};

export default config;
