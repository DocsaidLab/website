import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'DOCSAID',
  tagline: 'A playground for our developers',
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
    locales: ['zh-hant', 'en']
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          blogTitle: 'Docsaid blog!',
          blogDescription: 'A Docsaid powered blog!',
          postsPerPage: 'ALL',
          showReadingTime: true,
          blogSidebarTitle: 'All our posts',
          blogSidebarCount: 'ALL',
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
        sidebarPath: './sidebarsPapers.js',
        showLastUpdateAuthor: true,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'playground',
        path: 'playground',
        routeBasePath: 'playground',
        sidebarPath: './sidebarsPlayground.js',
        showLastUpdateAuthor: true,
      },
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
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
            label: 'Docs',
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
          },
          {
            label: 'Papers',
            type: 'docSidebar',
            sidebarId: 'papersSidebar',
            position: 'left',
            docsPluginId: 'papers'
          },
          {
            label: 'Blog',
            to: '/blog',
            position: 'left'
          },
          {
            label: 'Playground',
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
        ],
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        }
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
      },
      footer: {
        style: 'dark',
        links: [
          {
            label: 'Docs',
            to: '/',
          },
          {
            label: 'Papers',
            to: '/papers/intro',
          },
          {
            label: 'Blog',
            to: '/blog',
          },
          {
            label: 'GitHub',
            href: 'https://github.com/DocsaidLab',
          },
          {
            label: '使用條款',
            to: '/blog/terms-of-service',
          },
          {
            label: '隱私政策',
            to: '/blog/privacy-policy',
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} DOCSAID.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
