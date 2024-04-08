// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'DOCSAID',
  tagline: 'A playground for our developers',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://docsaid.org',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'DocsaidLab', // Usually your GitHub org/user name.
  projectName: 'blog', // Usually your repo name.
  deploymentBranch: 'gh-pages',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'zh-hant',
    locales: ['zh-hant'],
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
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          blogTitle: 'Docsaid blog!',
          blogDescription: 'A Docsaid powered blog!',
          postsPerPage: 'ALL',
          showReadingTime: true,
          blogSidebarTitle: 'All our posts',
          blogSidebarCount: 'ALL',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
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
      // Replace with your project's social card
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
