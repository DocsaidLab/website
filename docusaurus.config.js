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
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl:
          //   'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          routeBasePath: '/',
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
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: '',
        hideOnScroll: true,
        logo: {
          alt: 'Docsaid Logo',
          src: 'img/docsaid_logo.png',
        },
        items: [
          {
            label: 'Documents',
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
          },
          {
            label: 'Blog',
            to: '/',
            position: 'left'
          },
          {
            label: '用戶指南',
            type: 'dropdown',
            position: 'right',
            items: [
              {
                to: '/terms-of-service',
                label: '使用條款',
              },
              {
                to: '/privacy-policy',
                label: '隱私政策',
              }
            ]
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
            title: 'Docs',
            items: [
              {
                label: 'Tutorial',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/DocsaidLab',
              },
            ],
          },
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
