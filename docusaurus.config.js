import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'DOCSAID',
  tagline: 'A playground for our developers.',
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
          postsPerPage: 9,
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
      papers: {
        sidebar: {
          hideable: false,
          autoCollapseCategories: false,
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
            to: '/docs',
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
            href: 'https://docsaid.org/blog/terms-of-service',
          },
          {
            label: '隱私政策',
            href: 'https://docsaid.org/blog/privacy-policy',
          }
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
