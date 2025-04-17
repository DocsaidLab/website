import Link from '@docusaurus/Link';
import {
  BlogPostProvider,
  useBlogPost,
} from '@docusaurus/plugin-content-blog/client';
import { HtmlClassNameProvider, ThemeClassNames } from '@docusaurus/theme-common';
import BlogLayout from '@theme/BlogLayout';
import BlogPostItem from '@theme/BlogPostItem';
import BlogPostPageMetadata from '@theme/BlogPostPage/Metadata';
import BlogPostPageStructuredData from '@theme/BlogPostPage/StructuredData';
import BlogPostPaginator from '@theme/BlogPostPaginator';
import ContentVisibility from '@theme/ContentVisibility';
import TOC from '@theme/TOC';
import clsx from 'clsx';
import React, { useEffect, useState } from 'react';
import styles from './index.module.css';

/** 小眼睛圖示 (SVG) */
function EyeIcon({className, size = 16}) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      style={{
        marginRight: 4,
        verticalAlign: 'middle',
      }}
    >
      <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zm0 12c-2.48 0-4.5-2.02-4.5-4.5S9.52 7.5 12 7.5s4.5 2.02 4.5 4.5-2.02 4.5-4.5 4.5z"/>
      <circle cx="12" cy="12" r="2.25"/>
    </svg>
  );
}

function BlogPostPageContent({sidebar, children}) {
  const {metadata, toc, frontMatter = {}} = useBlogPost();
  const {nextItem, prevItem} = metadata;

  // 取得 slug
  const slug = frontMatter.slug || '';

  // 新增：點閱數狀態
  const [viewCount, setViewCount] = useState(null);

  useEffect(() => {
    if (!slug) return;

    fetch(`https://api.docsaid.org/pageviews/track?slug=${slug}`, {
      method: 'POST'
    })
      .then(res => res.json())
      .then(data => {
        console.log(data.message || 'Track done');
      })
      .catch(err => console.error(err));

    fetch(`https://api.docsaid.org/pageviews/count/${slug}`)
      .then(res => res.json())
      .then(data => setViewCount(data.count))
      .catch(() => setViewCount(0));
  }, [slug]);

  // 解構 frontMatter
  const {
    hide_table_of_contents: hideTableOfContents = false,
    toc_min_heading_level: tocMinHeadingLevel = 2,
    toc_max_heading_level: tocMaxHeadingLevel = 3,
    no_comments = false,
    image
  } = frontMatter;

  const {title, date, readingTime, tags, authors} = metadata;

  const authorsArray = Array.isArray(authors) ? authors : (authors ? [authors] : []);

  // Hero 區塊：若 frontMatter.image 有設定，就做為背景圖
  const hero = image && (
    <div className={styles.postHero} style={{ backgroundImage: `url(${image})` }}>
      <div className={styles.postHeroOverlay}>
        <h1 className={styles.postTitle}>{title}</h1>
        <div className={styles.postMeta}>
          {/* 作者資訊 */}
          {authorsArray.length > 0 && (
            <div className={styles.postAuthors}>
              {authorsArray.map((author, idx) => {
                return (
                  <div className={styles.postAuthor} key={idx}>
                    {author.imageURL && (
                      <img
                        className={styles.postAuthorImg}
                        src={author.imageURL}
                        alt={author.name}
                      />
                    )}
                    <div className={styles.postAuthorText}>
                      {author.url ? (
                        <a
                          href={author.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={styles.postAuthorLink}
                        >
                          <span className={styles.postAuthorName}>{author.name}</span>
                        </a>
                      ) : (
                        <span className={styles.postAuthorName}>{author.name}</span>
                      )}
                      {author.title && (
                        <span className={styles.postAuthorTitle}>{author.title}</span>
                      )}
                      {author.description && (
                        <p className={styles.postAuthorDesc}>{author.description}</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* 發布日期、閱讀時間、標籤 */}
          <div className={styles.postMetaInfo}>
            <div className={styles.postMetaRow}>
              {date && (
                <span className={styles.postDate}>
                  {new Date(date).toLocaleDateString(undefined, {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </span>
              )}
              {readingTime && (
                <span className={styles.postReadingTime}>
                  {Math.ceil(readingTime)} min read
                </span>
              )}

              {/* 在這裡插入「瀏覽數」 */}
              {viewCount !== null && (
                <span className={styles.postViewCount}>
                  <EyeIcon size={16} />
                  {viewCount}
                </span>
              )}
            </div>
            {tags && tags.length > 0 && (
              <div className={styles.postTags}>
                {tags.map((tag) => (
                  <Link to={tag.permalink} key={tag.label} className={styles.postTag}>
                    {tag.label}
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <BlogLayout
      sidebar={sidebar}
      hero={hero}
      toc={
        !hideTableOfContents && toc.length > 0 ? (
          <TOC
            toc={toc}
            minHeadingLevel={tocMinHeadingLevel}
            maxHeadingLevel={tocMaxHeadingLevel}
          />
        ) : undefined
      }
    >
      <ContentVisibility metadata={metadata} />

      <article
        className="markdown"
        style={{maxWidth: '800px', margin: '2rem auto'}}
      >
        <BlogPostItem>{children}</BlogPostItem>
      </article>

      {/* 上一篇/下一篇 */}
      {(nextItem || prevItem) && (
        <BlogPostPaginator nextItem={nextItem} prevItem={prevItem} />
      )}
    </BlogLayout>
  );
}

export default function BlogPostPage(props) {
  const {content: BlogPostContent} = props;
  return (
    <BlogPostProvider content={props.content} isBlogPostPage>
      <HtmlClassNameProvider
        className={clsx(
          ThemeClassNames.wrapper.blogPages,
          ThemeClassNames.page.blogPostPage,
        )}
      >
        <BlogPostPageMetadata />
        <BlogPostPageStructuredData />
        <BlogPostPageContent sidebar={props.sidebar}>
          <BlogPostContent />
        </BlogPostPageContent>
      </HtmlClassNameProvider>
    </BlogPostProvider>
  );
}
