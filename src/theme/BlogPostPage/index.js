import Link from '@docusaurus/Link';
import {
  BlogPostProvider,
  useBlogPost,
} from '@docusaurus/plugin-content-blog/client';
import { HtmlClassNameProvider, ThemeClassNames } from '@docusaurus/theme-common';
import GiscusComment from '@site/src/components/GiscusComment';
import BlogLayout from '@theme/BlogLayout';
import BlogPostItem from '@theme/BlogPostItem';
import BlogPostPageMetadata from '@theme/BlogPostPage/Metadata';
import BlogPostPageStructuredData from '@theme/BlogPostPage/StructuredData';
import BlogPostPaginator from '@theme/BlogPostPaginator';
import ContentVisibility from '@theme/ContentVisibility';
import TOC from '@theme/TOC';
import clsx from 'clsx';
import React from 'react';
import styles from './index.module.css';

function BlogPostPageContent({sidebar, children}) {
  const {metadata, toc, frontMatter = {}} = useBlogPost();
  const {nextItem, prevItem} = metadata;

  // 解構frontMatter時提供預設值以防止undefined
  const {
    hide_table_of_contents: hideTableOfContents = false,
    toc_min_heading_level: tocMinHeadingLevel = 2,
    toc_max_heading_level: tocMaxHeadingLevel = 3,
    no_comments = false,
    image
  } = frontMatter;

  const {title, description, authors, date, readingTime, tags} = metadata;
  const authorsArray = Array.isArray(authors) ? authors : (authors ? [authors] : []);

  // Hero 區塊
  const hero = image && (
    <div className={styles.postHero} style={{ backgroundImage: `url(${image})` }}>
      <div className={styles.postHeroOverlay}>
        <h1 className={styles.postTitle}>{title}</h1>
        <div className={styles.postMeta}>
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

                  {/* 新增: 包作者資訊的容器 */}
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

      {!no_comments && <GiscusComment />}

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
