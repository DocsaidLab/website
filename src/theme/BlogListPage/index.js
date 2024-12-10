import Link from '@docusaurus/Link';
import BlogLayout from '@theme/BlogLayout';
import BlogListPaginator from '@theme/BlogListPaginator';
import React from 'react';
import styles from './index.module.css';

function BlogListPageContent(props) {
  const {metadata, items, sidebar} = props;

  return (
    <BlogLayout sidebar={sidebar}>
      <div className={styles.blogCardGrid}>
        {items.map(({content: BlogPostContent}) => {
          const {frontMatter, metadata} = BlogPostContent;
          const {title, description, image, tags} = frontMatter;
          const {permalink, readingTime, authors} = metadata;

          // 確保 authors 為陣列
          const authorsArray = Array.isArray(authors) ? authors : (authors ? [authors] : []);

          return (
            <div className={styles.blogCard} key={permalink}>
              {image && (
                <div className={styles.blogCardImageWrapper}>
                  <img className={styles.blogCardImage} src={image} alt={title} />
                </div>
              )}
              <div className={styles.blogCardContent}>
                <h2 className={styles.blogCardTitle}>
                  <Link to={permalink}>{title}</Link>
                </h2>
                {description && <p className={styles.blogCardDescription}>{description}</p>}

                {/* 若有 tags，顯示 tags 區塊 */}
                {tags && tags.length > 0 && (
                  <div className={styles.blogCardTags}>
                    {tags.map((tag) => (
                      <span className={styles.blogCardTag} key={tag}>
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {(authorsArray.length > 0 || readingTime) && (
                  <div className={styles.blogCardFooter}>
                    {authorsArray.length > 0 && (
                      <div className={styles.blogCardAuthors}>
                        {authorsArray.map((author, idx) => (
                          <div className={styles.blogCardAuthor} key={idx}>
                            {author.imageURL && (
                              <img
                                src={author.imageURL}
                                alt={author.name}
                                className={styles.blogCardAuthorImg}
                              />
                            )}
                            <span className={styles.blogCardAuthorName}>{author.name}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {readingTime && (
                      <div className={styles.blogCardReadingTime}>
                        {Math.ceil(readingTime)} min read
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <BlogListPaginator metadata={metadata} />
    </BlogLayout>
  );
}

export default function BlogListPage(props) {
  return <BlogListPageContent {...props} />;
}
