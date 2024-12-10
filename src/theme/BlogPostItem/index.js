import { useBlogPost } from '@docusaurus/plugin-content-blog/client';
import BackToTopButton from '@theme/BackToTopButton';
import BlogPostItemContainer from '@theme/BlogPostItem/Container';
import BlogPostItemContent from '@theme/BlogPostItem/Content';
import BlogPostItemFooter from '@theme/BlogPostItem/Footer';
import BlogPostItemHeader from '@theme/BlogPostItem/Header';
import clsx from 'clsx';
import React from 'react';

// apply a bottom margin in list view
function useContainerClassName() {
  const {isBlogPostPage} = useBlogPost();
  return !isBlogPostPage ? 'margin-bottom--xl' : undefined;
}

export default function BlogPostItem({ children, className }) {
  const containerClassName = useContainerClassName();
  const {metadata, frontMatter, isBlogPostPage} = useBlogPost();

  return (
    <BlogPostItemContainer className={clsx(containerClassName, className)}>
      {isBlogPostPage ? (
        // 在文章內頁只顯示文章內容，不顯示標題、作者、日期等資訊
        <div>
          <BlogPostItemContent>{children}</BlogPostItemContent>
        </div>
      ) : (
        // 列表模式下的行為保持不變，或可依需求精簡
        <div>
          <a href={metadata.permalink}>
            {frontMatter.image && (
              <img className='margin-bottom--sm' loading='lazy' src={frontMatter.image} alt={frontMatter.title}/>
            )}
            <BlogPostItemHeader />
            <p>{frontMatter.description}</p>
          </a>
          <BlogPostItemFooter />
        </div>
      )}
      <BackToTopButton />
    </BlogPostItemContainer>
  );
}
