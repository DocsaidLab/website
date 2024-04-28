import { useBlogPost } from '@docusaurus/theme-common/internal';
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
export default function BlogPostItem({
  children,
  className}
) {

  const containerClassName = useContainerClassName();
  const {metadata,frontMatter,isBlogPostPage} = useBlogPost();

  return (
    <BlogPostItemContainer className={clsx(containerClassName, className)}>
      {isBlogPostPage ? (
        <div>
          <BlogPostItemHeader />
          <BlogPostItemContent>{children}</BlogPostItemContent>
          <BlogPostItemFooter />
        </div>
      ) : (
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
