import BlogSidebar from '@theme/BlogSidebar';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import React from 'react';
export default function BlogLayout(props) {
  const {sidebar, toc, children, ...layoutProps} = props;
  const hasSidebar = sidebar && sidebar.items.length > 0;
  return (
    <Layout {...layoutProps}>
      <div className="container margin-vert--lg">
        <div className="row">
          <main
            className={clsx('col', {
              'col--9': hasSidebar,
              'col--9 col--offset-1': !hasSidebar,
            })}>
            {children}
          </main>
          <BlogSidebar sidebar={sidebar} />
          {toc && <div className="col col--2">{toc}</div>}
        </div>
      </div>
    </Layout>
  );
}
