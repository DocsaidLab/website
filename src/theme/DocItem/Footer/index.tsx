import { useDoc } from '@docusaurus/plugin-content-docs/client';
import type { WrapperProps } from "@docusaurus/types";
import GiscusComment from '@site/src/components/GiscusComment';
import Footer from "@theme-original/DocItem/Footer";
import type FooterType from "@theme/DocItem/Footer";
import React from "react";




type Props = WrapperProps<typeof FooterType>;

export default function FooterWrapper(props: Props): JSX.Element {
  {
  }
  const { metadata, frontMatter, assets } = useDoc();
  const { no_comments } = frontMatter;
  const { title, slug } = metadata;
  {
  }
  return (
    <>
      <Footer {...props} />
      {!no_comments && (
        <GiscusComment />
      )}
    </>
  );
}
