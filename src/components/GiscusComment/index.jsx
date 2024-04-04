import { useColorMode } from '@docusaurus/theme-common';
import Giscus from "@giscus/react";
import React from 'react';

export default function GiscusComment() {
    const { colorMode } = useColorMode();

    return (
        <div style={{ marginTop: '3rem' }}> {/* 增加邊距 */}
            <Giscus
                repo="DocsaidLab/blog"
                repoId="R_kgDOK0_How"
                category="Announcements"
                categoryId="DIC_kwDOK0_Ho84CeZ07"
                mapping="title"
                strict="0"
                reactionsEnabled="1"
                emitMetadata="0"
                inputPosition="top"
                theme={colorMode}
                lang="zh-TW"
                loading="lazy"
                crossorigin="anonymous"
                async
            />
        </div>
    );
}