"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["47187"],{76110:function(n,e,r){r.r(e),r.d(e,{metadata:()=>c,contentTitle:()=>i,default:()=>d,assets:()=>o,toc:()=>l,frontMatter:()=>t});var c=JSON.parse('{"id":"capybara/funcs/vision/ipcam/ipcamcapture","title":"IpcamCapture","description":"IpcamCapture(url str) -> None","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/funcs/vision/ipcam/ipcamcapture.md","sourceDirName":"capybara/funcs/vision/ipcam","slug":"/capybara/funcs/vision/ipcam/ipcamcapture","permalink":"/ja/docs/capybara/funcs/vision/ipcam/ipcamcapture","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"ipcam","permalink":"/ja/docs/category/ipcam"},"next":{"title":"WebDemo","permalink":"/ja/docs/capybara/funcs/vision/ipcam/webdemo"}}'),s=r("85893"),a=r("50065");let t={},i="IpcamCapture",o={},l=[];function p(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,a.a)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.header,{children:(0,s.jsx)(e.h1,{id:"ipcamcapture",children:"IpcamCapture"})}),"\n",(0,s.jsxs)(e.blockquote,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/ipcam/camera.py#L11",children:"IpcamCapture(url: int, str, color_base: str) -> None"})}),"\n"]}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsxs)(e.p,{children:[(0,s.jsx)(e.strong,{children:"\u8AAC\u660E"}),"\uFF1AIP \u30AB\u30E1\u30E9\u304B\u3089\u753B\u50CF\u3092\u30AD\u30E3\u30D7\u30C1\u30E3\u3057\u307E\u3059\u3002"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u5F15\u6570"})}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"url"})," (",(0,s.jsx)(e.code,{children:"int"}),", ",(0,s.jsx)(e.code,{children:"str"}),")\uFF1A\u30D3\u30C7\u30AA\u30BD\u30FC\u30B9\u306E\u8B58\u5225\u5B50\u3002\u3053\u308C\u306F\u30ED\u30FC\u30AB\u30EB\u63A5\u7D9A\u306E\u30AB\u30E1\u30E9\u30C7\u30D0\u30A4\u30B9\u30A4\u30F3\u30C7\u30C3\u30AF\u30B9\u3001\u307E\u305F\u306F IP \u30AB\u30E1\u30E9\u306E\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A2\u30C9\u30EC\u30B9\u3092\u542B\u3080\u6587\u5B57\u5217\u3067\u3059\u3002\u30ED\u30FC\u30AB\u30EB\u30AB\u30E1\u30E9\u306E\u5834\u5408\u30010 \u306F\u901A\u5E38\u30C7\u30D5\u30A9\u30EB\u30C8\u30AB\u30E1\u30E9\u3067\u3059\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F 0\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"color_base"})," (",(0,s.jsx)(e.code,{children:"str"}),")\uFF1A\u51FA\u529B\u30D5\u30EC\u30FC\u30E0\u306E\u8272\u7A7A\u9593\u3002'BGR' \u307E\u305F\u306F 'RGB' \u306B\u8A2D\u5B9A\u3067\u304D\u307E\u3059\u3002OpenCV \u306E\u5165\u529B\u30D5\u30EC\u30FC\u30E0\u306F\u5E38\u306B BGR \u5F62\u5F0F\u3067\u3059\u3002color_base \u304C 'RGB' \u306B\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u308B\u5834\u5408\u3001\u30D5\u30EC\u30FC\u30E0\u306F\u8FD4\u3055\u308C\u308B\u524D\u306B BGR \u304B\u3089 RGB \u306B\u5909\u63DB\u3055\u308C\u307E\u3059\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F 'BGR'\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u5C5E\u6027"})}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"color_base"})," (",(0,s.jsx)(e.code,{children:"str"}),")\uFF1A\u51FA\u529B\u30D5\u30EC\u30FC\u30E0\u306E\u8272\u7A7A\u9593\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u30E1\u30BD\u30C3\u30C9"})}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"get_frame() -> np.ndarray"}),"\uFF1A\u73FE\u5728\u30AD\u30E3\u30D7\u30C1\u30E3\u3055\u308C\u305F\u30D5\u30EC\u30FC\u30E0\u3092\u53D6\u5F97\u3057\u307E\u3059\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u4F8B"})}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\ncam = cb.IpcamCapture(url='http://your_ip:your_port/video')\nfor frame in cam:\n    cb.imwrite(frame, 'frame.jpg')\n"})}),"\n"]}),"\n"]})]})}function d(n={}){let{wrapper:e}={...(0,a.a)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(p,{...n})}):p(n)}},50065:function(n,e,r){r.d(e,{Z:function(){return i},a:function(){return t}});var c=r(67294);let s={},a=c.createContext(s);function t(n){let e=c.useContext(a);return c.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function i(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:t(n.components),c.createElement(a.Provider,{value:e},n.children)}}}]);