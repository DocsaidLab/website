"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["69853"],{23666:function(n,e,r){r.r(e),r.d(e,{metadata:()=>c,contentTitle:()=>a,default:()=>d,assets:()=>o,toc:()=>l,frontMatter:()=>i});var c=JSON.parse('{"id":"capybara/funcs/vision/functionals/centercrop","title":"centercrop","description":"centercrop(img: np.ndarray) -> np.ndarray","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/funcs/vision/functionals/centercrop.md","sourceDirName":"capybara/funcs/vision/functionals","slug":"/capybara/funcs/vision/functionals/centercrop","permalink":"/en/docs/capybara/funcs/vision/functionals/centercrop","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"functionals","permalink":"/en/docs/category/functionals"},"next":{"title":"gaussianblur","permalink":"/en/docs/capybara/funcs/vision/functionals/gaussianblur"}}'),t=r("85893"),s=r("50065");let i={},a="centercrop",o={},l=[];function p(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...n.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(e.header,{children:(0,t.jsx)(e.h1,{id:"centercrop",children:"centercrop"})}),"\n",(0,t.jsxs)(e.blockquote,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L374",children:"centercrop(img: np.ndarray) -> np.ndarray"})}),"\n"]}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsxs)(e.p,{children:[(0,t.jsx)(e.strong,{children:"Description"}),": Performs a center crop on the input image."]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"img"})," (",(0,t.jsx)(e.code,{children:"np.ndarray"}),"): The input image to be center-cropped."]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Returns"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"np.ndarray"}),": The cropped image."]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Example"})}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\nimg = cb.imresize(img, [128, 256])\ncrop_img = cb.centercrop(img)\n"})}),"\n",(0,t.jsx)(e.p,{children:"The green box represents the area of the center crop."}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.img,{alt:"centercrop",src:r(79792).Z+"",width:"630",height:"256"})}),"\n"]}),"\n"]})]})}function d(n={}){let{wrapper:e}={...(0,s.a)(),...n.components};return e?(0,t.jsx)(e,{...n,children:(0,t.jsx)(p,{...n})}):p(n)}},79792:function(n,e,r){r.d(e,{Z:function(){return c}});let c=r.p+"assets/images/test_centercrop-67126c32c625b509fdee0d42ce85698d.jpg"},50065:function(n,e,r){r.d(e,{Z:function(){return a},a:function(){return i}});var c=r(67294);let t={},s=c.createContext(t);function i(n){let e=c.useContext(s);return c.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function a(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:i(n.components),c.createElement(s.Provider,{value:e},n.children)}}}]);