"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["19865"],{2534:function(n,e,r){r.r(e),r.d(e,{metadata:()=>c,contentTitle:()=>s,default:()=>p,assets:()=>a,toc:()=>l,frontMatter:()=>o});var c=JSON.parse('{"id":"capybara/funcs/vision/functionals/imcvtcolor","title":"imcvtcolor","description":"imcvtcolor(img Union[int, str]) -> np.ndarray","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/funcs/vision/functionals/imcvtcolor.md","sourceDirName":"capybara/funcs/vision/functionals","slug":"/capybara/funcs/vision/functionals/imcvtcolor","permalink":"/en/docs/capybara/funcs/vision/functionals/imcvtcolor","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"imcropboxes","permalink":"/en/docs/capybara/funcs/vision/functionals/imcropboxes"},"next":{"title":"meanblur","permalink":"/en/docs/capybara/funcs/vision/functionals/meanblur"}}'),i=r("85893"),t=r("50065");let o={},s="imcvtcolor",a={},l=[];function d(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.a)(),...n.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(e.header,{children:(0,i.jsx)(e.h1,{id:"imcvtcolor",children:"imcvtcolor"})}),"\n",(0,i.jsxs)(e.blockquote,{children:["\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L96",children:"imcvtcolor(img: np.ndarray, cvt_mode: Union[int, str]) -> np.ndarray"})}),"\n"]}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsxs)(e.p,{children:[(0,i.jsx)(e.strong,{children:"Description"}),": Converts the input image to a different color space."]}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"img"})," (",(0,i.jsx)(e.code,{children:"np.ndarray"}),"): The input image to be converted."]}),"\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"cvt_mode"})," (",(0,i.jsx)(e.code,{children:"Union[int, str]"}),"): The color conversion mode. It can be an integer constant representing the conversion code or a string representing an OpenCV color conversion name. For example, ",(0,i.jsx)(e.code,{children:"BGR2GRAY"})," is used to convert a BGR image to grayscale. For available options, refer to the ",(0,i.jsxs)(e.a,{href:"https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html",children:[(0,i.jsx)(e.strong,{children:"OpenCV COLOR"})," documentation"]}),"."]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Returns"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsxs)(e.li,{children:[(0,i.jsx)(e.strong,{children:"np.ndarray"}),": The image in the desired color space."]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(e.li,{children:["\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.strong,{children:"Example"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\nycrcb_img = cb.imcvtcolor(img, 'BGR2YCrCb')\n"})}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.img,{alt:"imcvtcolor_ycrcb",src:r(98826).Z+"",width:"426",height:"256"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\ngray_img = cb.imcvtcolor(img, 'BGR2GRAY')\n"})}),"\n",(0,i.jsx)(e.p,{children:(0,i.jsx)(e.img,{alt:"imcvtcolor_gray",src:r(78502).Z+"",width:"426",height:"256"})}),"\n"]}),"\n"]})]})}function p(n={}){let{wrapper:e}={...(0,t.a)(),...n.components};return e?(0,i.jsx)(e,{...n,children:(0,i.jsx)(d,{...n})}):d(n)}},78502:function(n,e,r){r.d(e,{Z:function(){return c}});let c=r.p+"assets/images/test_imcvtcolor_gray-54c79a2c6642900e976eb44d460f2d83.jpg"},98826:function(n,e,r){r.d(e,{Z:function(){return c}});let c=r.p+"assets/images/test_imcvtcolor_ycrcb-c4c02b18d21bca1524232a041ab26761.jpg"},50065:function(n,e,r){r.d(e,{Z:function(){return s},a:function(){return o}});var c=r(67294);let i={},t=c.createContext(i);function o(n){let e=c.useContext(t);return c.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function s(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(i):n.components||i:o(n.components),c.createElement(t.Provider,{value:e},n.children)}}}]);