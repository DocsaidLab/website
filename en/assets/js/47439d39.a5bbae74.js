"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["42970"],{68010:function(n,e,s){s.r(e),s.d(e,{metadata:()=>i,contentTitle:()=>c,default:()=>u,assets:()=>l,toc:()=>o,frontMatter:()=>r});var i=JSON.parse('{"id":"capybara/funcs/vision/functionals/imadjust","title":"imadjust","description":"imadjust(img Tuple[int, int] = (0, 255), gamma str = \'BGR\') -> np.ndarray","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/funcs/vision/functionals/imadjust.md","sourceDirName":"capybara/funcs/vision/functionals","slug":"/capybara/funcs/vision/functionals/imadjust","permalink":"/en/docs/capybara/funcs/vision/functionals/imadjust","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"gaussianblur","permalink":"/en/docs/capybara/funcs/vision/functionals/gaussianblur"},"next":{"title":"imbinarize","permalink":"/en/docs/capybara/funcs/vision/functionals/imbinarize"}}'),t=s("85893"),a=s("50065");let r={},c="imadjust",l={},o=[];function d(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,a.a)(),...n.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(e.header,{children:(0,t.jsx)(e.h1,{id:"imadjust",children:"imadjust"})}),"\n",(0,t.jsxs)(e.blockquote,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L122",children:"imadjust(img: np.ndarray, rng_out: Tuple[int, int] = (0, 255), gamma: float = 1.0, color_base: str = 'BGR') -> np.ndarray"})}),"\n"]}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsxs)(e.p,{children:[(0,t.jsx)(e.strong,{children:"Description"}),": Adjusts the intensity of an image."]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Parameters"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"img"})," (",(0,t.jsx)(e.code,{children:"np.ndarray"}),"): The input image to adjust the intensity. It can be 2-D or 3-D."]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"rng_out"})," (",(0,t.jsx)(e.code,{children:"Tuple[int, int]"}),"): The target intensity range for the output image. Default is (0, 255)."]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"gamma"})," (",(0,t.jsx)(e.code,{children:"float"}),"): The value used for gamma correction. If gamma is less than 1, the mapping will be skewed toward higher (brighter) output values. If gamma is greater than 1, the mapping will be skewed toward lower (darker) output values. Default is 1.0 (linear mapping)."]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"color_base"})," (",(0,t.jsx)(e.code,{children:"str"}),"): The color basis of the input image. Should be 'BGR' or 'RGB'. Default is 'BGR'."]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Returns"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"np.ndarray"}),": The adjusted image."]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"Example"})}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\nadj_img = cb.imadjust(img, gamma=2)\n"})}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.img,{alt:"imadjust",src:s(15911).Z+"",width:"426",height:"256"})}),"\n"]}),"\n"]})]})}function u(n={}){let{wrapper:e}={...(0,a.a)(),...n.components};return e?(0,t.jsx)(e,{...n,children:(0,t.jsx)(d,{...n})}):d(n)}},15911:function(n,e,s){s.d(e,{Z:function(){return i}});let i=s.p+"assets/images/test_imadjust-d2e6bf3028c135b953e2f1a0bb952fe4.jpg"},50065:function(n,e,s){s.d(e,{Z:function(){return c},a:function(){return r}});var i=s(67294);let t={},a=i.createContext(t);function r(n){let e=i.useContext(a);return i.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:r(n.components),i.createElement(a.Provider,{value:e},n.children)}}}]);