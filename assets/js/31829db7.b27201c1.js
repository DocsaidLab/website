"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["97388"],{67462:function(e,n,r){r.r(n),r.d(n,{metadata:()=>i,contentTitle:()=>d,default:()=>h,assets:()=>a,toc:()=>l,frontMatter:()=>o});var i=JSON.parse('{"id":"capybara/funcs/vision/improc/imread","title":"imread","description":"imread(path str = \'BGR\', verbose: bool = False) -> Union[np.ndarray, None]","source":"@site/docs/capybara/funcs/vision/improc/imread.md","sourceDirName":"capybara/funcs/vision/improc","slug":"/capybara/funcs/vision/improc/imread","permalink":"/docs/capybara/funcs/vision/improc/imread","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734827263000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"imencode","permalink":"/docs/capybara/funcs/vision/improc/imencode"},"next":{"title":"imwrite","permalink":"/docs/capybara/funcs/vision/improc/imwrite"}}'),s=r("85893"),c=r("50065");let o={},d="imread",a={},l=[];function t(e){let n={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,c.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.header,{children:(0,s.jsx)(n.h1,{id:"imread",children:"imread"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/improc.py#L197",children:"imread(path: Union[str, Path], color_base: str = 'BGR', verbose: bool = False) -> Union[np.ndarray, None]"})}),"\n"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u8B80\u53D6\u5716\u7247\uFF0C\u57FA\u65BC\u4E0D\u540C\u7684\u5F71\u50CF\u683C\u5F0F\uFF0C\u4F7F\u7528\u4E0D\u540C\u7684\u8B80\u53D6\u65B9\u5F0F\uFF0C\u5176\u652F\u63F4\u7684\u683C\u5F0F\u8AAA\u660E\u5982\u4E0B\uFF1A"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:".heic"}),"\uFF1A\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"read_heic_to_numpy"})," \u8B80\u53D6\uFF0C\u4E26\u8F49\u63DB\u6210 ",(0,s.jsx)(n.code,{children:"BGR"})," \u683C\u5F0F\u3002"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:".jpg"}),"\uFF1A\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"jpgread"})," \u8B80\u53D6\uFF0C\u4E26\u8F49\u63DB\u6210 ",(0,s.jsx)(n.code,{children:"BGR"})," \u683C\u5F0F\u3002"]}),"\n",(0,s.jsxs)(n.li,{children:["\u5176\u4ED6\u683C\u5F0F\uFF1A\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"cv2.imread"})," \u8B80\u53D6\uFF0C\u4E26\u8F49\u63DB\u6210 ",(0,s.jsx)(n.code,{children:"BGR"})," \u683C\u5F0F\u3002"]}),"\n",(0,s.jsxs)(n.li,{children:["\u82E5\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"jpgread"})," \u8B80\u53D6\u70BA ",(0,s.jsx)(n.code,{children:"None"}),"\uFF0C\u5247\u6703\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"cv2.imread"})," \u9032\u884C\u8B80\u53D6\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"path"})," (",(0,s.jsx)(n.code,{children:"Union[str, Path]"}),")\uFF1A\u8981\u8B80\u53D6\u7684\u5716\u7247\u8DEF\u5F91\u3002"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"color_base"})," (",(0,s.jsx)(n.code,{children:"str"}),")\uFF1A\u5716\u7247\u7684\u8272\u5F69\u7A7A\u9593\u3002\u5982\u679C\u4E0D\u662F ",(0,s.jsx)(n.code,{children:"BGR"}),"\uFF0C\u5C07\u4F7F\u7528 ",(0,s.jsx)(n.code,{children:"imcvtcolor"})," \u51FD\u6578\u9032\u884C\u8F49\u63DB\u3002\u9810\u8A2D\u70BA ",(0,s.jsx)(n.code,{children:"BGR"}),"\u3002"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"verbose"})," (",(0,s.jsx)(n.code,{children:"bool"}),")\uFF1A\u5982\u679C\u8A2D\u7F6E\u70BA True\uFF0C\u7576\u8B80\u53D6\u7684\u5716\u7247\u70BA None \u6642\uFF0C\u5C07\u767C\u51FA\u8B66\u544A\u3002\u9810\u8A2D\u70BA False\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"np.ndarray"}),"\uFF1A\u6210\u529F\u6642\u8FD4\u56DE\u5716\u7247\u7684 numpy ndarray\uFF0C\u5426\u5247\u8FD4\u56DE None\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\n"})}),"\n"]}),"\n"]})]})}function h(e={}){let{wrapper:n}={...(0,c.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(t,{...e})}):t(e)}},50065:function(e,n,r){r.d(n,{Z:function(){return d},a:function(){return o}});var i=r(67294);let s={},c=i.createContext(s);function o(e){let n=i.useContext(c);return i.useMemo(function(){return"function"==typeof e?e(n):{...n,...e}},[n,e])}function d(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),i.createElement(c.Provider,{value:n},e.children)}}}]);