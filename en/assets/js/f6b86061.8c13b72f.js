"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["88820"],{89573:function(e,n,o){o.r(n),o.d(n,{metadata:()=>t,contentTitle:()=>i,default:()=>h,assets:()=>d,toc:()=>a,frontMatter:()=>c});var t=JSON.parse('{"id":"capybara/funcs/structures/box_mode","title":"BoxMode","description":"BoxMode","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/funcs/structures/box_mode.md","sourceDirName":"capybara/funcs/structures","slug":"/capybara/funcs/structures/box_mode","permalink":"/en/docs/capybara/funcs/structures/box_mode","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","previous":{"title":"Structures","permalink":"/en/docs/category/structures"},"next":{"title":"Box","permalink":"/en/docs/capybara/funcs/structures/box"}}'),s=o("85893"),r=o("50065");let c={sidebar_position:1},i="BoxMode",d={},a=[];function l(e){let n={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.header,{children:(0,s.jsx)(n.h1,{id:"boxmode",children:"BoxMode"})}),"\n",(0,s.jsxs)(n.blockquote,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/boxes.py#L26",children:"BoxMode"})}),"\n"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Description"}),":"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.code,{children:"BoxMode"})," is an enumeration class used to represent different bounding box formats."]}),"\n",(0,s.jsx)(n.p,{children:"Generally, there are three common bounding box representations:"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"XYXY"}),": Represented as ",(0,s.jsx)(n.code,{children:"(x0, y0, x1, y1)"}),", using absolute floating-point coordinates. The coordinate range is ",(0,s.jsx)(n.code,{children:"[0, w]"})," and ",(0,s.jsx)(n.code,{children:"[0, h]"}),"."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"XYWH"}),": Represented as ",(0,s.jsx)(n.code,{children:"(x0, y0, w, h)"}),", using absolute floating-point coordinates. ",(0,s.jsx)(n.code,{children:"(x0, y0)"})," is the top-left corner of the bounding box, and ",(0,s.jsx)(n.code,{children:"(w, h)"})," is the width and height of the bounding box."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"CXCYWH"}),": Represented as ",(0,s.jsx)(n.code,{children:"(xc, yc, w, h)"}),", using absolute floating-point coordinates. ",(0,s.jsx)(n.code,{children:"(xc, yc)"})," is the center of the bounding box, and ",(0,s.jsx)(n.code,{children:"(w, h)"})," is the width and height of the bounding box."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:["We believe a good design should allow smooth conversion between these formats. Therefore, the ",(0,s.jsx)(n.code,{children:"BoxMode"})," class implements a ",(0,s.jsx)(n.code,{children:"convert"})," method for this purpose. You can refer to the following example for usage. Additionally, this class also implements an ",(0,s.jsx)(n.code,{children:"align_code"})," method, which accepts case-insensitive strings and converts them to uppercase representation."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Example"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"import capybara as cb\nimport numpy as np\n\nbox = np.array([10, 20, 50, 80]).astype(np.float32)\nbox = cb.BoxMode.convert(box, from_mode=cb.BoxMode.XYXY, to_mode=cb.BoxMode.XYWH)\n# >>> array([10, 20, 40, 60])\n\n# Using string to represent the mode\nbox = cb.BoxMode.convert(box, from_mode='XYWH', to_mode='CXCYWH')\n# >>> array([30, 50, 40, 60])\n"})}),"\n"]}),"\n"]})]})}function h(e={}){let{wrapper:n}={...(0,r.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(l,{...e})}):l(e)}},50065:function(e,n,o){o.d(n,{Z:function(){return i},a:function(){return c}});var t=o(67294);let s={},r=t.createContext(s);function c(e){let n=t.useContext(r);return t.useMemo(function(){return"function"==typeof e?e(n):{...n,...e}},[n,e])}function i(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:c(e.components),t.createElement(r.Provider,{value:n},e.children)}}}]);