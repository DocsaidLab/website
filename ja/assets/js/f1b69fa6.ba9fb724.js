"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["66250"],{67096:function(n,e,i){i.r(e),i.d(e,{metadata:()=>r,contentTitle:()=>c,default:()=>p,assets:()=>l,toc:()=>a,frontMatter:()=>o});var r=JSON.parse('{"id":"capybara/funcs/vision/morphology/imclose","title":"imclose","description":"imclose(img Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, \\"MORPH\\"] = \\"MORPH.RECT\\") -> np.ndarray","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/funcs/vision/morphology/imclose.md","sourceDirName":"capybara/funcs/vision/morphology","slug":"/capybara/funcs/vision/morphology/imclose","permalink":"/ja/docs/capybara/funcs/vision/morphology/imclose","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"imblackhat","permalink":"/ja/docs/capybara/funcs/vision/morphology/imblackhat"},"next":{"title":"imdilate","permalink":"/ja/docs/capybara/funcs/vision/morphology/imdilate"}}'),s=i("85893"),t=i("50065");let o={},c="imclose",l={},a=[];function d(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.a)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.header,{children:(0,s.jsx)(e.h1,{id:"imclose",children:"imclose"})}),"\n",(0,s.jsxs)(e.blockquote,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L105",children:'imclose(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray'})}),"\n"]}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsxs)(e.p,{children:[(0,s.jsx)(e.strong,{children:"\u8AAC\u660E"}),"\uFF1A\u30AF\u30ED\u30FC\u30B8\u30F3\u30B0\u6F14\u7B97\uFF1A\u81A8\u5F35\u5F8C\u306B\u4FB5\u98DF\u3092\u884C\u3046\u904E\u7A0B\u3067\u3001\u7269\u4F53\u5185\u90E8\u306E\u5C0F\u3055\u306A\u7A74\u3092\u57CB\u3081\u305F\u308A\u3001\u7269\u4F53\u306E\u30A8\u30C3\u30B8\u3092\u6ED1\u3089\u304B\u306B\u3057\u305F\u308A\u3001\u4E8C\u3064\u306E\u7269\u4F53\u3092\u63A5\u7D9A\u3059\u308B\u305F\u3081\u306B\u4F7F\u7528\u3055\u308C\u307E\u3059\u3002\u30DE\u30EB\u30C1\u30C1\u30E3\u30CD\u30EB\u753B\u50CF\u306E\u5834\u5408\u3001\u5404\u30C1\u30E3\u30CD\u30EB\u306F\u500B\u5225\u306B\u51E6\u7406\u3055\u308C\u307E\u3059\u3002"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u5F15\u6570"})}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"img"})," (",(0,s.jsx)(e.code,{children:"np.ndarray"}),")\uFF1A\u5165\u529B\u753B\u50CF\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"ksize"})," (",(0,s.jsx)(e.code,{children:"Union[int, Tuple[int, int]]"}),")\uFF1A\u69CB\u9020\u8981\u7D20\u306E\u30B5\u30A4\u30BA\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F (3, 3)\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"kstruct"})," (",(0,s.jsx)(e.code,{children:"MORPH"}),")\uFF1A\u8981\u7D20\u306E\u5F62\u72B6\u3002",(0,s.jsx)(e.code,{children:'"MORPH.CROSS"'}),", ",(0,s.jsx)(e.code,{children:'"MORPH.RECT"'}),", ",(0,s.jsx)(e.code,{children:'"MORPH.ELLIPSE"'})," \u306E\u3044\u305A\u308C\u304B\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F ",(0,s.jsx)(e.code,{children:'"MORPH.RECT"'}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:["\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.strong,{children:"\u4F8B"})}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-python",children:"import numpy as np\nimport capybara as cb\n\nimg = np.array([[1, 1, 1, 0, 0],\n                [1, 1, 1, 0, 0],\n                [1, 1, 1, 0, 0],\n                [0, 0, 0, 0, 0], # <- Look at this row\n                [0, 0, 0, 1, 1],\n                [0, 0, 0, 1, 1]], dtype=np.uint8)\n\nclosed_img = cb.imclose(img, ksize=3, kstruct='CROSS')\n\n# Kernel will be like this:\n# >>> np.array([[0, 1, 0],\n#               [1, 1, 1],\n#               [0, 1, 0]], dtype=np.uint8)\n\n# After closing, the image will be like this:\n# >>> np.array([[1, 1, 1, 0, 0],\n#               [1, 1, 1, 0, 0],\n#               [1, 1, 1, 0, 0],\n#               [0, 0, 1, 1, 0], # <- 1's are connected\n#               [0, 0, 0, 1, 1],\n#               [0, 0, 0, 1, 1]], dtype=np.uint8)\n"})}),"\n"]}),"\n"]})]})}function p(n={}){let{wrapper:e}={...(0,t.a)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(d,{...n})}):d(n)}},50065:function(n,e,i){i.d(e,{Z:function(){return c},a:function(){return o}});var r=i(67294);let s={},t=r.createContext(s);function o(n){let e=r.useContext(t);return r.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:o(n.components),r.createElement(t.Provider,{value:e},n.children)}}}]);