"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["96755"],{25446:function(n,e,i){i.r(e),i.d(e,{metadata:()=>r,contentTitle:()=>o,default:()=>p,assets:()=>c,toc:()=>l,frontMatter:()=>a});var r=JSON.parse('{"id":"capybara/funcs/vision/morphology/imgradient","title":"imgradient","description":"imgradient(img Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, \\"MORPH\\"] = \\"MORPH.RECT\\") -> np.ndarray","source":"@site/docs/capybara/funcs/vision/morphology/imgradient.md","sourceDirName":"capybara/funcs/vision/morphology","slug":"/capybara/funcs/vision/morphology/imgradient","permalink":"/docs/capybara/funcs/vision/morphology/imgradient","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734827263000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"imerode","permalink":"/docs/capybara/funcs/vision/morphology/imerode"},"next":{"title":"imopen","permalink":"/docs/capybara/funcs/vision/morphology/imopen"}}'),t=i("85893"),s=i("50065");let a={},o="imgradient",c={},l=[];function d(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...n.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(e.header,{children:(0,t.jsx)(e.h1,{id:"imgradient",children:"imgradient"})}),"\n",(0,t.jsxs)(e.blockquote,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/morphology.py#L135",children:'imgradient(img: np.ndarray, ksize: Union[int, Tuple[int, int]] = (3, 3), kstruct: Union[str, int, "MORPH"] = "MORPH.RECT") -> np.ndarray'})}),"\n"]}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsxs)(e.p,{children:[(0,t.jsx)(e.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u68AF\u5EA6\u904B\u7B97\uFF1A\u81A8\u8139\u5716\u50CF\u6E1B\u53BB\u4FB5\u8755\u5716\u50CF\u7684\u7D50\u679C\u3002\u5C0D\u65BC\u591A\u901A\u9053\u5716\u50CF\uFF0C\u6BCF\u500B\u901A\u9053\u90FD\u5C07\u7368\u7ACB\u8655\u7406\u3002\u610F\u7FA9\u662F\u53EF\u4EE5\u7528\u4F86\u63D0\u53D6\u7269\u9AD4\u7684\u908A\u7DE3\u3002"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"\u53C3\u6578"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"img"})," (",(0,t.jsx)(e.code,{children:"np.ndarray"}),")\uFF1A\u8F38\u5165\u5716\u50CF\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"ksize"})," (",(0,t.jsx)(e.code,{children:"Union[int, Tuple[int, int]]"}),")\uFF1A\u7D50\u69CB\u5143\u7D20\u7684\u5927\u5C0F\u3002\u9810\u8A2D\u70BA (3, 3)\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"kstruct"})," (",(0,t.jsx)(e.code,{children:"MORPH"}),')\uFF1A\u5143\u7D20\u5F62\u72C0\uFF0C\u53EF\u4EE5\u662F "MORPH.CROSS", "MORPH.RECT", "MORPH.ELLIPSE" \u4E4B\u4E00\u3002\u9810\u8A2D\u70BA "MORPH.RECT"\u3002']}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",children:"import numpy as np\nimport capybara as cb\n\nimg = np.array([[0, 0, 0, 0, 0],\n                [0, 1, 1, 1, 0],\n                [0, 1, 1, 1, 0],\n                [0, 1, 1, 1, 0],\n                [0, 0, 0, 0, 0]], dtype=np.uint8)\n\ngradient_img = cb.imgradient(img, ksize=3, kstruct='RECT')\n\n# Kernel will be like this:\n# >>> np.array([[1, 1, 1],\n#               [1, 1, 1],\n#               [1, 1, 1]], dtype=np.uint8)\n\n# After gradient, the image will be like this:\n# >>> np.array([[1, 1, 1, 1, 1],\n#               [1, 1, 1, 1, 1],\n#               [1, 1, 0, 1, 1],\n#               [1, 1, 1, 1, 1],\n#               [1, 1, 1, 1, 1]], dtype=np.uint8)\n"})}),"\n"]}),"\n"]})]})}function p(n={}){let{wrapper:e}={...(0,s.a)(),...n.components};return e?(0,t.jsx)(e,{...n,children:(0,t.jsx)(d,{...n})}):d(n)}},50065:function(n,e,i){i.d(e,{Z:function(){return o},a:function(){return a}});var r=i(67294);let t={},s=r.createContext(t);function a(n){let e=r.useContext(s);return r.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function o(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:a(n.components),r.createElement(s.Provider,{value:e},n.children)}}}]);