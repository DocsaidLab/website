"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["19569"],{81917:function(n,e,i){i.r(e),i.d(e,{metadata:()=>s,contentTitle:()=>a,default:()=>p,assets:()=>c,toc:()=>d,frontMatter:()=>l});var s=JSON.parse('{"id":"capybara/funcs/vision/functionals/pad","title":"pad","description":"pad(img Union[int, Tuple[int, int], Tuple[int, int, int, int]], fillvalue Union[str, int, BORDER] = BORDER.CONSTANT) -> np.ndarray","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/funcs/vision/functionals/pad.md","sourceDirName":"capybara/funcs/vision/functionals","slug":"/capybara/funcs/vision/functionals/pad","permalink":"/ja/docs/capybara/funcs/vision/functionals/pad","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"medianblur","permalink":"/ja/docs/capybara/funcs/vision/functionals/medianblur"},"next":{"title":"geometric","permalink":"/ja/docs/category/geometric"}}'),t=i("85893"),r=i("50065");let l={},a="pad",c={},d=[];function o(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.a)(),...n.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(e.header,{children:(0,t.jsx)(e.h1,{id:"pad",children:"pad"})}),"\n",(0,t.jsxs)(e.blockquote,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L194",children:"pad(img: np.ndarray, pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]], fill_value: Optional[Union[int, Tuple[int, int, int]]] = 0, pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT) -> np.ndarray"})}),"\n"]}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsxs)(e.p,{children:[(0,t.jsx)(e.strong,{children:"\u8AAC\u660E"}),"\uFF1A\u5165\u529B\u753B\u50CF\u306B\u5BFE\u3057\u3066\u30D1\u30C7\u30A3\u30F3\u30B0\u51E6\u7406\u3092\u884C\u3044\u307E\u3059\u3002"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"\u5F15\u6570"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"img"})," (",(0,t.jsx)(e.code,{children:"np.ndarray"}),")\uFF1A\u30D1\u30C7\u30A3\u30F3\u30B0\u51E6\u7406\u3092\u884C\u3046\u5165\u529B\u753B\u50CF\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"pad_size"})," (",(0,t.jsx)(e.code,{children:"Union[int, Tuple[int, int], Tuple[int, int, int, int]]"}),")\uFF1A\u30D1\u30C7\u30A3\u30F3\u30B0\u306E\u30B5\u30A4\u30BA\u3002\u6574\u6570\u3092\u6307\u5B9A\u3059\u308B\u3068\u3059\u3079\u3066\u306E\u8FBA\u306B\u540C\u3058\u30D1\u30C7\u30A3\u30F3\u30B0\u91CF\u304C\u9069\u7528\u3055\u308C\u3001\u30BF\u30D7\u30EB ",(0,t.jsx)(e.code,{children:"(pad_top, pad_bottom, pad_left, pad_right)"})," \u3067\u5404\u8FBA\u306B\u7570\u306A\u308B\u30D1\u30C7\u30A3\u30F3\u30B0\u91CF\u3092\u6307\u5B9A\u3067\u304D\u307E\u3059\u3002\u3082\u3057\u304F\u306F ",(0,t.jsx)(e.code,{children:"(pad_height, pad_width)"})," \u3068\u3057\u3066\u9AD8\u3055\u3068\u5E45\u306B\u540C\u3058\u30D1\u30C7\u30A3\u30F3\u30B0\u91CF\u3092\u6307\u5B9A\u3059\u308B\u3053\u3068\u3082\u3067\u304D\u307E\u3059\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"fill_value"})," (",(0,t.jsx)(e.code,{children:"Optional[Union[int, Tuple[int, int, int]]]"}),")\uFF1A\u30D1\u30C7\u30A3\u30F3\u30B0\u306B\u4F7F\u7528\u3059\u308B\u5024\u3002\u30AB\u30E9\u30FC\u753B\u50CF\uFF083 \u30C1\u30E3\u30CD\u30EB\uFF09\u306E\u5834\u5408\u3001",(0,t.jsx)(e.code,{children:"fill_value"})," \u306F\u6574\u6570\u307E\u305F\u306F ",(0,t.jsx)(e.code,{children:"(R, G, B)"})," \u306E\u30BF\u30D7\u30EB\u3067\u6307\u5B9A\u3067\u304D\u307E\u3059\u3002\u30B0\u30EC\u30FC\u30B9\u30B1\u30FC\u30EB\u753B\u50CF\uFF081 \u30C1\u30E3\u30CD\u30EB\uFF09\u306E\u5834\u5408\u3001",(0,t.jsx)(e.code,{children:"fill_value"})," \u306F\u6574\u6570\u3067\u3059\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F 0\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"pad_mode"})," (",(0,t.jsx)(e.code,{children:"Union[str, int, BORDER]"}),")\uFF1A\u30D1\u30C7\u30A3\u30F3\u30B0\u30E2\u30FC\u30C9\u3002\u9078\u629E\u80A2\u306F\u4EE5\u4E0B\u306E\u901A\u308A\u3067\u3059\uFF1A","\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.code,{children:"BORDER.CONSTANT"}),"\uFF1A\u5B9A\u6570\u5024\uFF08",(0,t.jsx)(e.code,{children:"fill_value"}),"\uFF09\u3092\u4F7F\u7528\u3057\u3066\u30D1\u30C7\u30A3\u30F3\u30B0\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.code,{children:"BORDER.REPLICATE"}),"\uFF1A\u7AEF\u306E\u30D4\u30AF\u30BB\u30EB\u3092\u30B3\u30D4\u30FC\u3057\u3066\u30D1\u30C7\u30A3\u30F3\u30B0\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.code,{children:"BORDER.REFLECT"}),"\uFF1A\u7AEF\u3092\u53CD\u5C04\u3055\u305B\u3066\u30D1\u30C7\u30A3\u30F3\u30B0\u3002"]}),"\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.code,{children:"BORDER.REFLECT101"}),"\uFF1A\u7AEF\u3092\u53CD\u5C04\u3055\u305B\u3001\u4EBA\u5DE5\u7684\u306A\u75D5\u8DE1\u3092\u907F\u3051\u308B\u305F\u3081\u306B\u5FAE\u8ABF\u6574\u3057\u3066\u30D1\u30C7\u30A3\u30F3\u30B0\u3002\n\u30C7\u30D5\u30A9\u30EB\u30C8\u306F ",(0,t.jsx)(e.code,{children:"BORDER.CONSTANT"}),"\u3002"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"\u623B\u308A\u5024"})}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsxs)(e.li,{children:[(0,t.jsx)(e.strong,{children:"np.ndarray"}),"\uFF1A\u30D1\u30C7\u30A3\u30F3\u30B0\u51E6\u7406\u5F8C\u306E\u753B\u50CF\u3002"]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(e.li,{children:["\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.strong,{children:"\u4F8B"})}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\nimg = cb.imread('lena.png')\npad_img = cb.pad(img, pad_size=20, fill_value=(255, 0, 0))\n\n# \u30D1\u30C7\u30A3\u30F3\u30B0\u5F8C\u306E\u753B\u50CF\u3092\u5143\u306E\u30B5\u30A4\u30BA\u306B\u30EA\u30B5\u30A4\u30BA\u3057\u3066\u8996\u899A\u5316\npad_img = cb.imresize(pad_img, [img.shape[0], img.shape[1]])\n"})}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.img,{alt:"pad",src:i(90328).Z+"",width:"426",height:"256"})}),"\n"]}),"\n"]})]})}function p(n={}){let{wrapper:e}={...(0,r.a)(),...n.components};return e?(0,t.jsx)(e,{...n,children:(0,t.jsx)(o,{...n})}):o(n)}},90328:function(n,e,i){i.d(e,{Z:function(){return s}});let s=i.p+"assets/images/test_pad-73035954aa862b84f070f6481de0c9d4.jpg"},50065:function(n,e,i){i.d(e,{Z:function(){return a},a:function(){return l}});var s=i(67294);let t={},r=s.createContext(t);function l(n){let e=s.useContext(r);return s.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function a(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:l(n.components),s.createElement(r.Provider,{value:e},n.children)}}}]);