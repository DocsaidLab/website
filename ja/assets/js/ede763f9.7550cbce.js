"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["79115"],{56869:function(n,o,s){s.r(o),s.d(o,{metadata:()=>l,contentTitle:()=>t,default:()=>h,assets:()=>c,toc:()=>d,frontMatter:()=>i});var l=JSON.parse('{"id":"capybara/funcs/structures/polygon","title":"Polygon","description":"Polygon(array bool = False)","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/funcs/structures/polygon.md","sourceDirName":"capybara/funcs/structures","slug":"/capybara/funcs/structures/polygon","permalink":"/ja/docs/capybara/funcs/structures/polygon","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"sidebarPosition":4,"frontMatter":{"sidebar_position":4},"sidebar":"tutorialSidebar","previous":{"title":"Boxes","permalink":"/ja/docs/capybara/funcs/structures/boxes"},"next":{"title":"Polygons","permalink":"/ja/docs/capybara/funcs/structures/polygons"}}'),e=s("85893"),r=s("50065");let i={sidebar_position:4},t="Polygon",c={},d=[];function a(n){let o={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.a)(),...n.components};return(0,e.jsxs)(e.Fragment,{children:[(0,e.jsx)(o.header,{children:(0,e.jsx)(o.h1,{id:"polygon",children:"Polygon"})}),"\n",(0,e.jsxs)(o.blockquote,{children:["\n",(0,e.jsx)(o.p,{children:(0,e.jsx)(o.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/structures/polygons.py#L64",children:"Polygon(array: _Polygon, normalized: bool = False)"})}),"\n"]}),"\n",(0,e.jsxs)(o.ul,{children:["\n",(0,e.jsxs)(o.li,{children:["\n",(0,e.jsxs)(o.p,{children:[(0,e.jsx)(o.strong,{children:"\u8AAC\u660E"}),"\uFF1A"]}),"\n",(0,e.jsxs)(o.p,{children:[(0,e.jsx)(o.code,{children:"Polygon"})," \u306F\u3001\u591A\u89D2\u5F62\u3092\u8868\u3059\u30AF\u30E9\u30B9\u3067\u3059\u3002\u3053\u306E\u30AF\u30E9\u30B9\u306F\u3001\u591A\u89D2\u5F62\u306E\u5EA7\u6A19\u3092\u64CD\u4F5C\u3059\u308B\u305F\u3081\u306E\u591A\u304F\u306E\u30E1\u30BD\u30C3\u30C9\u3092\u63D0\u4F9B\u3057\u307E\u3059\u3002\u4F8B\u3048\u3070\u3001\u5EA7\u6A19\u306E\u6B63\u898F\u5316\u3001\u975E\u6B63\u898F\u5316\u3001\u591A\u89D2\u5F62\u306E\u30AF\u30EA\u30C3\u30D4\u30F3\u30B0\u3001\u79FB\u52D5\u3001\u30B9\u30B1\u30FC\u30EA\u30F3\u30B0\u3001\u51F8\u591A\u89D2\u5F62\u3078\u306E\u5909\u63DB\u3001\u6700\u5C0F\u5916\u63A5\u77E9\u5F62\u3078\u306E\u5909\u63DB\u3001\u5883\u754C\u30DC\u30C3\u30AF\u30B9\u3078\u306E\u5909\u63DB\u306A\u3069\u3067\u3059\u3002"]}),"\n"]}),"\n",(0,e.jsxs)(o.li,{children:["\n",(0,e.jsx)(o.p,{children:(0,e.jsx)(o.strong,{children:"\u30D1\u30E9\u30E1\u30FC\u30BF"})}),"\n",(0,e.jsxs)(o.ul,{children:["\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"array"})," (",(0,e.jsx)(o.code,{children:"_Polygon"}),")\uFF1A\u591A\u89D2\u5F62\u306E\u5EA7\u6A19\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"normalized"})," (",(0,e.jsx)(o.code,{children:"bool"}),")\uFF1A\u591A\u89D2\u5F62\u306E\u5EA7\u6A19\u304C\u6B63\u898F\u5316\u3055\u308C\u3066\u3044\u308B\u304B\u3069\u3046\u304B\u3092\u793A\u3059\u30D5\u30E9\u30B0\u3002\u30C7\u30D5\u30A9\u30EB\u30C8\u306F ",(0,e.jsx)(o.code,{children:"False"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,e.jsxs)(o.li,{children:["\n",(0,e.jsx)(o.p,{children:(0,e.jsx)(o.strong,{children:"\u5C5E\u6027"})}),"\n",(0,e.jsxs)(o.ul,{children:["\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"normalized"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u6B63\u898F\u5316\u72B6\u614B\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"moments"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u30E2\u30FC\u30E1\u30F3\u30C8\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"area"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u9762\u7A4D\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"arclength"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u5468\u56F2\u9577\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"centroid"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u8CEA\u91CF\u4E2D\u5FC3\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"boundingbox"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u5883\u754C\u30DC\u30C3\u30AF\u30B9\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"min_circle"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u6700\u5C0F\u5916\u63A5\u5186\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"min_box"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u6700\u5C0F\u5916\u63A5\u77E9\u5F62\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"orientation"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u5411\u304D\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"min_box_wh"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u6700\u5C0F\u5916\u63A5\u77E9\u5F62\u306E\u5E45\u3068\u9AD8\u3055\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"extent"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u5360\u6709\u7387\u3092\u53D6\u5F97\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"solidity"}),"\uFF1A\u591A\u89D2\u5F62\u306E\u5145\u5B9F\u5EA6\uFF08solidity\uFF09\u3092\u53D6\u5F97\u3002"]}),"\n"]}),"\n"]}),"\n",(0,e.jsxs)(o.li,{children:["\n",(0,e.jsx)(o.p,{children:(0,e.jsx)(o.strong,{children:"\u30E1\u30BD\u30C3\u30C9"})}),"\n",(0,e.jsxs)(o.ul,{children:["\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"copy"}),"()\uFF1A\u591A\u89D2\u5F62\u30AA\u30D6\u30B8\u30A7\u30AF\u30C8\u3092\u30B3\u30D4\u30FC\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"numpy"}),"()\uFF1A\u591A\u89D2\u5F62\u30AA\u30D6\u30B8\u30A7\u30AF\u30C8\u3092 numpy \u914D\u5217\u306B\u5909\u63DB\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"normalize"}),"(",(0,e.jsx)(o.code,{children:"w: float, h: float"}),")\uFF1A\u591A\u89D2\u5F62\u306E\u5EA7\u6A19\u3092\u6B63\u898F\u5316\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"denormalize"}),"(",(0,e.jsx)(o.code,{children:"w: float, h: float"}),")\uFF1A\u591A\u89D2\u5F62\u306E\u5EA7\u6A19\u3092\u975E\u6B63\u898F\u5316\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"clip"}),"(",(0,e.jsx)(o.code,{children:"xmin: int, ymin: int, xmax: int, ymax: int"}),")\uFF1A\u591A\u89D2\u5F62\u3092\u30AF\u30EA\u30C3\u30D4\u30F3\u30B0\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"shift"}),"(",(0,e.jsx)(o.code,{children:"shift_x: float, shift_y: float"}),")\uFF1A\u591A\u89D2\u5F62\u3092\u79FB\u52D5\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"scale"}),"(",(0,e.jsx)(o.code,{children:"distance: int, join_style: JOIN_STYLE = JOIN_STYLE.mitre"}),")\uFF1A\u591A\u89D2\u5F62\u3092\u30B9\u30B1\u30FC\u30EA\u30F3\u30B0\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"to_convexhull"}),"()\uFF1A\u591A\u89D2\u5F62\u3092\u51F8\u591A\u89D2\u5F62\u306B\u5909\u63DB\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"to_min_boxpoints"}),"()\uFF1A\u591A\u89D2\u5F62\u3092\u6700\u5C0F\u5916\u63A5\u77E9\u5F62\u306E\u5EA7\u6A19\u306B\u5909\u63DB\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"to_box"}),"(",(0,e.jsx)(o.code,{children:"box_mode: str = 'xyxy'"}),")\uFF1A\u591A\u89D2\u5F62\u3092\u5883\u754C\u30DC\u30C3\u30AF\u30B9\u306B\u5909\u63DB\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"to_list"}),"(",(0,e.jsx)(o.code,{children:"flatten: bool = False"}),")\uFF1A\u591A\u89D2\u5F62\u3092\u30EA\u30B9\u30C8\u306B\u5909\u63DB\u3002"]}),"\n",(0,e.jsxs)(o.li,{children:[(0,e.jsx)(o.strong,{children:"is_empty"}),"(",(0,e.jsx)(o.code,{children:"threshold: int = 3"}),")\uFF1A\u591A\u89D2\u5F62\u304C\u7A7A\u304B\u3069\u3046\u304B\u3092\u5224\u5B9A\u3002"]}),"\n"]}),"\n"]}),"\n",(0,e.jsxs)(o.li,{children:["\n",(0,e.jsx)(o.p,{children:(0,e.jsx)(o.strong,{children:"\u4F8B"})}),"\n",(0,e.jsx)(o.pre,{children:(0,e.jsx)(o.code,{className:"language-python",children:"import capybara as cb\n\npolygon = cb.Polygon([[10., 20.], [50, 20.], [50, 80.], [10., 80.]])\nprint(polygon)\n# >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])\n\npolygon1 = polygon.normalize(100, 100)\nprint(polygon1)\n# >>> Polygon([[0.1 0.2], [0.5 0.2], [0.5 0.8], [0.1 0.8]])\n\npolygon2 = polygon.denormalize(100, 100)\nprint(polygon2)\n# >>> Polygon([[1000. 2000.], [5000. 2000.], [5000. 8000.], [1000. 8000.]])\n\npolygon3 = polygon.clip(20, 20, 60, 60)\nprint(polygon3)\n# >>> Polygon([[20. 20.], [50. 20.], [50. 60.], [20. 60.]])\n\npolygon4 = polygon.shift(10, 10)\nprint(polygon4)\n# >>> Polygon([[20. 30.], [60. 30.], [60. 90.], [20. 90.]])\n\npolygon5 = polygon.scale(10)\nprint(polygon5)\n# >>> Polygon([[0. 10.], [60. 10.], [60. 90.], [0. 90.]])\n\npolygon6 = polygon.to_convexhull()\nprint(polygon6)\n# >>> Polygon([[50. 80.], [10. 80.], [10. 20.], [50. 20.]])\n\npolygon7 = polygon.to_min_boxpoints()\nprint(polygon7)\n# >>> Polygon([[10. 20.], [50. 20.], [50. 80.], [10. 80.]])\n\npolygon8 = polygon.to_box('xywh')\nprint(polygon8)\n# >>> Box([10. 20. 40. 60.]), BoxMode.XYWH\n\npolygon9 = polygon.to_list()\nprint(polygon9)\n# >>> [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]\n\npolygon10 = polygon.is_empty()\nprint(polygon10)\n# >>> False\n"})}),"\n"]}),"\n"]})]})}function h(n={}){let{wrapper:o}={...(0,r.a)(),...n.components};return o?(0,e.jsx)(o,{...n,children:(0,e.jsx)(a,{...n})}):a(n)}},50065:function(n,o,s){s.d(o,{Z:function(){return t},a:function(){return i}});var l=s(67294);let e={},r=l.createContext(e);function i(n){let o=l.useContext(r);return l.useMemo(function(){return"function"==typeof n?n(o):{...o,...n}},[o,n])}function t(n){let o;return o=n.disableParentContext?"function"==typeof n.components?n.components(e):n.components||e:i(n.components),l.createElement(r.Provider,{value:o},n.children)}}}]);