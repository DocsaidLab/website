"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["35754"],{71236:function(n,e,i){i.r(e),i.d(e,{metadata:()=>r,contentTitle:()=>d,default:()=>h,assets:()=>o,toc:()=>t,frontMatter:()=>l});var r=JSON.parse('{"id":"capybara/pipconfig","title":"PIP \u30D1\u30E9\u30E1\u30FC\u30BF\u8A2D\u5B9A","description":"\u3053\u306E\u7AE0\u3067\u306F\u3001pip \u306E\u8A2D\u5B9A\u30E1\u30AB\u30CB\u30BA\u30E0\u306B\u3064\u3044\u3066\u8A73\u3057\u304F\u8AAC\u660E\u3057\u3001\u8907\u6570\u306E Python \u74B0\u5883\u3067\u30D1\u30C3\u30B1\u30FC\u30B8\u306E\u7AF6\u5408\u3084\u6A29\u9650\u306E\u554F\u984C\u3092\u907F\u3051\u308B\u65B9\u6CD5\u3092\u7D39\u4ECB\u3057\u307E\u3059\u3002","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/pipconfig.md","sourceDirName":"capybara","slug":"/capybara/pipconfig","permalink":"/ja/docs/capybara/pipconfig","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"sidebarPosition":4,"frontMatter":{"sidebar_position":4},"sidebar":"tutorialSidebar","previous":{"title":"\u9032\u968E\u30A4\u30F3\u30B9\u30C8\u30FC\u30EB","permalink":"/ja/docs/capybara/advance"},"next":{"title":"\u51FD\u5F0F\u5EAB","permalink":"/ja/docs/category/\u51FD\u5F0F\u5EAB"}}'),s=i("85893"),c=i("50065");let l={sidebar_position:4},d="PIP \u30D1\u30E9\u30E1\u30FC\u30BF\u8A2D\u5B9A",o={},t=[{value:"\u4F7F\u7528\u65B9\u6CD5",id:"\u4F7F\u7528\u65B9\u6CD5",level:2},{value:"\u512A\u5148\u9806\u4F4D",id:"\u512A\u5148\u9806\u4F4D",level:2},{value:"\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u4F8B",id:"\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u4F8B",level:2}];function p(n){let e={admonition:"admonition",code:"code",h1:"h1",h2:"h2",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,c.a)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.header,{children:(0,s.jsx)(e.h1,{id:"pip-\u30D1\u30E9\u30E1\u30FC\u30BF\u8A2D\u5B9A",children:"PIP \u30D1\u30E9\u30E1\u30FC\u30BF\u8A2D\u5B9A"})}),"\n",(0,s.jsx)(e.p,{children:"\u3053\u306E\u7AE0\u3067\u306F\u3001pip \u306E\u8A2D\u5B9A\u30E1\u30AB\u30CB\u30BA\u30E0\u306B\u3064\u3044\u3066\u8A73\u3057\u304F\u8AAC\u660E\u3057\u3001\u8907\u6570\u306E Python \u74B0\u5883\u3067\u30D1\u30C3\u30B1\u30FC\u30B8\u306E\u7AF6\u5408\u3084\u6A29\u9650\u306E\u554F\u984C\u3092\u907F\u3051\u308B\u65B9\u6CD5\u3092\u7D39\u4ECB\u3057\u307E\u3059\u3002"}),"\n",(0,s.jsx)(e.h2,{id:"\u4F7F\u7528\u65B9\u6CD5",children:"\u4F7F\u7528\u65B9\u6CD5"}),"\n",(0,s.jsx)(e.p,{children:"Linux/macOS \u30B7\u30B9\u30C6\u30E0\u3067\u306F\u3001\u6B21\u306E\u30B3\u30DE\u30F3\u30C9\u3092\u4F7F\u7528\u3057\u3066\u30ED\u30FC\u30AB\u30EB\u3068\u30B0\u30ED\u30FC\u30D0\u30EB\u8A2D\u5B9A\u3092\u7BA1\u7406\u3067\u304D\u307E\u3059\uFF1A"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-bash",children:"python -m pip config [<file-option>] list\npython -m pip config [<file-option>] [--editor <editor-path>] edit\n"})}),"\n",(0,s.jsxs)(e.p,{children:["\u3053\u3053\u3067\u3001",(0,s.jsx)(e.code,{children:"<file-option>"})," \u306F\u6B21\u306E\u30AA\u30D7\u30B7\u30E7\u30F3\u3092\u6307\u5B9A\u3067\u304D\u307E\u3059\uFF1A"]}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"--global"}),"\uFF1A\u30B7\u30B9\u30C6\u30E0\u5168\u4F53\u306E\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u6307\u5B9A\u3057\u307E\u3059\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"--user"}),"\uFF1A\u30E6\u30FC\u30B6\u30FC\u5358\u4F4D\u306E\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u6307\u5B9A\u3057\u307E\u3059\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"--site"}),"\uFF1A\u73FE\u5728\u306E\u4EEE\u60F3\u74B0\u5883\u5185\u306E\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u6307\u5B9A\u3057\u307E\u3059\u3002"]}),"\n"]}),"\n",(0,s.jsxs)(e.p,{children:[(0,s.jsx)(e.code,{children:"--editor"})," \u30D1\u30E9\u30E1\u30FC\u30BF\u3092\u4F7F\u7528\u3059\u308B\u3068\u3001\u5916\u90E8\u30A8\u30C7\u30A3\u30BF\u306E\u30D1\u30B9\u3092\u6307\u5B9A\u3067\u304D\u307E\u3059\u3002\u3053\u306E\u30D1\u30E9\u30E1\u30FC\u30BF\u3092\u6307\u5B9A\u3057\u306A\u3044\u5834\u5408\u3001",(0,s.jsx)(e.code,{children:"VISUAL"})," \u307E\u305F\u306F ",(0,s.jsx)(e.code,{children:"EDITOR"})," \u74B0\u5883\u5909\u6570\u306B\u57FA\u3065\u3044\u3066\u30C7\u30D5\u30A9\u30EB\u30C8\u306E\u30C6\u30AD\u30B9\u30C8\u30A8\u30C7\u30A3\u30BF\u304C\u4F7F\u7528\u3055\u308C\u307E\u3059\u3002"]}),"\n",(0,s.jsx)(e.p,{children:"\u4F8B\u3048\u3070\u3001Vim \u30A8\u30C7\u30A3\u30BF\u3092\u4F7F\u7528\u3057\u3066\u30B0\u30ED\u30FC\u30D0\u30EB\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u7DE8\u96C6\u3057\u305F\u3044\u5834\u5408\u3001\u6B21\u306E\u30B3\u30DE\u30F3\u30C9\u3092\u4F7F\u7528\u3057\u307E\u3059\uFF1A"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-bash",children:"python -m pip config --global --editor vim edit\n"})}),"\n",(0,s.jsx)(e.admonition,{type:"tip",children:(0,s.jsxs)(e.p,{children:["Windows \u30B7\u30B9\u30C6\u30E0\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u5834\u5408\u3001\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306F ",(0,s.jsx)(e.code,{children:"%APPDATA%\\pip\\pip.ini"})," \u306B\u3042\u308B\u304B\u3001",(0,s.jsx)(e.code,{children:"%HOMEPATH%\\.pip\\pip.ini"})," \u306E\u3088\u3046\u306A\u30D1\u30B9\u3067\u78BA\u8A8D\u3067\u304D\u307E\u3059\u3002\u516C\u5F0F\u30C9\u30AD\u30E5\u30E1\u30F3\u30C8\u3092\u53C2\u7167\u3059\u308B\u304B\u3001",(0,s.jsx)(e.code,{children:"pip config list"})," \u30B3\u30DE\u30F3\u30C9\u3092\u4F7F\u7528\u3057\u3066\u5B9F\u969B\u306E\u5834\u6240\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002"]})}),"\n",(0,s.jsx)(e.h2,{id:"\u512A\u5148\u9806\u4F4D",children:"\u512A\u5148\u9806\u4F4D"}),"\n",(0,s.jsx)(e.p,{children:"\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u512A\u5148\u9806\u4F4D\u306F\u975E\u5E38\u306B\u91CD\u8981\u3067\u3059\u3002\u6B21\u306F\u3001\u3042\u306A\u305F\u306E\u30DE\u30B7\u30F3\u306B\u5B58\u5728\u3059\u308B\u53EF\u80FD\u6027\u306E\u3042\u308B\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u512A\u5148\u9806\u4F4D\u9806\u306B\u30EA\u30B9\u30C8\u3057\u305F\u3082\u306E\u3067\u3059\uFF1A"}),"\n",(0,s.jsxs)(e.ol,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"\u30B5\u30A4\u30C8\u30EC\u30D9\u30EB\u306E\u30D5\u30A1\u30A4\u30EB"}),"\uFF1A","\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsx)(e.li,{children:(0,s.jsx)(e.code,{children:"/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf"})}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"\u30E6\u30FC\u30B6\u30FC\u30EC\u30D9\u30EB\u306E\u30D5\u30A1\u30A4\u30EB"}),"\uFF1A","\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsx)(e.li,{children:(0,s.jsx)(e.code,{children:"/home/user/.config/pip/pip.conf"})}),"\n",(0,s.jsx)(e.li,{children:(0,s.jsx)(e.code,{children:"/home/user/.pip/pip.conf"})}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.strong,{children:"\u30B0\u30ED\u30FC\u30D0\u30EB\u30EC\u30D9\u30EB\u306E\u30D5\u30A1\u30A4\u30EB"}),"\uFF1A","\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsx)(e.li,{children:(0,s.jsx)(e.code,{children:"/etc/pip.conf"})}),"\n",(0,s.jsx)(e.li,{children:(0,s.jsx)(e.code,{children:"/etc/xdg/pip/pip.conf"})}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(e.p,{children:"Python \u74B0\u5883\u3067\u306F\u3001pip \u306F\u3053\u306E\u9806\u5E8F\u3067\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u63A2\u3057\u3001\u9069\u7528\u3057\u307E\u3059\u3002"}),"\n",(0,s.jsx)(e.p,{children:"\u3069\u306E\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3092\u7DE8\u96C6\u3057\u3066\u3044\u308B\u306E\u304B\u3092\u78BA\u8A8D\u3059\u308B\u3053\u3068\u306F\u3001\u8FFD\u8DE1\u304C\u96E3\u3057\u3044\u30A8\u30E9\u30FC\u3092\u907F\u3051\u308B\u305F\u3081\u306B\u91CD\u8981\u3067\u3059\u3002"}),"\n",(0,s.jsx)(e.h2,{id:"\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u4F8B",children:"\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u4F8B"}),"\n",(0,s.jsx)(e.p,{children:"\u4EE5\u4E0B\u306F\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u306E\u4F8B\u3067\u3059\uFF1A"}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-ini",children:"[global]\nindex-url = https://pypi.org/simple\ntrusted-host = pypi.org\n               pypi.python.org\n               files.pythonhosted.org\nextra-index-url = https://pypi.anaconda.org/simple\n"})}),"\n",(0,s.jsx)(e.p,{children:"\u3053\u306E\u8A2D\u5B9A\u30D5\u30A1\u30A4\u30EB\u3067\u306F\u3001\u5404\u30D1\u30E9\u30E1\u30FC\u30BF\u306E\u610F\u5473\u306F\u6B21\u306E\u901A\u308A\u3067\u3059\uFF1A"}),"\n",(0,s.jsxs)(e.ul,{children:["\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"index-url"}),"\uFF1Apip \u304C\u30D1\u30C3\u30B1\u30FC\u30B8\u3092\u30A4\u30F3\u30B9\u30C8\u30FC\u30EB\u3059\u308B\u969B\u306B\u4F7F\u7528\u3059\u308B\u30C7\u30D5\u30A9\u30EB\u30C8\u306E\u30EA\u30DD\u30B8\u30C8\u30EA\u3092\u8A2D\u5B9A\u3057\u307E\u3059\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"trusted-host"}),"\uFF1AHTTPS \u3092\u4F7F\u7528\u3057\u3066\u5B89\u5168\u78BA\u8A8D\u3092\u884C\u3046\u5FC5\u8981\u306E\u306A\u3044\u30DB\u30B9\u30C8\u3092\u30EA\u30B9\u30C8\u3057\u3001SSL \u30A8\u30E9\u30FC\u3092\u9632\u304E\u307E\u3059\u3002"]}),"\n",(0,s.jsxs)(e.li,{children:[(0,s.jsx)(e.code,{children:"extra-index-url"}),"\uFF1A\u4F9D\u5B58\u95A2\u4FC2\u30D1\u30C3\u30B1\u30FC\u30B8\u3092\u691C\u7D22\u304A\u3088\u3073\u30A4\u30F3\u30B9\u30C8\u30FC\u30EB\u3059\u308B\u305F\u3081\u306E\u8FFD\u52A0\u306E\u30BD\u30FC\u30B9 URL \u3092\u63D0\u4F9B\u3057\u307E\u3059\u3002",(0,s.jsx)(e.code,{children:"index-url"})," \u3068\u7570\u306A\u308A\u3001",(0,s.jsx)(e.code,{children:"index-url"})," \u3067\u898B\u3064\u304B\u3089\u306A\u3044\u30D1\u30C3\u30B1\u30FC\u30B8\u304C\u3042\u308B\u5834\u5408\u3001pip \u306F ",(0,s.jsx)(e.code,{children:"extra-index-url"})," \u3092\u53C2\u7167\u3057\u3066\u63A2\u3057\u307E\u3059\u3002"]}),"\n"]}),"\n",(0,s.jsx)(e.admonition,{type:"warning",children:(0,s.jsx)(e.p,{children:"\u8907\u6570\u306E\u30BD\u30FC\u30B9\u3092\u4F7F\u7528\u3059\u308B\u5834\u5408\u3001\u3059\u3079\u3066\u306E\u30BD\u30FC\u30B9\u304C\u4FE1\u983C\u3067\u304D\u308B\u3082\u306E\u3067\u3042\u308B\u3079\u304D\u3067\u3059\u3002\u306A\u305C\u306A\u3089\u3001\u30A4\u30F3\u30B9\u30C8\u30FC\u30EB\u904E\u7A0B\u3067\u3053\u308C\u3089\u306E\u30BD\u30FC\u30B9\u304B\u3089\u6700\u9069\u306A\u30D0\u30FC\u30B8\u30E7\u30F3\u304C\u9078\u629E\u3055\u308C\u308B\u304B\u3089\u3067\u3059\u3002\u4FE1\u983C\u3055\u308C\u3066\u3044\u306A\u3044\u30BD\u30FC\u30B9\u306F\u30BB\u30AD\u30E5\u30EA\u30C6\u30A3\u30EA\u30B9\u30AF\u3092\u4F34\u3046\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\u3002"})}),"\n",(0,s.jsx)(e.admonition,{type:"tip",children:(0,s.jsxs)(e.p,{children:["\u30D7\u30E9\u30A4\u30D9\u30FC\u30C8\u306A\u30D1\u30C3\u30B1\u30FC\u30B8\u30B5\u30FC\u30D0\u30FC\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u5834\u5408\u3084\u3001\u8A8D\u8A3C\u306E\u305F\u3081\u306B\u30E6\u30FC\u30B6\u30FC\u540D\u3068\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u6307\u5B9A\u3059\u308B\u5FC5\u8981\u304C\u3042\u308B\u5834\u5408\u306F\u3001",(0,s.jsx)(e.code,{children:"pip.conf"})," \u306B\u305D\u306E\u60C5\u5831\u3092\u8A18\u8F09\u3057\u3066\u81EA\u52D5\u5316\u3067\u304D\u307E\u3059\u304C\u3001\u30D5\u30A1\u30A4\u30EB\u306E\u6A29\u9650\u3092\u9069\u5207\u306B\u7BA1\u7406\u3057\u3066\u5B89\u5168\u3092\u78BA\u4FDD\u3057\u3066\u304F\u3060\u3055\u3044\u3002"]})})]})}function h(n={}){let{wrapper:e}={...(0,c.a)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(p,{...n})}):p(n)}},50065:function(n,e,i){i.d(e,{Z:function(){return d},a:function(){return l}});var r=i(67294);let s={},c=r.createContext(s);function l(n){let e=r.useContext(c);return r.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function d(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:l(n.components),r.createElement(c.Provider,{value:e},n.children)}}}]);