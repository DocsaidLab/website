"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["62097"],{74550:function(e,t,r){r.r(t),r.d(t,{metadata:()=>n,contentTitle:()=>o,default:()=>p,assets:()=>c,toc:()=>u,frontMatter:()=>l});var n=JSON.parse('{"id":"capybara/index","title":"Capybara","description":"\u65B0\u305F\u306A\u958B\u767A\u8005\u3092\u8FCE\u3048\u3001DocsaidKit \u306E\u518D\u69CB\u7BC9\u306B\u53D6\u308A\u7D44\u307F\u3001\u3044\u304F\u3064\u304B\u306E\u90E8\u5206\u3092\u3053\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306B\u5206\u5272\u3057\u307E\u3057\u305F\u3002\u57FA\u672C\u7684\u306A\u5185\u5BB9\u306F\u5909\u308F\u308A\u307E\u305B\u3093\u304C\u3001\u4ECA\u5F8C\u3001\u4ED6\u306E\u30D1\u30C3\u30B1\u30FC\u30B8\u306E\u4F9D\u5B58\u95A2\u4FC2\u3082 Capybara \u306B\u79FB\u884C\u3057\u3066\u3044\u304D\u307E\u3059\u3002\u79FB\u884C\u304C\u5B8C\u4E86\u3059\u308B\u307E\u3067\u306F DocsaidKit \u306E\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u3092\u7D9A\u3051\u3001Capybara \u304C\u5B8C\u6210\u3057\u305F\u5F8C\u3001DocsaidKit \u306F\u5EC3\u6B62\u3055\u308C\u308B\u4E88\u5B9A\u3067\u3059\u3002","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/capybara/index.md","sourceDirName":"capybara","slug":"/capybara/","permalink":"/ja/docs/capybara/","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"\u56DE\u6E2C\u30B7\u30B9\u30C6\u30E0","permalink":"/ja/docs/autotraderx/masterlink/backtesting"},"next":{"title":"\u30A4\u30F3\u30C8\u30ED\u30C0\u30AF\u30B7\u30E7\u30F3","permalink":"/ja/docs/capybara/intro"}}'),a=r("85893"),i=r("50065"),s=r("94301");let l={},o="Capybara",c={},u=[];function d(e){let t={a:"a",admonition:"admonition",h1:"h1",header:"header",hr:"hr",img:"img",li:"li",p:"p",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(t.header,{children:(0,a.jsx)(t.h1,{id:"capybara",children:"Capybara"})}),"\n",(0,a.jsx)(t.admonition,{type:"info",children:(0,a.jsx)(t.p,{children:"\u65B0\u305F\u306A\u958B\u767A\u8005\u3092\u8FCE\u3048\u3001DocsaidKit \u306E\u518D\u69CB\u7BC9\u306B\u53D6\u308A\u7D44\u307F\u3001\u3044\u304F\u3064\u304B\u306E\u90E8\u5206\u3092\u3053\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306B\u5206\u5272\u3057\u307E\u3057\u305F\u3002\u57FA\u672C\u7684\u306A\u5185\u5BB9\u306F\u5909\u308F\u308A\u307E\u305B\u3093\u304C\u3001\u4ECA\u5F8C\u3001\u4ED6\u306E\u30D1\u30C3\u30B1\u30FC\u30B8\u306E\u4F9D\u5B58\u95A2\u4FC2\u3082 Capybara \u306B\u79FB\u884C\u3057\u3066\u3044\u304D\u307E\u3059\u3002\u79FB\u884C\u304C\u5B8C\u4E86\u3059\u308B\u307E\u3067\u306F DocsaidKit \u306E\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u3092\u7D9A\u3051\u3001Capybara \u304C\u5B8C\u6210\u3057\u305F\u5F8C\u3001DocsaidKit \u306F\u5EC3\u6B62\u3055\u308C\u308B\u4E88\u5B9A\u3067\u3059\u3002"})}),"\n",(0,a.jsx)(t.p,{children:"\u3053\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306E\u76EE\u7684\u306F\u3001\u753B\u50CF\u51E6\u7406\u3001\u30D5\u30A1\u30A4\u30EB\u64CD\u4F5C\u3001\u65E5\u5E38\u7684\u306A\u30E6\u30FC\u30C6\u30A3\u30EA\u30C6\u30A3\u3092\u63D0\u4F9B\u3059\u308B\u3053\u3068\u3067\u3059\u3002\u958B\u767A\u904E\u7A0B\u3067\u4F5C\u6210\u3055\u308C\u305F\u30C4\u30FC\u30EB\u30DC\u30C3\u30AF\u30B9\u3067\u3059\u3002"}),"\n",(0,a.jsxs)(t.ul,{children:["\n",(0,a.jsx)(t.li,{children:(0,a.jsx)(t.a,{href:"https://github.com/DocsaidLab/Capybara",children:(0,a.jsx)(t.strong,{children:"Capybara Github"})})}),"\n"]}),"\n",(0,a.jsx)(t.hr,{}),"\n",(0,a.jsx)(t.p,{children:(0,a.jsx)(t.img,{alt:"title",src:r(75984).Z+"",width:"1792",height:"1024"})}),"\n",(0,a.jsx)(t.hr,{}),"\n","\n",(0,a.jsx)(s.Z,{})]})}function p(e={}){let{wrapper:t}={...(0,i.a)(),...e.components};return t?(0,a.jsx)(t,{...e,children:(0,a.jsx)(d,{...e})}):d(e)}},75984:function(e,t,r){r.d(t,{Z:function(){return n}});let n=r.p+"assets/images/title-ee87288720a6b42d2bf5af1390acb501.webp"},94301:function(e,t,r){r.d(t,{Z:()=>b});var n=r("85893");r("67294");var a=r("67026"),i=r("69369"),s=r("83012"),l=r("43115"),o=r("63150"),c=r("96025"),u=r("34403");let d={cardContainer:"cardContainer_fWXF",cardTitle:"cardTitle_rnsV",cardDescription:"cardDescription_PWke"};function p(e){let{href:t,children:r}=e;return(0,n.jsx)(s.Z,{href:t,className:(0,a.Z)("card padding--lg",d.cardContainer),children:r})}function h(e){let{href:t,icon:r,title:i,description:s}=e;return(0,n.jsxs)(p,{href:t,children:[(0,n.jsxs)(u.Z,{as:"h2",className:(0,a.Z)("text--truncate",d.cardTitle),title:i,children:[r," ",i]}),s&&(0,n.jsx)("p",{className:(0,a.Z)("text--truncate",d.cardDescription),title:s,children:s})]})}function f(e){let{item:t}=e,r=(0,i.LM)(t),a=function(){let{selectMessage:e}=(0,l.c)();return t=>e(t,(0,c.I)({message:"1 item|{count} items",id:"theme.docs.DocCard.categoryDescription.plurals",description:"The default description for a category card in the generated index about how many items this category includes"},{count:t}))}();return r?(0,n.jsx)(h,{href:r,icon:"\uD83D\uDDC3\uFE0F",title:t.label,description:t.description??a(t.items.length)}):null}function m(e){let{item:t}=e,r=(0,o.Z)(t.href)?"\uD83D\uDCC4\uFE0F":"\uD83D\uDD17",a=(0,i.xz)(t.docId??void 0);return(0,n.jsx)(h,{href:t.href,icon:r,title:t.label,description:t.description??a?.description})}function x(e){let{item:t}=e;switch(t.type){case"link":return(0,n.jsx)(m,{item:t});case"category":return(0,n.jsx)(f,{item:t});default:throw Error(`unknown item type ${JSON.stringify(t)}`)}}function g(e){let{className:t}=e,r=(0,i.jA)();return(0,n.jsx)(b,{items:r.items,className:t})}function b(e){let{items:t,className:r}=e;if(!t)return(0,n.jsx)(g,{...e});let s=(0,i.MN)(t);return(0,n.jsx)("section",{className:(0,a.Z)("row",r),children:s.map((e,t)=>(0,n.jsx)("article",{className:"col col--6 margin-bottom--lg",children:(0,n.jsx)(x,{item:e})},t))})}},43115:function(e,t,r){r.d(t,{c:function(){return o}});var n=r(67294),a=r(2933);let i=["zero","one","two","few","many","other"];function s(e){return i.filter(t=>e.includes(t))}let l={locale:"en",pluralForms:s(["one","other"]),select:e=>1===e?"one":"other"};function o(){let e=function(){let{i18n:{currentLocale:e}}=(0,a.Z)();return(0,n.useMemo)(()=>{try{return function(e){let t=new Intl.PluralRules(e);return{locale:e,pluralForms:s(t.resolvedOptions().pluralCategories),select:e=>t.select(e)}}(e)}catch(t){return console.error(`Failed to use Intl.PluralRules for locale "${e}".
Docusaurus will fallback to the default (English) implementation.
Error: ${t.message}
`),l}},[e])}();return{selectMessage:(t,r)=>(function(e,t,r){let n=e.split("|");if(1===n.length)return n[0];n.length>r.pluralForms.length&&console.error(`For locale=${r.locale}, a maximum of ${r.pluralForms.length} plural forms are expected (${r.pluralForms.join(",")}), but the message contains ${n.length}: ${e}`);let a=r.select(t);return n[Math.min(r.pluralForms.indexOf(a),n.length-1)]})(r,t,e)}}},50065:function(e,t,r){r.d(t,{Z:function(){return l},a:function(){return s}});var n=r(67294);let a={},i=n.createContext(a);function s(e){let t=n.useContext(i);return n.useMemo(function(){return"function"==typeof e?e(t):{...t,...e}},[t,e])}function l(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:s(e.components),n.createElement(i.Provider,{value:t},e.children)}}}]);