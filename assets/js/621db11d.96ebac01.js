"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["39854"],{73111:function(e,t,s){s.r(t),s.d(t,{default:()=>p});var l=s("85893");s("67294");var a=s("67026"),r=s("79741"),n=s("84681"),o=s("58267"),i=s("51225"),c=s("84315"),u=s("34403"),h=s("21389");let d={authorListItem:"authorListItem_n3yI"};function g(e){let{author:t}=e;return(0,l.jsx)("li",{className:d.authorListItem,children:(0,l.jsx)(h.Z,{as:"h2",author:t,count:t.count})})}function m(e){let{authors:t}=e;return(0,l.jsx)("section",{className:(0,a.Z)("margin-vert--lg",d.authorsListSection),children:(0,l.jsx)("ul",{children:t.map(e=>(0,l.jsx)(g,{author:e},e.key))})})}function p(e){let{authors:t,sidebar:s}=e,h=(0,o.HV)();return(0,l.jsxs)(r.FG,{className:(0,a.Z)(n.k.wrapper.blogPages,n.k.page.blogAuthorsListPage),children:[(0,l.jsx)(r.d,{title:h}),(0,l.jsx)(c.Z,{tag:"blog_authors_list"}),(0,l.jsxs)(i.Z,{sidebar:s,children:[(0,l.jsx)(u.Z,{as:"h1",children:h}),(0,l.jsx)(m,{authors:t})]})]})}},58267:function(e,t,s){s.d(t,{HV:function(){return n},Wi:function(){return r}}),s(85893),s(67294);var l=s(96025),a=s(43115);function r(e){let t=function(){let{selectMessage:e}=(0,a.c)();return t=>e(t,(0,l.I)({id:"theme.blog.post.plurals",description:'Pluralized label for "{count} posts". Use as much plural forms (separated by "|") as your language support (see https://www.unicode.org/cldr/cldr-aux/charts/34/supplemental/language_plural_rules.html)',message:"One post|{count} posts"},{count:t}))}();return(0,l.I)({id:"theme.blog.tagTitle",description:"The title of the page for a blog tag",message:'{nPosts} tagged with "{tagName}"'},{nPosts:t(e.count),tagName:e.label})}let n=()=>(0,l.I)({id:"theme.blog.authorsList.pageTitle",message:"Authors",description:"The title of the authors page"})},51225:function(e,t,s){s.d(t,{Z:function(){return o}});var l=s(85893),a=s(69825),r=s(1568),n=s(67026);function o(e){let{sidebar:t,toc:s,children:o,hero:i,...c}=e,u=t&&t.items.length>0;return(0,l.jsxs)(r.Z,{...c,children:[i&&(0,l.jsx)("div",{className:"blog-hero-fullwidth",children:i}),(0,l.jsx)("div",{className:"container margin-vert--lg",children:(0,l.jsxs)("div",{className:"row",children:[(0,l.jsx)("main",{className:(0,n.Z)("col",{"col--9":u,"col--9 col--offset-1":!u}),children:o}),(0,l.jsx)(a.Z,{sidebar:t}),s&&(0,l.jsx)("div",{className:"col col--2",children:s})]})})]})}s(67294)}}]);