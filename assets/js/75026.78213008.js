"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["75026"],{41835:function(e,t,a){a.d(t,{wj:()=>i,nO:()=>c,iZ:()=>h,cH:()=>v,n4:()=>u,Ne:()=>C,ci:()=>N});var r=a("85893"),l=a("67294"),n=a("85346"),s=a("78312");function i(){let e=(0,s.Z)(),t=e?.data?.blogMetadata;if(!t)throw Error("useBlogMetadata() can't be called on the current route because the blog metadata could not be found in route context");return t}let o=l.createContext(null);function u(e){let{children:t,content:a,isBlogPostPage:n=!1}=e,s=function(e){let{content:t,isBlogPostPage:a}=e;return(0,l.useMemo)(()=>({metadata:t.metadata,frontMatter:t.frontMatter,assets:t.assets,toc:t.toc,isBlogPostPage:a}),[t,a])}({content:a,isBlogPostPage:n});return(0,r.jsx)(o.Provider,{value:s,children:t})}function c(){let e=(0,l.useContext)(o);if(null===e)throw new n.i6("BlogPostProvider");return e}var m=a("4757"),g=a("2933");let d=e=>new Date(e).toISOString();function h(){var e,t,a;let r=i(),{assets:l,metadata:n}=c(),{siteConfig:s}=(0,g.Z)(),{withBaseUrl:o}=(0,m.Cg)(),{date:u,title:h,description:b,frontMatter:f,lastUpdatedAt:j}=n,x=l.image??f.image,v=f.keywords??[],N=j?d(j):void 0,C=`${s.url}${n.permalink}`;return{"@context":"https://schema.org","@type":"BlogPosting","@id":C,mainEntityOfPage:C,url:C,headline:h,name:h,description:b,datePublished:u,...N?{dateModified:N}:{},...function(e){let t=e.map(p);return{author:1===t.length?t[0]:t}}(n.authors),...(e=x,t=o,a=h,e?{image:function(e){let{imageUrl:t,caption:a}=e;return{"@type":"ImageObject","@id":t,url:t,contentUrl:t,caption:a}}({imageUrl:t(e,{absolute:!0}),caption:`title image for the blog post: ${a}`})}:{}),...v?{keywords:v}:{},isPartOf:{"@type":"Blog","@id":`${s.url}${r.blogBasePath}`,name:r.blogTitle}}}function p(e){return{"@type":"Person",...e.name?{name:e.name}:{},...e.title?{description:e.title}:{},...e.url?{url:e.url}:{},...e.email?{email:e.email}:{},...e.imageURL?{image:e.imageURL}:{}}}var b=a("16550"),f=a("83012"),j=a("38341"),x=a("79246");function v(e){let{pathname:t}=(0,b.TH)();return(0,l.useMemo)(()=>e.filter(e=>{var a,r;return a=e,r=t,(!a.unlisted||!!(0,x.Mg)(a.permalink,r))&&!0}),[e,t])}function N(e){let t=Object.entries((0,j.vM)(e,e=>`${new Date(e.date).getFullYear()}`));return t.reverse(),t}function C(e){let{items:t,ulClassName:a,liClassName:l,linkClassName:n,linkActiveClassName:s}=e;return(0,r.jsx)("ul",{className:a,children:t.map(e=>(0,r.jsx)("li",{className:l,children:(0,r.jsx)(f.Z,{isNavLink:!0,to:e.permalink,className:n,activeClassName:s,children:e.title})},e.permalink))})}},69825:function(e,t,a){a.d(t,{Z:()=>P});var r=a("85893"),l=a("67294"),n=a("54704"),s=a("67026"),i=a("96025"),o=a("41835"),u=a("30140"),c=a("34403");function m(e){let{year:t,yearGroupHeadingClassName:a,children:l}=e;return(0,r.jsxs)("div",{role:"group",children:[(0,r.jsx)(c.Z,{as:"h3",className:a,children:t}),l]})}let g=(0,l.memo)(function(e){let{items:t,yearGroupHeadingClassName:a,ListComponent:l}=e;if(!(0,u.L)().blog.sidebar.groupByYear)return(0,r.jsx)(l,{items:t});{let e=(0,o.ci)(t);return(0,r.jsx)(r.Fragment,{children:e.map(e=>{let[t,n]=e;return(0,r.jsx)(m,{year:t,yearGroupHeadingClassName:a,children:(0,r.jsx)(l,{items:n})},t)})})}}),d="sidebar_re4s",h="sidebarItemTitle_pO2u",p="sidebarItemList_Yudw",b="sidebarItem__DBe",f="sidebarItemLink_mo7H",j="sidebarItemLinkActive_I1ZP",x="yearGroupHeading_rMGB",v=e=>{let{items:t}=e;return(0,r.jsx)(o.Ne,{items:t,ulClassName:(0,s.Z)(p,"clean-list"),liClassName:b,linkClassName:f,linkActiveClassName:j})},N=(0,l.memo)(function(e){let{sidebar:t}=e,a=(0,o.cH)(t.items);return(0,r.jsx)("aside",{className:"col col--3",children:(0,r.jsxs)("nav",{className:(0,s.Z)(d,"thin-scrollbar"),"aria-label":(0,i.I)({id:"theme.blog.sidebar.navAriaLabel",message:"Blog recent posts navigation",description:"The ARIA label for recent posts in the blog sidebar"}),children:[(0,r.jsx)("div",{className:(0,s.Z)(h,"margin-bottom--md"),children:t.title}),(0,r.jsx)(g,{items:a,ListComponent:v,yearGroupHeadingClassName:x})]})})});var C=a("11179");let _="yearGroupHeading_QT03",k=e=>{let{items:t}=e;return(0,r.jsx)(o.Ne,{items:t,ulClassName:"menu__list",liClassName:"menu__list-item",linkClassName:"menu__link",linkActiveClassName:"menu__link--active"})};function Z(e){let{sidebar:t}=e,a=(0,o.cH)(t.items);return(0,r.jsx)(g,{items:a,ListComponent:k,yearGroupHeadingClassName:_})}let y=(0,l.memo)(function(e){return(0,r.jsx)(C.Zo,{component:Z,props:e})});function P(e){let{sidebar:t}=e,a=(0,n.i)();return t?.items.length?"mobile"===a?(0,r.jsx)(y,{sidebar:t}):(0,r.jsx)(N,{sidebar:t}):null}},93867:function(e,t,a){a.r(t),a.d(t,{default:()=>b});var r=a("85893");a("67294");var l=a("67026"),n=a("96025");let s=()=>(0,n.I)({id:"theme.tags.tagsPageTitle",message:"Tags",description:"The title of the tag list page"});var i=a("79741"),o=a("84681"),u=a("51225"),c=a("48627"),m=a("34403");let g="tag_Nnez";function d(e){let{letterEntry:t}=e;return(0,r.jsxs)("article",{children:[(0,r.jsx)(m.Z,{as:"h2",id:t.letter,children:t.letter}),(0,r.jsx)("ul",{className:"padding--none",children:t.tags.map(e=>(0,r.jsx)("li",{className:g,children:(0,r.jsx)(c.Z,{...e})},e.permalink))}),(0,r.jsx)("hr",{})]})}function h(e){let{tags:t}=e,a=function(e){let t={};return Object.values(e).forEach(e=>{let a=e.label[0].toUpperCase();t[a]??=[],t[a].push(e)}),Object.entries(t).sort((e,t)=>{let[a]=e,[r]=t;return a.localeCompare(r)}).map(e=>{let[t,a]=e;return{letter:t,tags:a.sort((e,t)=>e.label.localeCompare(t.label))}})}(t);return(0,r.jsx)("section",{className:"margin-vert--lg",children:a.map(e=>(0,r.jsx)(d,{letterEntry:e},e.letter))})}var p=a("84315");function b(e){let{tags:t,sidebar:a}=e,n=s();return(0,r.jsxs)(i.FG,{className:(0,l.Z)(o.k.wrapper.blogPages,o.k.page.blogTagsListPage),children:[(0,r.jsx)(i.d,{title:n}),(0,r.jsx)(p.Z,{tag:"blog_tags_list"}),(0,r.jsxs)(u.Z,{sidebar:a,children:[(0,r.jsx)(m.Z,{as:"h1",children:n}),(0,r.jsx)(h,{tags:t})]})]})}},48627:function(e,t,a){a.d(t,{Z:()=>i});var r=a("85893");a("67294");var l=a("67026"),n=a("83012");let s={tag:"tag_zVej",tagRegular:"tagRegular_sFm0",tagWithCount:"tagWithCount_h2kH"};function i(e){let{permalink:t,label:a,count:i,description:o}=e;return(0,r.jsxs)(n.Z,{href:t,title:o,className:(0,l.Z)(s.tag,i?s.tagWithCount:s.tagRegular),children:[a,i&&(0,r.jsx)("span",{children:i})]})}}}]);