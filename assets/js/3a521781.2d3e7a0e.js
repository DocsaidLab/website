"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["19296"],{65402:function(s,e,a){a.r(e),a.d(e,{metadata:()=>n,contentTitle:()=>c,default:()=>d,assets:()=>r,toc:()=>m,frontMatter:()=>t});var n=JSON.parse('{"id":"object-detection/yolov2/index","title":"[16.12] YOLO-V2","description":"\u64F4\u5C55\u5927\u91CF\u985E\u5225","source":"@site/papers/object-detection/1612-yolov2/index.md","sourceDirName":"object-detection/1612-yolov2","slug":"/object-detection/yolov2/","permalink":"/papers/object-detection/yolov2/","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1733839479000,"frontMatter":{"title":"[16.12] YOLO-V2","authors":"Zephyr"},"sidebar":"papersSidebar","previous":{"title":"[15.12] SSD","permalink":"/papers/object-detection/ssd/"},"next":{"title":"[17.08] RetinaNet","permalink":"/papers/object-detection/retinanet/"}}'),l=a("85893"),i=a("50065");let t={title:"[16.12] YOLO-V2",authors:"Zephyr"},c=void 0,r={},m=[{value:"\u64F4\u5C55\u5927\u91CF\u985E\u5225",id:"\u64F4\u5C55\u5927\u91CF\u985E\u5225",level:2},{value:"\u5B9A\u7FA9\u554F\u984C",id:"\u5B9A\u7FA9\u554F\u984C",level:2},{value:"\u89E3\u6C7A\u554F\u984C",id:"\u89E3\u6C7A\u554F\u984C",level:2},{value:"\u9328\u6846\u9078\u64C7",id:"\u9328\u6846\u9078\u64C7",level:3},{value:"\u9810\u6E2C\u6846\u8A2D\u8A08",id:"\u9810\u6E2C\u6846\u8A2D\u8A08",level:3},{value:"\u591A\u5C3A\u5EA6\u8A13\u7DF4",id:"\u591A\u5C3A\u5EA6\u8A13\u7DF4",level:3},{value:"Darknet-19",id:"darknet-19",level:3},{value:"\u5206\u985E\u982D\u8A2D\u8A08",id:"\u5206\u985E\u982D\u8A2D\u8A08",level:3},{value:"\u8A0E\u8AD6",id:"\u8A0E\u8AD6",level:2},{value:"\u5728 PASCAL VOC \u4E0A\u7684\u5BE6\u9A57",id:"\u5728-pascal-voc-\u4E0A\u7684\u5BE6\u9A57",level:3},{value:"\u5F9E V1 \u5230 V2",id:"\u5F9E-v1-\u5230-v2",level:3},{value:"\u7D50\u8AD6",id:"\u7D50\u8AD6",level:2}];function h(s){let e={a:"a",annotation:"annotation",h2:"h2",h3:"h3",hr:"hr",img:"img",li:"li",math:"math",mi:"mi",mo:"mo",mrow:"mrow",msub:"msub",ol:"ol",p:"p",semantics:"semantics",span:"span",strong:"strong",ul:"ul",...(0,i.a)(),...s.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(e.h2,{id:"\u64F4\u5C55\u5927\u91CF\u985E\u5225",children:"\u64F4\u5C55\u5927\u91CF\u985E\u5225"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.a,{href:"https://arxiv.org/abs/1612.08242",children:(0,l.jsx)(e.strong,{children:"YOLO9000: Better, Faster, Stronger"})})}),"\n",(0,l.jsx)(e.hr,{}),"\n",(0,l.jsx)(e.h2,{id:"\u5B9A\u7FA9\u554F\u984C",children:"\u5B9A\u7FA9\u554F\u984C"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 problem",src:a(58492).Z+"",width:"1040",height:"568"})}),"\n",(0,l.jsx)(e.p,{children:"\u56DE\u9867 YOLO v1\uFF0C\u5728\u6700\u5F8C\u7684\u5206\u6790\u7AE0\u7BC0\u4E2D\uFF0C\u4F5C\u8005\u6307\u51FA YOLO v1 \u5B58\u5728\u5927\u91CF\u5B9A\u4F4D\u932F\u8AA4\uFF0C\u9084\u6709\u4F4E\u53EC\u56DE\u7387\u7684\u554F\u984C\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u70BA\u4E86\u89E3\u6C7A\u9019\u500B\u554F\u984C\uFF0C\u5F15\u5165\u9328\u9EDE\u7684\u6982\u5FF5\u770B\u8D77\u4F86\u52E2\u5728\u5FC5\u884C\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u9664\u6B64\u4E4B\u5916\uFF0C\u8FD1\u671F\u4E5F\u6709\u8A31\u591A\u66F4\u597D\u7684\u8A13\u7DF4\u65B9\u5F0F\u88AB\u63D0\u51FA\u4F86\uFF0C\u5728\u9019\u88E1\u4F5C\u8005\u4E5F\u628A\u9019\u4E9B\u65B9\u6CD5\u4E00\u8D77\u7D0D\u5165\u8003\u91CF\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"\u89E3\u6C7A\u554F\u984C",children:"\u89E3\u6C7A\u554F\u984C"}),"\n",(0,l.jsx)(e.h3,{id:"\u9328\u6846\u9078\u64C7",children:"\u9328\u6846\u9078\u64C7"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 anchor",src:a(97407).Z+"",width:"1224",height:"672"})}),"\n",(0,l.jsx)(e.p,{children:"\u9996\u5148\u662F\u9328\u6846\u7684\u9078\u64C7\uFF0C\u4E4B\u524D\u7684\u65B9\u6CD5\u662F\u4F7F\u7528\u4E00\u7D44\u56FA\u5B9A\u6BD4\u4F8B\u7684\u9328\u6846\uFF0C\u4F8B\u5982 1:1, 1:2, 2:1 \u7B49\u7B49\uFF0C\u4F46\u4F5C\u8005\u8A8D\u70BA\u65E2\u7136\u8981\u7528\u5728\u7279\u5B9A\u7684\u8CC7\u6599\u96C6\u4E0A\uFF0C\u90A3\u4E0D\u59A8\u76F4\u63A5\u5728\u8CC7\u6599\u96C6\u4E0A\u627E\u5230\u6700\u4F73\u7684\u9328\u6846\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u9019\u88E1\u63A1\u7528\u7684\u65B9\u6CD5\u5C31\u662F\u628A\u6240\u6709\u8A13\u7DF4\u8CC7\u6599\u7684\u5BEC\u9AD8\u6BD4\u4F8B\u90FD\u7B97\u51FA\u4F86\uFF0C\u7136\u5F8C\u7528 K-means \u805A\u985E\u7684\u65B9\u5F0F\u627E\u5230\u6700\u4F73\u7684\u9328\u6846\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5982\u4E0A\u5716\uFF0C\u4F5C\u8005\u5206\u5225\u5728 COCO \u8207 VOC \u8CC7\u6599\u96C6\u4E0A\u627E\u5230\u4E86 N \u500B\u9328\u6846\uFF0C\u4E26\u4E14\u7D93\u904E\u5BE6\u9A57\uFF0C\u767C\u73FE\u4F7F\u7528 5 \u500B\u9328\u6846\u53EF\u4EE5\u9054\u5230\u6700\u4F73\u7684\u6B0A\u8861\u3002"}),"\n",(0,l.jsx)(e.h3,{id:"\u9810\u6E2C\u6846\u8A2D\u8A08",children:"\u9810\u6E2C\u6846\u8A2D\u8A08"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 detection",src:a(84571).Z+"",width:"1224",height:"900"})}),"\n",(0,l.jsx)(e.p,{children:"\u5728 Faster R-CNN \u7684\u8AD6\u6587\u4E2D\uFF0C\u4F7F\u7528 RPN \u9032\u884C\u5340\u57DF\u63D0\u8B70\uFF0C\u6A21\u578B\u6700\u5F8C\u6703\u9810\u6E2C\u4E00\u500B\u504F\u79FB\u91CF\uFF0C\u9019\u500B\u504F\u79FB\u91CF\u6703\u88AB\u7528\u4F86\u4FEE\u6B63\u9810\u6E2C\u6846\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u4E0A\u4E00\u7248\u7684 YOLO \u4E2D\uFF0C\u4F5C\u8005\u4F7F\u7528\u4E86 7x7 \u7684\u7DB2\u683C\uFF0C\u6BCF\u500B\u7DB2\u683C\u9810\u6E2C 2 \u500B\u6846\uFF0C\u9019\u5176\u5BE6\u662F\u4E00\u500B\u4E0D\u53D7\u9650\u5236\u7684\u9810\u6E2C\u7A7A\u9593\uFF0C\u9019\u6A23\u7684\u8A2D\u8A08\u76F8\u6BD4\u65BC Faster R-CNN \u4F86\u8AAA\uFF0C\u6A21\u578B\u8CA0\u64D4\u66F4\u5927\u3002\u56E0\u6B64\u9019\u88E1\u4F5C\u8005\u4E5F\u5F15\u5165\u4E86\u504F\u79FB\u91CF\u7684\u6982\u5FF5\uFF0C\u900F\u904E\u66F4\u591A\u7684\u5148\u9A57\u77E5\u8B58\uFF0C\u964D\u4F4E\u6A21\u578B\u7684\u8CA0\u64D4\u3002"}),"\n",(0,l.jsxs)(e.p,{children:["\u4FEE\u6539\u5F8C\u7684\u8A2D\u8A08\u5982\u4E0A\u5716\uFF0C\u6A21\u578B\u9810\u6E2C\u7684\u6578\u503C\u5F9E\u539F\u672C\u7684 ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsxs)(e.mrow,{children:[(0,l.jsx)(e.mi,{children:"x"}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsx)(e.mi,{children:"y"}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsx)(e.mi,{children:"w"}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsx)(e.mi,{children:"h"})]}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"x, y, w, h"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.8889em",verticalAlign:"-0.1944em"}}),(0,l.jsx)(e.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(e.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(e.span,{className:"mord mathnormal",style:{marginRight:"0.02691em"},children:"w"}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(e.span,{className:"mord mathnormal",children:"h"})]})})]})," \u6539\u70BA ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsxs)(e.mrow,{children:[(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"x"})]}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"y"})]}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"w"})]}),(0,l.jsx)(e.mo,{separator:"true",children:","}),(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"h"})]})]}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"t_x, t_y, t_w, t_h"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.9012em",verticalAlign:"-0.2861em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",children:"x"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.03588em"},children:"y"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.2861em"},children:(0,l.jsx)(e.span,{})})})]})})]}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.02691em"},children:"w"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]}),(0,l.jsx)(e.span,{className:"mpunct",children:","}),(0,l.jsx)(e.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.3361em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",children:"h"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]}),"\uFF0C\u6BCF\u500B\u90E8\u5206\u7684\u8655\u7406\u65B9\u5F0F\u5982\u4E0B\uFF1A"]}),"\n",(0,l.jsxs)(e.ol,{children:["\n",(0,l.jsxs)(e.li,{children:["\n",(0,l.jsxs)(e.p,{children:[(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"x"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"t_x"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.7651em",verticalAlign:"-0.15em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",children:"x"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u8207 ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"y"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"t_y"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.9012em",verticalAlign:"-0.2861em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.03588em"},children:"y"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.2861em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u4EE3\u8868\u7DB2\u683C\u4E2D\u5FC3\u7684\u504F\u79FB\u91CF\uFF0C\u9810\u6E2C\u7D50\u679C\u7D93\u904E\u4E00\u5C64 sigmoid \u51FD\u6578\uFF0C\u4F7F\u5F97\u9810\u6E2C\u7D50\u679C\u4ECB\u65BC 0 \u8207 1 \u4E4B\u9593\uFF0C\u4E5F\u5C31\u662F\u8AAA\u9810\u6E2C\u4E2D\u5FC3\u4E0D\u6703\u8D85\u51FA\u7DB2\u683C\u7684\u7BC4\u570D\u3002\u4E0A\u5716\u4E2D\u7684 ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"c"}),(0,l.jsx)(e.mi,{children:"x"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"c_x"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.5806em",verticalAlign:"-0.15em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"c"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",children:"x"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u8207 ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"c"}),(0,l.jsx)(e.mi,{children:"y"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"c_y"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.7167em",verticalAlign:"-0.2861em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"c"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.03588em"},children:"y"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.2861em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u4EE3\u8868\u7DB2\u683C\u7684\u5DE6\u4E0A\u89D2\u5EA7\u6A19\u3002"]}),"\n"]}),"\n",(0,l.jsxs)(e.li,{children:["\n",(0,l.jsxs)(e.p,{children:[(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"w"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"t_w"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.7651em",verticalAlign:"-0.15em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",style:{marginRight:"0.02691em"},children:"w"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u8207 ",(0,l.jsxs)(e.span,{className:"katex",children:[(0,l.jsx)(e.span,{className:"katex-mathml",children:(0,l.jsx)(e.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(e.semantics,{children:[(0,l.jsx)(e.mrow,{children:(0,l.jsxs)(e.msub,{children:[(0,l.jsx)(e.mi,{children:"t"}),(0,l.jsx)(e.mi,{children:"h"})]})}),(0,l.jsx)(e.annotation,{encoding:"application/x-tex",children:"t_h"})]})})}),(0,l.jsx)(e.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(e.span,{className:"base",children:[(0,l.jsx)(e.span,{className:"strut",style:{height:"0.7651em",verticalAlign:"-0.15em"}}),(0,l.jsxs)(e.span,{className:"mord",children:[(0,l.jsx)(e.span,{className:"mord mathnormal",children:"t"}),(0,l.jsx)(e.span,{className:"msupsub",children:(0,l.jsxs)(e.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(e.span,{className:"vlist-r",children:[(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.3361em"},children:(0,l.jsxs)(e.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(e.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(e.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(e.span,{className:"mord mathnormal mtight",children:"h"})})]})}),(0,l.jsx)(e.span,{className:"vlist-s",children:"\u200B"})]}),(0,l.jsx)(e.span,{className:"vlist-r",children:(0,l.jsx)(e.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(e.span,{})})})]})})]})]})})]})," \u4EE3\u8868\u9810\u6E2C\u6846\u7684\u5BEC\u9AD8\uFF0C\u9810\u6E2C\u7D50\u679C\u6703\u5148\u7D93\u904E exp \u51FD\u6578\uFF0C\u4F7F\u5F97\u9810\u6E2C\u7D50\u679C\u70BA\u6B63\u6578\uFF0C\u7136\u5F8C\u518D\u4E58\u4E0A\u9328\u6846\u7684\u5BEC\u9AD8\u3002"]}),"\n"]}),"\n"]}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u9019\u4E00\u6B65\u9A5F\u4E2D\uFF0C\u4F5C\u8005\u7279\u610F\u5C07\u7DB2\u8DEF\u7684\u8F38\u5165\u5F9E\u539F\u672C\u7684 448x448 \u6539\u70BA 416x416\uFF0C\u9019\u6A23\u53EF\u4EE5\u8B93\u7DB2\u683C\u7684\u5927\u5C0F\u8B8A\u70BA 13x13\uFF0C\u9019\u6A23\u7684\u8A2D\u8A08\u53EF\u4EE5\u8B93\u7DB2\u683C\u7684\u4E2D\u5FC3\u9EDE\u843D\u5728\u7DB2\u683C\u7684\u4EA4\u53C9\u9EDE\u4E0A\uFF0C\u53EF\u4EE5\u8B93\u7DB2\u683C\u7684\u4E2D\u5FC3\u9EDE\u66F4\u6E96\u78BA\u3002\u7D93\u904E\u9019\u6A23\u7684\u8A2D\u8A08\uFF0C\u4F5C\u8005\u767C\u73FE\u6A21\u578B\u7684\u6E96\u78BA\u7387\u7565\u70BA\u4E0B\u964D\uFF0C\u4F46\u662F\u53EC\u56DE\u7387\u5927\u5E45\u63D0\u5347\uFF0C\u5F9E\u539F\u672C\u7684 81% \u63D0\u5347\u5230 88%\u3002"}),"\n",(0,l.jsx)(e.h3,{id:"\u591A\u5C3A\u5EA6\u8A13\u7DF4",children:"\u591A\u5C3A\u5EA6\u8A13\u7DF4"}),"\n",(0,l.jsx)(e.p,{children:"\u9664\u4E86\u8ABF\u6574\u9810\u6E2C\u6846\u7684\u8A2D\u8A08\uFF0C\u4F5C\u8005\u4E5F\u5F15\u5165\u4E86\u591A\u5C3A\u5EA6\u8A13\u7DF4\u7684\u6982\u5FF5\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5728\u6A21\u578B\u7684\u8A13\u7DF4\u904E\u7A0B\u4E2D\uFF0C\u4F5C\u8005\u6703\u6BCF\u9694 10 \u500B batch \u5C31\u96A8\u6A5F\u9078\u64C7\u4E00\u500B\u5C3A\u5EA6\uFF0C\u4EE5 32 \u7684\u500D\u6578\u63D0\u53D6\uFF0C\u4F8B\u5982 320x320, 352x352, 384x384 \u7B49\uFF0C\u6700\u5C0F\u7684\u5C3A\u5EA6\u70BA 320x320\uFF0C\u6700\u5927\u7684\u5C3A\u5EA6\u70BA 608x608\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u9019\u7A2E\u65B9\u5F0F\u53EF\u4EE5\u8B93\u6A21\u578B\u5728\u4E0D\u540C\u5C3A\u5EA6\u4E0B\u90FD\u6709\u5F88\u597D\u7684\u8868\u73FE\uFF0C\u4E26\u4E14\u53EF\u4EE5\u63D0\u5347\u6A21\u578B\u7684\u6CDB\u5316\u80FD\u529B\u3002"}),"\n",(0,l.jsx)(e.h3,{id:"darknet-19",children:"Darknet-19"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 darknet",src:a(88438).Z+"",width:"872",height:"1080"})}),"\n",(0,l.jsx)(e.p,{children:"\u70BA\u4E86\u5F97\u5230\u66F4\u5FEB\u7684\u63A8\u8AD6\u901F\u5EA6\uFF0C\u4F5C\u8005\u4E0D\u4F7F\u7528\u5176\u4ED6\u73FE\u6210\u7684\u9AA8\u5E79\u7DB2\u8DEF\uFF0C\u800C\u662F\u81EA\u5DF1\u8A2D\u8A08\u4E86\u4E00\u500B\u7DB2\u8DEF\uFF0C\u7A31\u70BA Darknet-19\uFF0C\u6A21\u578B\u67B6\u69CB\u8A2D\u8A08\u5982\u4E0A\u8868\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u548C VGG \u985E\u4F3C\uFF0C\u4F5C\u8005\u4F7F\u7528\u4E86 3x3 \u7684\u5377\u7A4D\u6838\uFF0C\u5728\u6BCF\u500B\u6C60\u5316\u5C64\u4E4B\u5F8C\u52A0\u500D\u901A\u9053\u6578\uFF0C\u4F7F\u7528\u4E86\u6279\u6A19\u6E96\u5316\uFF0C\u4E26\u4E14\u5728\u6700\u5F8C\u52A0\u5165\u4E86\u5168\u9023\u63A5\u5C64\u3002\u9019\u500B\u6A21\u578B\u5148\u5728 ImageNet \u4E0A\u9032\u884C\u4E86\u8A13\u7DF4\uFF0Ctop1 \u6E96\u78BA\u5EA6\u9054\u5230 76.5%\uFF0Ctop5 \u6E96\u78BA\u5EA6\u9054\u5230 93.3%\u3002\u8A13\u7DF4\u904E\u7A0B\u4E2D\u4F7F\u7528\u6A19\u6E96\u7684\u5716\u50CF\u589E\u5F37\u6280\u8853\uFF0C\u4F8B\u5982\u96A8\u6A5F\u7FFB\u8F49\u3001\u96A8\u6A5F\u88C1\u526A\u7B49\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u6700\u5F8C\u628A\u9019\u500B\u6A21\u578B\u8F49\u79FB\u5230\u76EE\u6A19\u6AA2\u6E2C\u7684\u4EFB\u52D9\u4E0A\uFF0C\u62BD\u63DB\u539F\u672C\u7684\u5206\u985E\u982D\uFF0C\u6539\u70BA\u76EE\u6A19\u6AA2\u6E2C\u982D\u5F8C\u9032\u884C\u5FAE\u8ABF\uFF0C\u9019\u6A23\u5C31\u5B8C\u6210\u4E86\u6574\u500B\u6A21\u578B\u7684\u8A2D\u8A08\u3002"}),"\n",(0,l.jsx)(e.h3,{id:"\u5206\u985E\u982D\u8A2D\u8A08",children:"\u5206\u985E\u982D\u8A2D\u8A08"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 classification",src:a(68261).Z+"",width:"1072",height:"1080"})}),"\n",(0,l.jsx)(e.p,{children:"\u70BA\u4E86\u62D3\u5C55\u7269\u4EF6\u5075\u6E2C\u7684\u985E\u5225\u6578\u91CF\uFF0C\u4F5C\u8005\u5F15\u5165\u4E86 WordTree \u7684\u6982\u5FF5\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u4F5C\u8005\u901A\u904E\u5C07\u5206\u985E\u548C\u6AA2\u6E2C\u6578\u64DA\u96C6\u806F\u5408\u8A13\u7DF4\u4F86\u63D0\u5347\u7269\u9AD4\u6AA2\u6E2C\u548C\u5206\u985E\u7684\u80FD\u529B\u3002\u8A72\u65B9\u6CD5\u5229\u7528\u5E36\u6709\u6AA2\u6E2C\u6A19\u7C64\u7684\u5716\u50CF\u5B78\u7FD2\u6AA2\u6E2C\u7279\u5B9A\u4FE1\u606F\uFF0C\u5982\u908A\u754C\u6846\u5750\u6A19\u548C\u7269\u9AD4\u5B58\u5728\u6027\uFF0C\u4E26\u5229\u7528\u53EA\u6709\u985E\u5225\u6A19\u7C64\u7684\u5716\u50CF\u64F4\u5C55\u53EF\u6AA2\u6E2C\u7684\u7269\u9AD4\u985E\u5225\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u9019\u7A2E\u65B9\u6CD5\u9762\u81E8\u7684\u6311\u6230\u4E4B\u4E00\u662F\u5982\u4F55\u5408\u4F75\u5169\u500B\u6578\u64DA\u96C6\u7684\u6A19\u7C64\uFF0C\u56E0\u70BA\u6AA2\u6E2C\u6578\u64DA\u96C6\u901A\u5E38\u53EA\u6709\u4E00\u822C\u6A19\u7C64\uFF08\u5982\u300C\u72D7\u300D\u6216\u300C\u8239\u300D\uFF09\uFF0C\u800C\u5206\u985E\u6578\u64DA\u96C6\u5247\u6709\u66F4\u8A73\u7D30\u7684\u6A19\u7C64\uFF08\u5982\u5404\u7A2E\u72D7\u7684\u54C1\u7A2E\uFF09\u3002\u50B3\u7D71\u7684\u5206\u985E\u65B9\u6CD5\u4F7F\u7528 softmax \u5C64\u4F86\u8A08\u7B97\u6240\u6709\u53EF\u80FD\u985E\u5225\u7684\u6700\u7D42\u6982\u7387\u5206\u4F48\uFF0C\u5047\u8A2D\u985E\u5225\u4E4B\u9593\u662F\u4E92\u65A5\u7684\uFF0C\u9019\u5728\u5408\u4F75\u6578\u64DA\u96C6\u6642\u6703\u5F15\u8D77\u554F\u984C\u3002\u56E0\u6B64\uFF0C\u4F5C\u8005\u63D0\u51FA\u4F7F\u7528 WordNet \u69CB\u5EFA\u4E00\u500B\u5C64\u6B21\u6A39\u6A21\u578B WordTree\uFF0C\u4EE5\u89E3\u6C7A\u9019\u4E00\u554F\u984C\u3002"}),"\n",(0,l.jsx)(e.p,{children:"WordTree \u6A21\u578B\u5229\u7528 WordNet \u7684\u5C64\u6B21\u7D50\u69CB\uFF0C\u5C07\u6982\u5FF5\u6309\u5C64\u6B21\u7D44\u7E54\u8D77\u4F86\uFF0C\u5F9E\u800C\u80FD\u5920\u9032\u884C\u66F4\u7D30\u7DFB\u7684\u5206\u985E\u3002\u9019\u7A2E\u6A21\u578B\u5728\u6BCF\u500B\u7BC0\u9EDE\u9810\u6E2C\u689D\u4EF6\u6982\u7387\uFF0C\u4E26\u901A\u904E\u4E58\u7A4D\u8A08\u7B97\u7279\u5B9A\u7BC0\u9EDE\u7684\u7D55\u5C0D\u6982\u7387\u3002\u4F8B\u5982\uFF1A\u8981\u77E5\u9053\u4E00\u5F35\u5716\u7247\u662F\u5426\u662F\u8AFE\u798F\u514B\u6897\uFF0C\u53EA\u9700\u6CBF\u8457\u5F9E\u6839\u7BC0\u9EDE\u5230\u8A72\u7BC0\u9EDE\u7684\u8DEF\u5F91\uFF0C\u4E58\u4E0A\u6BCF\u500B\u7BC0\u9EDE\u7684\u689D\u4EF6\u6982\u7387\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u4F5C\u8005\u5C07\u9019\u7A2E\u5C64\u6B21\u7D50\u69CB\u61C9\u7528\u65BC\u5206\u985E\u548C\u6AA2\u6E2C\u4EFB\u52D9\uFF0C\u8A13\u7DF4\u4E86\u4E00\u500B\u540D\u70BA YOLO9000 \u7684\u6A21\u578B\u3002\u9019\u500B\u6A21\u578B\u4F7F\u7528 COCO \u6AA2\u6E2C\u6578\u64DA\u96C6\u548C ImageNet \u5206\u985E\u6578\u64DA\u96C6\u9032\u884C\u806F\u5408\u8A13\u7DF4\uFF0C\u80FD\u5920\u5728\u5BE6\u6642\u6AA2\u6E2C 9000 \u591A\u500B\u7269\u9AD4\u985E\u5225\u7684\u540C\u6642\uFF0C\u4FDD\u6301\u8F03\u9AD8\u7684\u6E96\u78BA\u7387\u3002\u5BE6\u9A57\u7D50\u679C\u8868\u660E\uFF0CYOLO9000 \u5728 ImageNet \u6AA2\u6E2C\u4EFB\u52D9\u4E2D\uFF0C\u5373\u4F7F\u5C0D\u65BC\u5F9E\u672A\u898B\u904E\u6AA2\u6E2C\u6578\u64DA\u7684\u985E\u5225\uFF0C\u4E5F\u80FD\u53D6\u5F97\u826F\u597D\u7684\u6027\u80FD\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"\u8A0E\u8AD6",children:"\u8A0E\u8AD6"}),"\n",(0,l.jsx)(e.h3,{id:"\u5728-pascal-voc-\u4E0A\u7684\u5BE6\u9A57",children:"\u5728 PASCAL VOC \u4E0A\u7684\u5BE6\u9A57"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 pascal",src:a(29479).Z+"",width:"1224",height:"700"})}),"\n",(0,l.jsx)(e.p,{children:"\u5982\u4E0A\u8868\uFF0C\u4F5C\u8005\u5728 PASCAL VOC 2007 \u4E0A\u9032\u884C\u4E86\u5BE6\u9A57\uFF0C\u7D50\u679C\u986F\u793A YOLOv2 \u5728\u901F\u5EA6\u548C\u6E96\u78BA\u6027\u4E4B\u9593\u53D6\u5F97\u4E86\u826F\u597D\u7684\u5E73\u8861\u3002"}),"\n",(0,l.jsxs)(e.ul,{children:["\n",(0,l.jsx)(e.li,{children:"\u5728 288 \xd7 288 \u7684\u89E3\u6790\u5EA6\u4E0B\uFF0CYOLOv2 \u7684\u901F\u5EA6\u8D85\u904E 90 FPS\uFF0CmAP \u5E7E\u4E4E\u8207 Fast R-CNN \u4E00\u6A23\u597D\u3002\u9019\u4F7F\u5F97\u5B83\u975E\u5E38\u9069\u5408\u8F03\u5C0F\u7684 GPU\u3001\u9AD8\u5E40\u7387\u8996\u8A0A\u6216\u591A\u8996\u8A0A\u4E32\u6D41\u3002"}),"\n",(0,l.jsx)(e.li,{children:"\u5728 416 \xd7 416 \u7684\u89E3\u6790\u5EA6\u4E0B\uFF0CYOLOv2 \u7684 mAP \u9054\u5230 76.8%\uFF0C\u901F\u5EA6\u70BA 67 FPS\uFF0C\u9019\u7121\u7591\u662F\u7576\u6642\u6700\u5148\u9032\u7684\u5075\u6E2C\u5668\u4E4B\u4E00\u3002"}),"\n"]}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 performance",src:a(24636).Z+"",width:"1224",height:"900"})}),"\n",(0,l.jsx)(e.h3,{id:"\u5F9E-v1-\u5230-v2",children:"\u5F9E V1 \u5230 V2"}),"\n",(0,l.jsx)(e.p,{children:(0,l.jsx)(e.img,{alt:"yolo v2 ablation",src:a(17569).Z+"",width:"1224",height:"516"})}),"\n",(0,l.jsx)(e.p,{children:"\u4E0A\u8868\u5C55\u793A\u4E86\u4F5C\u8005\u5982\u4F55\u900F\u904E\u4E0D\u540C\u7684\u8A2D\u8A08\u4F86\u6539\u9032 YOLO \u7684\u6027\u80FD\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5F9E\u589E\u52A0 BatchNorm \u958B\u59CB\uFF0C\u4F5C\u8005\u9010\u6B65\u5F15\u5165\u4E86\u66F4\u591A\u7684\u8A2D\u8A08\uFF0C\u5305\u62EC\u66F4\u6DF1\u7684\u7DB2\u8DEF\u3001\u66F4\u591A\u7684\u9328\u6846\u3001\u66F4\u591A\u7684\u8A13\u7DF4\u6578\u64DA\u7B49\u3002\u9019\u4E9B\u6539\u9032\u4F7F\u5F97 YOLOv2 \u5728\u901F\u5EA6\u548C\u6E96\u78BA\u6027\u4E4B\u9593\u53D6\u5F97\u4E86\u66F4\u597D\u7684\u5E73\u8861\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u9019\u88E1\u53EF\u4EE5\u6CE8\u610F\u5230 anchor boxes \u7684\u5F15\u5165\u5728\u4F5C\u8005\u7684\u5BE6\u9A57\u4E2D\u6703\u964D\u4F4E mAP\uFF0C\u56E0\u6B64\u5F8C\u4F86\u9019\u88E1\u6539\u6210\u4F7F\u7528\u805A\u985E\u7684\u65B9\u5F0F\u627E\u5230\u6700\u4F73\u7684 anchor boxes\u3002"}),"\n",(0,l.jsx)(e.h2,{id:"\u7D50\u8AD6",children:"\u7D50\u8AD6"}),"\n",(0,l.jsx)(e.p,{children:"YOLOv2 \u5F15\u5165\u9328\u9EDE\u6280\u8853\uFF0C\u589E\u5F37\u4E86\u5C0D\u5C0F\u578B\u7269\u9AD4\u7684\u5075\u6E2C\u80FD\u529B\uFF0C\u4E26\u4E14\u4F7F\u7528 batchnorm \u4F86\u964D\u4F4E\u5C0D\u7279\u5B9A\u6578\u64DA\u96C6\u7684\u904E\u5EA6\u64EC\u5408\uFF0C\u5F9E\u800C\u63D0\u9AD8\u4E86\u6A21\u578B\u7684\u6CDB\u5316\u80FD\u529B\u3002\u540C\u6642\uFF0C\u6A21\u578B\u80FD\u5920\u9032\u884C\u7AEF\u5230\u7AEF\u7684\u8A13\u7DF4\u548C\u9810\u6E2C\uFF0C\u9019\u4F7F\u5F97\u6574\u500B\u8A13\u7DF4\u904E\u7A0B\u66F4\u70BA\u7C21\u5316\u548C\u9AD8\u6548\u3002\u53E6\u4E00\u65B9\u9762\uFF0C\u9019\u500B\u6A21\u578B\u4E5F\u5B58\u5728\u4E00\u4E9B\u7F3A\u9EDE\uFF0C\u4F8B\u5982\u5728\u8655\u7406\u5F62\u72C0\u4E0D\u898F\u5247\u7684\u7269\u9AD4\u7684\u5B9A\u4F4D\u4E0A\u53EF\u80FD\u4E0D\u5982\u5176\u4ED6\u65B9\u6CD5\u5982 Faster R-CNN \u90A3\u9EBC\u7CBE\u78BA\u3002"}),"\n",(0,l.jsx)(e.p,{children:"\u5118\u7BA1\u6E96\u78BA\u5EA6\u7565\u4F4E\uFF0C\u4F46\u6C92\u95DC\u4FC2\uFF0C\u81F3\u5C11\u5B83\u5F88\u5FEB\uFF01"}),"\n",(0,l.jsx)(e.p,{children:"\u50C5\u6191\u9019\u9EDE\u5C31\u8DB3\u4EE5\u8B93\u5B83\u6210\u70BA\u7576\u6642\u6700\u53D7\u6B61\u8FCE\u7684\u7269\u9AD4\u6AA2\u6E2C\u6A21\u578B\u3002"})]})}function d(s={}){let{wrapper:e}={...(0,i.a)(),...s.components};return e?(0,l.jsx)(e,{...s,children:(0,l.jsx)(h,{...s})}):h(s)}},97407:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img1-16d09caaa7cdb3c237558de3560e1370.jpg"},84571:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img2-dfeae8caf28af223c9141a6672747104.jpg"},24636:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img3-2eaa8b0acef981caf6f2044ac4d28fa0.jpg"},29479:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img4-5d2826dc6cb1bd6fe3fbe8cc57039307.jpg"},17569:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img5-3fc51e414a664ee2827c15e4cc3a12a4.jpg"},88438:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img6-b2d07692aa063d1b389bd7cb2a3de3cd.jpg"},68261:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img7-1f654af2e4e978071a5c3a537543d1ae.jpg"},58492:function(s,e,a){a.d(e,{Z:function(){return n}});let n=a.p+"assets/images/img9-0f4986c9ef5c9085ca41d1d2465b4f50.jpg"},50065:function(s,e,a){a.d(e,{Z:function(){return c},a:function(){return t}});var n=a(67294);let l={},i=n.createContext(l);function t(s){let e=n.useContext(i);return n.useMemo(function(){return"function"==typeof s?s(e):{...e,...s}},[e,s])}function c(s){let e;return e=s.disableParentContext?"function"==typeof s.components?s.components(l):s.components||l:t(s.components),n.createElement(i.Provider,{value:e},s.children)}}}]);