"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["16205"],{23089:function(e,n,s){s.r(n),s.d(n,{metadata:()=>t,contentTitle:()=>c,default:()=>a,assets:()=>l,toc:()=>h,frontMatter:()=>r});var t=JSON.parse('{"id":"docsaidkit/funcs/time/time2time","title":"Time2Time","description":"\u9019\u662F\u4E00\u500B\u7528\u4F86\u8F49\u63DB\u6642\u9593\u683C\u5F0F\u7684\u5DE5\u5177\u3002","source":"@site/docs/docsaidkit/funcs/time/time2time.md","sourceDirName":"docsaidkit/funcs/time","slug":"/docsaidkit/funcs/time/time2time","permalink":"/docs/docsaidkit/funcs/time/time2time","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1712674901000,"sidebarPosition":3,"frontMatter":{"sidebar_position":3},"sidebar":"tutorialSidebar","previous":{"title":"Timer","permalink":"/docs/docsaidkit/funcs/time/timer"},"next":{"title":"Vision","permalink":"/docs/category/vision-1"}}'),i=s("85893"),d=s("50065");let r={sidebar_position:3},c="Time2Time",l={},h=[{value:"timestamp2datetime",id:"timestamp2datetime",level:2},{value:"timestamp2time",id:"timestamp2time",level:2},{value:"timestamp2str",id:"timestamp2str",level:2},{value:"time2datetime",id:"time2datetime",level:2},{value:"time2timestamp",id:"time2timestamp",level:2},{value:"time2str",id:"time2str",level:2},{value:"datetime2time",id:"datetime2time",level:2},{value:"datetime2timestamp",id:"datetime2timestamp",level:2},{value:"datetime2str",id:"datetime2str",level:2},{value:"str2time",id:"str2time",level:2},{value:"str2datetime",id:"str2datetime",level:2},{value:"str2timestamp",id:"str2timestamp",level:2}];function m(e){let n={a:"a",blockquote:"blockquote",code:"code",h1:"h1",h2:"h2",header:"header",hr:"hr",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,d.a)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.header,{children:(0,i.jsx)(n.h1,{id:"time2time",children:"Time2Time"})}),"\n",(0,i.jsx)(n.p,{children:"\u9019\u662F\u4E00\u500B\u7528\u4F86\u8F49\u63DB\u6642\u9593\u683C\u5F0F\u7684\u5DE5\u5177\u3002"}),"\n",(0,i.jsx)(n.p,{children:"\u5728 Python \u4E2D\uFF0C\u591A\u500B\u4E0D\u540C\u6642\u9593\u5957\u4EF6\u7684\u8F49\u63DB\u4E00\u76F4\u90FD\u662F\u500B\u60F1\u4EBA\u7684\u554F\u984C\u3002"}),"\n",(0,i.jsx)(n.p,{children:"\u70BA\u4E86\u89E3\u6C7A\u9019\u500B\u554F\u984C\uFF0C\u6211\u5011\u958B\u767C\u4E86\u5E7E\u500B\u8F49\u63DB\u51FD\u6578\uFF0C\u4F7F\u5F97\u5728 datetime\u3001struct_time\u3001timestamp \u548C \u6642\u9593\u5B57\u4E32 \u4E4B\u9593\u7684\u8F49\u63DB\u8B8A\u5F97\u81EA\u7531\u3002"}),"\n",(0,i.jsx)(n.p,{children:"\u4EE5\u4E0B\u662F\u9019\u4E9B\u51FD\u6578\u4E4B\u9593\u95DC\u4FC2\u5716\uFF1A"}),"\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.img,{alt:"time2time",src:s(47105).Z+"",width:"1326",height:"683"})}),"\n",(0,i.jsx)(n.p,{children:"\u5982\u679C\u4F60\u597D\u5947\u4E0A\u9762\u90A3\u5F35\u5716\u662F\u600E\u9EBC\u756B\u51FA\u4F86\u7684\uFF0C\u53EF\u4EE5\u53C3\u8003\u4E0B\u9762\u7684 Mermaid \u7A0B\u5F0F\u78BC\uFF1A"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-mermaid",children:"graph TD\n    timestamp(timestamp)\n    struct_time(struct_time)\n    datetime(datetime)\n    str(\u6642\u9593\u5B57\u4E32)\n\n    timestamp --\x3e|timestamp2datetime| datetime\n    timestamp --\x3e|timestamp2time| struct_time\n    timestamp --\x3e|timestamp2str| str\n\n    struct_time --\x3e|time2datetime| datetime\n    struct_time --\x3e|time2timestamp| timestamp\n    struct_time --\x3e|time2str| str\n\n    datetime --\x3e|datetime2time| struct_time\n    datetime --\x3e|datetime2timestamp| timestamp\n    datetime --\x3e|datetime2str| str\n\n    str --\x3e|str2time| struct_time\n    str --\x3e|str2datetime| datetime\n    str --\x3e|str2timestamp| timestamp\n"})}),"\n",(0,i.jsx)(n.p,{children:"\u770B\u5716\u8AAA\u6545\u4E8B\uFF0C\u5148\u627E\u5230\u4F60\u9700\u8981\u7684\u8F49\u63DB\u51FD\u6578\u4E4B\u5F8C\uFF0C\u518D\u5F80\u4E0B\u627E\uFF1A"}),"\n",(0,i.jsx)(n.hr,{}),"\n",(0,i.jsx)(n.h2,{id:"timestamp2datetime",children:"timestamp2datetime"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L188",children:"timestamp2datetime(ts: Union[int, float]) -> datetime"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u6233\u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"ts"})," (",(0,i.jsx)(n.code,{children:"Union[int, float]"}),")\uFF1A\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"datetime"}),"\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\ndt = D.timestamp2datetime(ts)\nprint(dt)\n# >>> 2021-10-12 16:00:00\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"timestamp2time",children:"timestamp2time"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L192",children:"timestamp2time(ts: Union[int, float]) -> struct_time"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u6233\u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"ts"})," (",(0,i.jsx)(n.code,{children:"Union[int, float]"}),")\uFF1A\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"struct_time"}),"\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\nt = D.timestamp2time(ts)\nprint(t)\n# >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=0)\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"timestamp2str",children:"timestamp2str"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L196",children:"timestamp2str(ts: Union[int, float], fmt: str) -> str"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u6233\u8F49\u63DB\u70BA\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"ts"})," (",(0,i.jsx)(n.code,{children:"Union[int, float]"}),")\uFF1A\u6642\u9593\u6233\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"str"}),"\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\ns = D.timestamp2str(ts, fmt='%Y-%m-%d %H:%M:%S')\nprint(s)\n# >>> '2021-10-12 16:00:00'\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"time2datetime",children:"time2datetime"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L200",children:"time2datetime(t: struct_time) -> datetime"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"struct_time"})," \u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"t"})," (",(0,i.jsx)(n.code,{children:"struct_time"}),")\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"datetime"}),"\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\nt = D.timestamp2time(ts)\ndt = D.time2datetime(t)\nprint(dt)\n# >>> datetime.datetime(2021, 10, 12, 16, 0)\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"time2timestamp",children:"time2timestamp"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L206",children:"time2timestamp(t: struct_time) -> float"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"struct_time"})," \u8F49\u63DB\u70BA\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"t"})," (",(0,i.jsx)(n.code,{children:"struct_time"}),")\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"float"}),"\uFF1A\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\nt = D.timestamp2time(ts)\nts = D.time2timestamp(t)\nprint(ts)\n# >>> 1634025600.0\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"time2str",children:"time2str"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L212",children:"time2str(t: struct_time, fmt: str) -> str"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"struct_time"})," \u8F49\u63DB\u70BA\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"t"})," (",(0,i.jsx)(n.code,{children:"struct_time"}),")\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"str"}),"\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\nt = D.timestamp2time(ts)\ns = D.time2str(t, fmt='%Y-%m-%d %H:%M:%S')\nprint(s)\n# >>> '2021-10-12 16:00:00'\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"datetime2time",children:"datetime2time"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L218",children:"datetime2time(dt: datetime) -> struct_time"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"datetime"})," \u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"dt"})," (",(0,i.jsx)(n.code,{children:"datetime"}),")\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"struct_time"}),"\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\ndt = D.timestamp2datetime(ts)\nt = D.datetime2time(dt)\nprint(t)\n# >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=-1)\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"datetime2timestamp",children:"datetime2timestamp"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L224",children:"datetime2timestamp(dt: datetime) -> float"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"datetime"})," \u8F49\u63DB\u70BA\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"dt"})," (",(0,i.jsx)(n.code,{children:"datetime"}),")\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"float"}),"\uFF1A\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\ndt = D.timestamp2datetime(ts)\nts = D.datetime2timestamp(dt)\nprint(ts)\n# >>> 1634025600.0\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"datetime2str",children:"datetime2str"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L230",children:"datetime2str(dt: datetime, fmt: str) -> str"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07 ",(0,i.jsx)(n.code,{children:"datetime"})," \u8F49\u63DB\u70BA\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"dt"})," (",(0,i.jsx)(n.code,{children:"datetime"}),")\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"str"}),"\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\nts = 1634025600\ndt = D.timestamp2datetime(ts)\ns = D.datetime2str(dt, fmt='%Y-%m-%d %H:%M:%S')\nprint(s)\n# >>> '2021-10-12 16:00:00'\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"str2time",children:"str2time"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L236",children:"str2time(s: str, fmt: str) -> struct_time"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u5B57\u4E32\u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"s"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"struct_time"}),"\uFF1A",(0,i.jsx)(n.code,{children:"struct_time"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\ns = '2021-10-12 16:00:00'\nt = D.str2time(s, fmt='%Y-%m-%d %H:%M:%S')\nprint(t)\n# >>> time.struct_time(tm_year=2021, tm_mon=10, tm_mday=12, tm_hour=16, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=285, tm_isdst=-1)\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"str2datetime",children:"str2datetime"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L242",children:"str2datetime(s: str, fmt: str) -> datetime"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u5B57\u4E32\u8F49\u63DB\u70BA ",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"s"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"datetime"}),"\uFF1A",(0,i.jsx)(n.code,{children:"datetime"}),"\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\ns = '2021-10-12 16:00:00'\ndt = D.str2datetime(s, fmt='%Y-%m-%d %H:%M:%S')\nprint(dt)\n# >>> datetime.datetime(2021, 10, 12, 16, 0)\n"})}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"str2timestamp",children:"str2timestamp"}),"\n",(0,i.jsxs)(n.blockquote,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.a,{href:"https://github.com/DocsaidLab/DocsaidKit/blob/012540eebaebb2718987dd3ec0f7dcf40f403caa/docsaidkit/utils/time.py#L248",children:"str2timestamp(s: str, fmt: str) -> float"})}),"\n"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u5C07\u6642\u9593\u5B57\u4E32\u8F49\u63DB\u70BA\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u53C3\u6578"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"s"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u5B57\u4E32\u3002"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"fmt"})," (",(0,i.jsx)(n.code,{children:"str"}),")\uFF1A\u6642\u9593\u683C\u5F0F\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.strong,{children:"float"}),"\uFF1A\u6642\u9593\u6233\u3002"]}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:["\n",(0,i.jsx)(n.p,{children:(0,i.jsx)(n.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"import docsaidkit as D\n\ns = '2021-10-12 16:00:00'\nts = D.str2timestamp(s, fmt='%Y-%m-%d %H:%M:%S')\nprint(ts)\n# >>> 1634025600.0\n"})}),"\n"]}),"\n"]})]})}function a(e={}){let{wrapper:n}={...(0,d.a)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(m,{...e})}):m(e)}},47105:function(e,n,s){s.d(n,{Z:function(){return t}});let t=s.p+"assets/images/time2time-dcdfdce706509b2a7421e5b03659c93a.png"},50065:function(e,n,s){s.d(n,{Z:function(){return c},a:function(){return r}});var t=s(67294);let i={},d=t.createContext(i);function r(e){let n=t.useContext(d);return t.useMemo(function(){return"function"==typeof e?e(n):{...n,...e}},[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),t.createElement(d.Provider,{value:n},e.children)}}}]);