"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["82239"],{64220:function(n,e,i){i.r(e),i.d(e,{metadata:()=>r,contentTitle:()=>c,default:()=>p,assets:()=>t,toc:()=>l,frontMatter:()=>a});var r=JSON.parse('{"id":"capybara/funcs/vision/functionals/imcropbox","title":"imcropbox","description":"imcropbox(img Union[Box, np.ndarray], use_pad: bool = False) -> np.ndarray","source":"@site/docs/capybara/funcs/vision/functionals/imcropbox.md","sourceDirName":"capybara/funcs/vision/functionals","slug":"/capybara/funcs/vision/functionals/imcropbox","permalink":"/docs/capybara/funcs/vision/functionals/imcropbox","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"imbinarize","permalink":"/docs/capybara/funcs/vision/functionals/imbinarize"},"next":{"title":"imcropboxes","permalink":"/docs/capybara/funcs/vision/functionals/imcropboxes"}}'),o=i("85893"),s=i("50065");let a={},c="imcropbox",t={},l=[];function d(n){let e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...n.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(e.header,{children:(0,o.jsx)(e.h1,{id:"imcropbox",children:"imcropbox"})}),"\n",(0,o.jsxs)(e.blockquote,{children:["\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/vision/functionals.py#L257",children:"imcropbox(img: np.ndarray, box: Union[Box, np.ndarray], use_pad: bool = False) -> np.ndarray"})}),"\n"]}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsxs)(e.p,{children:[(0,o.jsx)(e.strong,{children:"\u8AAA\u660E"}),"\uFF1A\u4F7F\u7528\u63D0\u4F9B\u7684\u6846\u88C1\u526A\u8F38\u5165\u5F71\u50CF\u3002"]}),"\n"]}),"\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.strong,{children:"\u53C3\u6578"})}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.strong,{children:"img"})," (",(0,o.jsx)(e.code,{children:"np.ndarray"}),")\uFF1A\u8981\u88C1\u526A\u7684\u8F38\u5165\u5F71\u50CF\u3002"]}),"\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.strong,{children:"box"})," (",(0,o.jsx)(e.code,{children:"Union[Box, np.ndarray]"}),")\uFF1A\u88C1\u526A\u6846\u3002\u8F38\u5165\u53EF\u4EE5\u70BA Capybara \u81EA\u5B9A\u7FA9\u7684 Box \u7269\u4EF6\uFF0C\u7531 (x1, y1, x2, y2) \u5EA7\u6A19\u5B9A\u7FA9\uFF0C\u4E5F\u53EF\u4EE5\u662F\u5177\u6709\u76F8\u540C\u683C\u5F0F\u7684 NumPy \u9663\u5217\u3002"]}),"\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.strong,{children:"use_pad"})," (",(0,o.jsx)(e.code,{children:"bool"}),")\uFF1A\u662F\u5426\u4F7F\u7528\u586B\u5145\u4F86\u8655\u7406\u8D85\u51FA\u908A\u754C\u7684\u5340\u57DF\u3002\u5982\u679C\u8A2D\u7F6E\u70BA True\uFF0C\u5247\u5916\u90E8\u5340\u57DF\u5C07\u4F7F\u7528\u96F6\u586B\u5145\u3002\u9810\u8A2D\u70BA False\u3002"]}),"\n"]}),"\n"]}),"\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.strong,{children:"\u50B3\u56DE\u503C"})}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.strong,{children:"np.ndarray"}),"\uFF1A\u88C1\u526A\u5F8C\u7684\u5F71\u50CF\u3002"]}),"\n"]}),"\n"]}),"\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.strong,{children:"\u7BC4\u4F8B"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:"import capybara as cb\n\n# \u4F7F\u7528\u81EA\u5B9A\u7FA9 Box \u7269\u4EF6\nimg = cb.imread('lena.png')\nbox = cb.Box([50, 50, 200, 200], box_mode='xyxy')\ncropped_img = cb.imcropbox(img, box, use_pad=True)\n\n# Resize the cropped image to the original size for visualization\ncropped_img = cb.imresize(cropped_img, [img.shape[0], img.shape[1]])\n"})}),"\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.img,{alt:"imcropbox_box",src:i(9208).Z+"",width:"426",height:"256"})}),"\n"]}),"\n"]})]})}function p(n={}){let{wrapper:e}={...(0,s.a)(),...n.components};return e?(0,o.jsx)(e,{...n,children:(0,o.jsx)(d,{...n})}):d(n)}},9208:function(n,e,i){i.d(e,{Z:function(){return r}});let r=i.p+"assets/images/test_imcropbox-eb25b2219e00eeacef49aa1eef493662.jpg"},50065:function(n,e,i){i.d(e,{Z:function(){return c},a:function(){return a}});var r=i(67294);let o={},s=r.createContext(o);function a(n){let e=r.useContext(s);return r.useMemo(function(){return"function"==typeof n?n(e):{...e,...n}},[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(o):n.components||o:a(n.components),r.createElement(s.Provider,{value:e},n.children)}}}]);