"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["23206"],{71468:function(e,s,n){n.r(s),n.d(s,{metadata:()=>t,contentTitle:()=>o,default:()=>u,assets:()=>c,toc:()=>a,frontMatter:()=>l});var t=JSON.parse('{"id":"capybara/funcs/files/get_files","title":"get_files","description":"getfiles(folder Union[str, List[str], Tuple[str]] = None, recursive bool = True, sortpath bool = True) -> List[Path]","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/funcs/files/get_files.md","sourceDirName":"capybara/funcs/files","slug":"/capybara/funcs/files/get_files","permalink":"/en/docs/capybara/funcs/files/get_files","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"frontMatter":{},"sidebar":"tutorialSidebar","previous":{"title":"get_curdir","permalink":"/en/docs/capybara/funcs/files/get_curdir"},"next":{"title":"load_json","permalink":"/en/docs/capybara/funcs/files/load_json"}}'),r=n("85893"),i=n("50065");let l={},o="get_files",c={},a=[];function d(e){let s={a:"a",blockquote:"blockquote",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(s.header,{children:(0,r.jsx)(s.h1,{id:"get_files",children:"get_files"})}),"\n",(0,r.jsxs)(s.blockquote,{children:["\n",(0,r.jsx)(s.p,{children:(0,r.jsx)(s.a,{href:"https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/files_utils.py#L88",children:"get_files(folder: Union[str, Path], suffix: Union[str, List[str], Tuple[str]] = None, recursive: bool = True, return_pathlib: bool = True, sort_path: bool = True, ignore_letter_case: bool = True) -> List[Path]"})}),"\n"]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Description"}),": Retrieves all files in the specified folder. Note that the ",(0,r.jsx)(s.code,{children:"suffix"})," is case-insensitive, but be sure to include the ",(0,r.jsx)(s.code,{children:"."})," as part of the suffix. Many times, files are not found due to this issue."]}),"\n"]}),"\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Parameters"}),":"]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"folder"})," (",(0,r.jsx)(s.code,{children:"Union[str, Path]"}),"): The folder to search for files in."]}),"\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"suffix"})," (",(0,r.jsx)(s.code,{children:"Union[str, List[str], Tuple[str]]"}),"): The file extensions to retrieve. For example: ",(0,r.jsx)(s.code,{children:"['.jpg', '.png']"}),". Defaults to None, meaning all files in the folder are retrieved."]}),"\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"recursive"})," (",(0,r.jsx)(s.code,{children:"bool"}),"): Whether to include files in subfolders. Defaults to True."]}),"\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"return_pathlib"})," (",(0,r.jsx)(s.code,{children:"bool"}),"): Whether to return Path objects. Defaults to True."]}),"\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"sort_path"})," (",(0,r.jsx)(s.code,{children:"bool"}),"): Whether to return the list of paths sorted naturally. Defaults to True."]}),"\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"ignore_letter_case"})," (",(0,r.jsx)(s.code,{children:"bool"}),"): Whether to match file extensions case-insensitively. Defaults to True."]}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Return Value"}),":"]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:[(0,r.jsx)(s.strong,{children:"List[Path]"}),": A list of file paths."]}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Example"}),":"]}),"\n",(0,r.jsx)(s.pre,{children:(0,r.jsx)(s.code,{className:"language-python",children:"import capybara as cb\n\nfolder = '/path/to/your/folder'\nfiles = cb.get_files(folder, suffix=['.jpg', '.png'])\nprint(files)\n# >>> ['/path/to/your/folder/1.jpg', '/path/to/your/folder/2.png']\n"})}),"\n"]}),"\n"]})]})}function u(e={}){let{wrapper:s}={...(0,i.a)(),...e.components};return s?(0,r.jsx)(s,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},50065:function(e,s,n){n.d(s,{Z:function(){return o},a:function(){return l}});var t=n(67294);let r={},i=t.createContext(r);function l(e){let s=t.useContext(i);return t.useMemo(function(){return"function"==typeof e?e(s):{...s,...e}},[s,e])}function o(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),t.createElement(i.Provider,{value:s},e.children)}}}]);