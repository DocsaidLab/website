"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["7982"],{13429:function(e,n,i){i.r(n),i.d(n,{metadata:()=>o,contentTitle:()=>l,default:()=>p,assets:()=>a,toc:()=>c,frontMatter:()=>t});var o=JSON.parse('{"id":"capybara/pipconfig","title":"PIP Configuration","description":"This section takes a deeper look into the pip configuration mechanism to help you avoid package conflicts and permission issues across multiple Python environments.","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/pipconfig.md","sourceDirName":"capybara","slug":"/capybara/pipconfig","permalink":"/en/docs/capybara/pipconfig","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"sidebarPosition":4,"frontMatter":{"sidebar_position":4},"sidebar":"tutorialSidebar","previous":{"title":"Advanced","permalink":"/en/docs/capybara/advance"},"next":{"title":"\u51FD\u5F0F\u5EAB","permalink":"/en/docs/category/\u51FD\u5F0F\u5EAB"}}'),r=i("85893"),s=i("50065");let t={sidebar_position:4},l="PIP Configuration",a={},c=[{value:"Usage",id:"usage",level:2},{value:"Priority",id:"priority",level:2},{value:"Example Configuration File",id:"example-configuration-file",level:2}];function d(e){let n={admonition:"admonition",code:"code",h1:"h1",h2:"h2",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"pip-configuration",children:"PIP Configuration"})}),"\n",(0,r.jsx)(n.p,{children:"This section takes a deeper look into the pip configuration mechanism to help you avoid package conflicts and permission issues across multiple Python environments."}),"\n",(0,r.jsx)(n.h2,{id:"usage",children:"Usage"}),"\n",(0,r.jsx)(n.p,{children:"On Linux/macOS systems, you can use the following commands to manage local and global configurations:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"python -m pip config [<file-option>] list\npython -m pip config [<file-option>] [--editor <editor-path>] edit\n"})}),"\n",(0,r.jsxs)(n.p,{children:["Here, ",(0,r.jsx)(n.code,{children:"<file-option>"})," can be one of the following options:"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"--global"}),": Specifies the global configuration file for the operating system."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"--user"}),": Specifies the configuration file for the user level."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"--site"}),": Specifies the configuration file for the current virtual environment."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:["The ",(0,r.jsx)(n.code,{children:"--editor"})," parameter allows you to specify the path to an external editor. If this parameter is not provided, the default text editor will be used based on the ",(0,r.jsx)(n.code,{children:"VISUAL"})," or ",(0,r.jsx)(n.code,{children:"EDITOR"})," environment variable."]}),"\n",(0,r.jsx)(n.p,{children:"For example, to edit the global configuration file using the Vim editor, you can run:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"python -m pip config --global --editor vim edit\n"})}),"\n",(0,r.jsx)(n.admonition,{type:"tip",children:(0,r.jsxs)(n.p,{children:["If you are using Windows, the configuration file may be located at ",(0,r.jsx)(n.code,{children:"%APPDATA%\\pip\\pip.ini"})," or ",(0,r.jsx)(n.code,{children:"%HOMEPATH%\\.pip\\pip.ini"}),", among other paths. You can refer to the official documentation or use ",(0,r.jsx)(n.code,{children:"pip config list"})," to further confirm the actual location."]})}),"\n",(0,r.jsx)(n.h2,{id:"priority",children:"Priority"}),"\n",(0,r.jsx)(n.p,{children:"The priority of configuration files is crucial. Below is a list of configuration files that may exist on your machine, sorted by priority:"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Site-level files"}),":","\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf"})}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"User-level files"}),":","\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/home/user/.config/pip/pip.conf"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/home/user/.pip/pip.conf"})}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Global-level files"}),":","\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/etc/pip.conf"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/etc/xdg/pip/pip.conf"})}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"In a Python environment, pip will search for and apply configuration files in the order listed above."}),"\n",(0,r.jsx)(n.p,{children:"Ensuring you are modifying the correct configuration file can help prevent difficult-to-trace errors."}),"\n",(0,r.jsx)(n.h2,{id:"example-configuration-file",children:"Example Configuration File"}),"\n",(0,r.jsx)(n.p,{children:"Here is an example configuration file:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-ini",children:"[global]\nindex-url = https://pypi.org/simple\ntrusted-host = pypi.org\n               pypi.python.org\n               files.pythonhosted.org\nextra-index-url = https://pypi.anaconda.org/simple\n"})}),"\n",(0,r.jsx)(n.p,{children:"In this configuration file, the meanings of the parameters are as follows:"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"index-url"}),": Sets the default source that pip uses when installing packages."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"trusted-host"}),": Lists sources that do not require HTTPS for secure verification, to prevent SSL errors."]}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.code,{children:"extra-index-url"}),": Provides additional source addresses for searching and installing dependencies. Unlike ",(0,r.jsx)(n.code,{children:"index-url"}),", pip will look for ",(0,r.jsx)(n.code,{children:"extra-index-url"})," when the required package is not found in the source specified by ",(0,r.jsx)(n.code,{children:"index-url"}),"."]}),"\n"]}),"\n",(0,r.jsx)(n.admonition,{type:"warning",children:(0,r.jsx)(n.p,{children:"Please note that when using multiple sources, all sources should be trusted because the most suitable version of a package will be selected from these sources during installation. Untrusted sources may pose security risks."})}),"\n",(0,r.jsx)(n.admonition,{type:"tip",children:(0,r.jsxs)(n.p,{children:["If you have a private package server or need to specify a username and password for authentication, you can place your credentials in your ",(0,r.jsx)(n.code,{children:"pip.conf"})," for automation. However, ensure the file permissions are secure."]})})]})}function p(e={}){let{wrapper:n}={...(0,s.a)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},50065:function(e,n,i){i.d(n,{Z:function(){return l},a:function(){return t}});var o=i(67294);let r={},s=o.createContext(r);function t(e){let n=o.useContext(s);return o.useMemo(function(){return"function"==typeof e?e(n):{...n,...e}},[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:t(e.components),o.createElement(s.Provider,{value:n},e.children)}}}]);