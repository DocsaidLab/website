"use strict";(self.webpackChunkblog=self.webpackChunkblog||[]).push([["85919"],{54498:function(e,n,i){i.r(n),i.d(n,{metadata:()=>s,contentTitle:()=>a,default:()=>h,assets:()=>l,toc:()=>c,frontMatter:()=>o});var s=JSON.parse('{"id":"capybara/advance","title":"Advanced","description":"Common References","source":"@site/i18n/en/docusaurus-plugin-content-docs/current/capybara/advance.md","sourceDirName":"capybara","slug":"/capybara/advance","permalink":"/en/docs/capybara/advance","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedBy":"zephyr-sh","lastUpdatedAt":1734942587000,"sidebarPosition":3,"frontMatter":{"sidebar_position":3},"sidebar":"tutorialSidebar","previous":{"title":"Installation","permalink":"/en/docs/capybara/installation"},"next":{"title":"PIP Configuration","permalink":"/en/docs/capybara/pipconfig"}}'),r=i("85893"),t=i("50065");let o={sidebar_position:3},a="Advanced",l={},c=[{value:"Common References",id:"common-references",level:2},{value:"Environment Installation",id:"environment-installation",level:2},{value:"Use Docker!",id:"use-docker",level:2},{value:"Install Environment",id:"install-environment",level:3},{value:"Usage",id:"usage",level:2},{value:"Daily Use",id:"daily-use",level:3},{value:"Integrating gosu Configuration",id:"integrating-gosu-configuration",level:3},{value:"Image Build and Execution",id:"image-build-and-execution",level:3},{value:"Installing Internal Packages",id:"installing-internal-packages",level:3},{value:"Common Issue: Permission Denied",id:"common-issue-permission-denied",level:2},{value:"Summary",id:"summary",level:2}];function d(e){let n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",hr:"hr",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"advanced",children:"Advanced"})}),"\n",(0,r.jsx)(n.h2,{id:"common-references",children:"Common References"}),"\n",(0,r.jsx)(n.p,{children:"Before setting up the environment, here are a few official documents worth referencing:"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.strong,{children:"PyTorch Release Notes"})}),"\n",(0,r.jsxs)(n.p,{children:["NVIDIA provides the ",(0,r.jsx)(n.a,{href:"https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html",children:"PyTorch release notes"})," to help you understand the specific versions of PyTorch, CUDA, and cuDNN built into each image, reducing potential dependency conflicts."]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.hr,{}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"NVIDIA Runtime Setup"}),":"]}),"\n",(0,r.jsxs)(n.p,{children:["To use GPU with Docker, refer to the ",(0,r.jsx)(n.a,{href:"https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage",children:"Installation (Native GPU Support)"})," guide to ensure that NVIDIA drivers and container tools are correctly installed on your system."]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.hr,{}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.strong,{children:"NVIDIA Container Toolkit Installation"})}),"\n",(0,r.jsxs)(n.p,{children:["The official guide on ",(0,r.jsx)(n.a,{href:"https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",children:"Installing the NVIDIA Container Toolkit"})," is highly recommended to read carefully as it is crucial for GPU acceleration with Docker."]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.hr,{}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.strong,{children:"ONNXRuntime Release Notes"})}),"\n",(0,r.jsxs)(n.p,{children:["When using ONNXRuntime for inference, if GPU acceleration is needed, refer to the official ",(0,r.jsx)(n.a,{href:"https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements",children:"CUDA Execution Provider guide"})," to ensure version compatibility."]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.h2,{id:"environment-installation",children:"Environment Installation"}),"\n",(0,r.jsx)(n.p,{children:"Most deep learning projects will encounter dependency issues. The typical division of work is as follows:"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Training Environment"}),': PyTorch, OpenCV, CUDA, and cuDNN versions need to be compatible, or you\u2019ll often face issues like "a library cannot be loaded correctly."']}),"\n",(0,r.jsxs)(n.li,{children:[(0,r.jsx)(n.strong,{children:"Deployment Environment"}),": ONNXRuntime, OpenCV, and CUDA also need to match the correct versions, especially for GPU acceleration, where ONNXRuntime-CUDA requires specific CUDA versions."]}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:["One of the most common pitfalls is version mismatch between ",(0,r.jsx)(n.strong,{children:"PyTorch-CUDA"})," and ",(0,r.jsx)(n.strong,{children:"ONNXRuntime-CUDA"}),". When this happens, it\u2019s usually recommended to revert to the official tested combinations or carefully check their dependencies on CUDA and cuDNN versions."]}),"\n",(0,r.jsx)(n.admonition,{type:"tip",children:(0,r.jsx)(n.p,{children:"Why do they never match? \uD83D\uDCA2 \uD83D\uDCA2 \uD83D\uDCA2"})}),"\n",(0,r.jsx)(n.h2,{id:"use-docker",children:"Use Docker!"}),"\n",(0,r.jsxs)(n.p,{children:["To ensure consistency and portability, we ",(0,r.jsx)(n.strong,{children:"strongly recommend"})," using Docker. While it's feasible to set up the environment locally, in the long run, during collaborative development and deployment phases, more time will be spent dealing with unnecessary conflicts."]}),"\n",(0,r.jsx)(n.h3,{id:"install-environment",children:"Install Environment"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"cd Capybara\nbash docker/build.bash\n"})}),"\n",(0,r.jsxs)(n.p,{children:["The ",(0,r.jsx)(n.code,{children:"Dockerfile"})," used for building is also included in the project. If you're interested, you can refer to the ",(0,r.jsx)(n.a,{href:"https://github.com/DocsaidLab/Capybara/blob/main/docker/Dockerfile",children:(0,r.jsx)(n.strong,{children:"Capybara Dockerfile"})}),"."]}),"\n",(0,r.jsxs)(n.p,{children:['In the "inference environment," we use the base image ',(0,r.jsx)(n.code,{children:"nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"}),"."]}),"\n",(0,r.jsx)(n.p,{children:"This image is specifically designed for model deployment, so it doesn't include training environment packages, and you won't find libraries like PyTorch in it."}),"\n",(0,r.jsx)(n.p,{children:"Users can change the base image according to their needs, and related versions will be updated alongside ONNXRuntime."}),"\n",(0,r.jsxs)(n.p,{children:["For more information on inference images, refer to ",(0,r.jsx)(n.a,{href:"https://ngc.nvidia.com/catalog/containers/nvidia:cuda",children:(0,r.jsx)(n.strong,{children:"NVIDIA NGC"})}),"."]}),"\n",(0,r.jsx)(n.h2,{id:"usage",children:"Usage"}),"\n",(0,r.jsx)(n.p,{children:"The following demonstrates a common use case: running an external script via Docker and mounting the current directory inside the container."}),"\n",(0,r.jsx)(n.h3,{id:"daily-use",children:"Daily Use"}),"\n",(0,r.jsxs)(n.p,{children:["Suppose you have a script ",(0,r.jsx)(n.code,{children:"your_scripts.py"})," that you want to run using Python inside the inference container. The steps are as follows:"]}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:["Create a new ",(0,r.jsx)(n.code,{children:"Dockerfile"})," (named ",(0,r.jsx)(n.code,{children:"your_Dockerfile"}),"):"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-Dockerfile",metastring:'title="your_Dockerfile"',children:'# syntax=docker/dockerfile:experimental\nFROM capybara_infer_image:latest\n\n# Set working directory, users can change this based on their needs\nWORKDIR /code\n\n# Example: Install DocAligner\nRUN git clone https://github.com/DocsaidLab/DocAligner.git && \\\n    cd DocAligner && \\\n    python setup.py bdist_wheel && \\\n    pip install dist/*.whl && \\\n    cd .. && rm -rf DocAligner\n\nENTRYPOINT ["python"]\n'})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Build the image:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"docker build -f your_Dockerfile -t your_image_name .\n"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:["Write the execution script (for example, ",(0,r.jsx)(n.code,{children:"run_in_docker.sh"}),"):"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"#!/bin/bash\ndocker run \\\n    --gpus all \\\n    -v ${PWD}:/code \\\n    -it --rm your_image_name your_scripts.py\n"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsxs)(n.p,{children:["Run the script ",(0,r.jsx)(n.code,{children:"run_in_docker.sh"})," to perform inference."]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.admonition,{type:"tip",children:(0,r.jsxs)(n.p,{children:["If you want to enter the container and start bash instead of directly running Python, change ",(0,r.jsx)(n.code,{children:'ENTRYPOINT ["python"]'})," to ",(0,r.jsx)(n.code,{children:'ENTRYPOINT ["/bin/bash"]'}),"."]})}),"\n",(0,r.jsx)(n.h3,{id:"integrating-gosu-configuration",children:"Integrating gosu Configuration"}),"\n",(0,r.jsx)(n.p,{children:'In practice, you may encounter the issue of "output files inside the container being owned by root."'}),"\n",(0,r.jsx)(n.p,{children:"If multiple engineers share the same directory, it could lead to permission issues in the future."}),"\n",(0,r.jsxs)(n.p,{children:["This can be resolved with ",(0,r.jsx)(n.code,{children:"gosu"}),". Here's how we can modify the Dockerfile example:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-Dockerfile",metastring:'title="your_Dockerfile"',children:'# syntax=docker/dockerfile:experimental\nFROM capybara_infer_image:latest\n\nWORKDIR /code\n\n# Example: Install DocAligner\nRUN git clone https://github.com/DocsaidLab/DocAligner.git && \\\n    cd DocAligner && \\\n    python setup.py bdist_wheel && \\\n    pip install dist/*.whl && \\\n    cd .. && rm -rf DocAligner\n\nENV ENTRYPOINT_SCRIPT=/entrypoint.sh\n\n# Install gosu\nRUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*\n\n# Create the entrypoint script\nRUN printf \'#!/bin/bash\\n\\\n    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\\n\\\n        groupadd -g "$GROUP_ID" -o usergroup\\n\\\n        useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\\n\\\n        export HOME=/home/user\\n\\\n        chown -R "$USER_ID":"$GROUP_ID" /home/user\\n\\\n        chown -R "$USER_ID":"$GROUP_ID" /code\\n\\\n    fi\\n\\\n    \\n\\\n    # Check for parameters\\n\\\n    if [ $# -gt 0 ]; then\\n\\\n        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\\n\\\n    else\\n\\\n        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\\n\\\n    fi\' > "$ENTRYPOINT_SCRIPT"\n\nRUN chmod +x "$ENTRYPOINT_SCRIPT"\n\nENTRYPOINT ["/bin/bash", "/entrypoint.sh"]\n'})}),"\n",(0,r.jsx)(n.h3,{id:"image-build-and-execution",children:"Image Build and Execution"}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Build the image:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"docker build -f your_Dockerfile -t your_image_name .\n"})}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:"Run the image (GPU acceleration example):"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"#!/bin/bash\ndocker run \\\n    -e USER_ID=$(id -u) \\\n    -e GROUP_ID=$(id -g) \\\n    --gpus all \\\n    -v ${PWD}:/code \\\n    -it --rm your_image_name your_scripts.py\n"})}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"This way, the output files will automatically have the current user's permissions, making subsequent read/write operations easier."}),"\n",(0,r.jsx)(n.h3,{id:"installing-internal-packages",children:"Installing Internal Packages"}),"\n",(0,r.jsxs)(n.p,{children:["If you need to install ",(0,r.jsx)(n.strong,{children:"private packages"})," or ",(0,r.jsx)(n.strong,{children:"internal tools"})," (such as those hosted on a private PyPI), you can provide authentication credentials during the build process:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-Dockerfile",metastring:'title="your_Dockerfile"',children:'# syntax=docker/dockerfile:experimental\nFROM capybara_infer_image:latest\n\nWORKDIR /code\n\nARG PYPI_ACCOUNT\nARG PYPI_PASSWORD\n\n# Specify your internal package source\nENV SERVER_IP=192.168.100.100:28080/simple/\n\nRUN python -m pip install \\\n    --trusted-host 192.168.100.100 \\\n    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner\n\nENTRYPOINT ["python"]\n'})}),"\n",(0,r.jsx)(n.p,{children:"Then, pass the credentials during the build:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"docker build \\\n    -f your_Dockerfile \\\n    --build-arg PYPI_ACCOUNT=your_account \\\n    --build-arg PYPI_PASSWORD=your_password \\\n    -t your_image_name .\n"})}),"\n",(0,r.jsxs)(n.p,{children:["If your credentials are stored in ",(0,r.jsx)(n.code,{children:"pip.conf"}),", you can also parse the string to inject them, for example:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"docker build \\\n    -f your_Dockerfile \\\n    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \\\n    -t your_image_name .\n"})}),"\n",(0,r.jsx)(n.p,{children:"After building, every time you use it, simply execute the command within Docker as shown above."}),"\n",(0,r.jsx)(n.h2,{id:"common-issue-permission-denied",children:"Common Issue: Permission Denied"}),"\n",(0,r.jsxs)(n.p,{children:["If you encounter the error ",(0,r.jsx)(n.code,{children:"Permission denied"})," when running commands, it's a significant issue."]}),"\n",(0,r.jsxs)(n.p,{children:["After switching users with ",(0,r.jsx)(n.code,{children:"gosu"}),", your permissions will be restricted within certain boundaries. If you need to read/write files inside the container, you may encounter permission issues."]}),"\n",(0,r.jsxs)(n.p,{children:["For example: If you installed the ",(0,r.jsx)(n.code,{children:"DocAligner"})," package, it will automatically download model files during model initialization and place them in Python-related directories."]}),"\n",(0,r.jsx)(n.p,{children:"In this example, the model files are stored by default in:"}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt"})}),"\n"]}),"\n",(0,r.jsx)(n.p,{children:"This path is clearly outside the user's permission scope!"}),"\n",(0,r.jsx)(n.p,{children:"So, you will need to grant the user access to this directory when starting the container. Modify the Dockerfile as follows:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-Dockerfile",metastring:'title="your_Dockerfile" {23}',children:'# syntax=docker/dockerfile:experimental\nFROM capybara_infer_image:latest\n\nWORKDIR /code\n\nRUN git clone https://github.com/DocsaidLab/DocAligner.git && \\\n    cd DocAligner && \\\n    python setup.py bdist_wheel && \\\n    pip install dist/*.whl && \\\n    cd .. && rm -rf DocAligner\n\nENV ENTRYPOINT_SCRIPT=/entrypoint.sh\n\nRUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*\n\nRUN printf \'#!/bin/bash\\n\\\n    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\\n\\\n        groupadd -g "$GROUP_ID" -o usergroup\\n\\\n        useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\\n\\\n        export HOME=/home/user\\n\\\n        chown -R "$USER_ID":"$GROUP_ID" /home/user\\n\\\n        chown -R "$USER_ID":"$GROUP_ID" /code\\n\\\n        chmod -R 777 /usr/local/lib/python3.10/dist-packages\\n\\\n    fi\\n\\\n    \\n\\\n    if [ $# -gt 0 ]; then\\n\\\n        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\\n\\\n    else\\n\\\n        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\\n\\\n    fi\' > "$ENTRYPOINT_SCRIPT"\n\nRUN chmod +x "$ENTRYPOINT_SCRIPT"\n\nENTRYPOINT ["/bin/bash", "/entrypoint.sh"]\n'})}),"\n",(0,r.jsx)(n.p,{children:"If only specific directories need permission changes, you can modify the corresponding paths to avoid overexposing permissions."}),"\n",(0,r.jsx)(n.h2,{id:"summary",children:"Summary"}),"\n",(0,r.jsx)(n.p,{children:"Although using Docker requires more learning, it ensures environment consistency and significantly reduces unnecessary complications during deployment and collaborative development."}),"\n",(0,r.jsx)(n.p,{children:"This investment is definitely worth it, and we hope you enjoy the convenience it brings!"})]})}function h(e={}){let{wrapper:n}={...(0,t.a)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},50065:function(e,n,i){i.d(n,{Z:function(){return a},a:function(){return o}});var s=i(67294);let r={},t=s.createContext(r);function o(e){let n=s.useContext(t);return s.useMemo(function(){return"function"==typeof e?e(n):{...n,...e}},[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:o(e.components),s.createElement(t.Provider,{value:n},e.children)}}}]);