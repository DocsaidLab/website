// src/components/Dashboard/ApiDocs/apis/DocAligner.jsx
import React from "react";
import CodeBlock from "../CodeBlock";

const DocAlignerI18n = {
  "zh-hant": {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      這個 API 用於身分證、證件等影像的自動裁切與修正，可選擇是否進行中心裁切（do_center_crop）。
    `,
    parameters: "參數說明",
    codeExamples: "程式碼範例",
    requiredLabel: "必填",
    defaultLabel: "預設值",
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby",
    docAlignerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "要上傳的影像檔 (jpg, png 等)。"
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "false",
        desc: "是否進行中心裁切。"
      }
    ]
  },
  en: {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      This API automatically crops and aligns document images (e.g., ID cards).
      Optionally apply center cropping (do_center_crop).
    `,
    parameters: "Parameters",
    codeExamples: "Code Examples",
    requiredLabel: "Required",
    defaultLabel: "Default",
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby",
    docAlignerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "Uploaded image file (jpg, png, etc.)."
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "false",
        desc: "Whether to apply center cropping."
      }
    ]
  },
  ja: {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      このAPIはIDカードやパスポートなどの画像を自動トリミング・補正します。
      必要に応じて中心部分のトリミング(do_center_crop)を行います。
    `,
    parameters: "パラメータ",
    codeExamples: "コード例",
    requiredLabel: "必須",
    defaultLabel: "デフォルト",
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby",
    docAlignerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "アップロードする画像ファイル(jpg, pngなど)。"
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "false",
        desc: "中心部分のトリミングを行うかどうか。"
      }
    ]
  }
};

export default function DocAligner(locale = "en") {
  const text = DocAlignerI18n[locale] || DocAlignerI18n.en;
  return {
    key: "docaligner",
    text,
    title: text.docAlignerTitle,
    route: text.docAlignerPath,
    overview: text.docAlignerOverview,
    params: text.docAlignerParams,
    codeExamples: [
      {
        label: text.curlLabel,
        key: "curl",
        children: (
          <CodeBlock
            codeStr={`curl -X POST https://api.docsaid.org/docaligner-public-predict \\
  -H "Authorization: Bearer <Your-Public-Token>" \\
  -F "file=@/path/to/your/document.jpg" \\
  -F "do_center_crop=true"
`}
          />
        )
      },
      {
        label: text.pythonLabel,
        key: "python",
        children: (
          <CodeBlock
            codeStr={`import requests

url = "https://api.docsaid.org/docaligner-public-predict"
headers = {"Authorization": "Bearer <Your-Public-Token>"}
files = {"file": open("/path/to/your/document.jpg", "rb")}
data = {"do_center_crop": "true"}

res = requests.post(url, headers=headers, files=files, data=data)
print(res.json())
`}
          />
        )
      },
      {
        label: text.nodeLabel,
        key: "node",
        children: (
          <CodeBlock
            codeStr={`const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('/path/to/your/document.jpg'));
form.append('do_center_crop', 'true');

axios.post('https://api.docsaid.org/docaligner-public-predict', form, {
  headers: {
    ...form.getHeaders(),
    Authorization: 'Bearer <Your-Public-Token>'
  }
}).then(res => {
  console.log(res.data);
}).catch(err => {
  console.error(err);
});
`}
          />
        )
      },
      {
        label: text.jsLabel,
        key: "js",
        children: (
          <CodeBlock
            codeStr={`const form = new FormData();
form.append("file", document.getElementById("fileInput").files[0]);
form.append("do_center_crop", "true");

fetch("https://api.docsaid.org/docaligner-public-predict", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <Your-Public-Token>"
  },
  body: form
})
  .then(r => r.json())
  .then(data => console.log(data))
  .catch(err => console.error(err));
`}
          />
        )
      },
      {
        label: text.javaLabel,
        key: "java",
        children: (
          <CodeBlock
            codeStr={`OkHttpClient client = new OkHttpClient();

MediaType mediaType = MediaType.parse("multipart/form-data");
RequestBody fileBody = RequestBody.create(
  new File("/path/to/your/document.jpg"),
  MediaType.parse("image/jpeg")
);

MultipartBody requestBody = new MultipartBody.Builder()
  .setType(MultipartBody.FORM)
  .addFormDataPart("file", "document.jpg", fileBody)
  .addFormDataPart("do_center_crop", "true")
  .build();

Request request = new Request.Builder()
  .url("https://api.docsaid.org/docaligner-public-predict")
  .post(requestBody)
  .addHeader("Authorization", "Bearer <Your-Public-Token>")
  .build();

try (Response response = client.newCall(request).execute()) {
  System.out.println(response.body().string());
}
`}
          />
        )
      },
      {
        label: text.rubyLabel,
        key: "ruby",
        children: (
          <CodeBlock
            codeStr={`require 'net/http'
require 'uri'

url = URI("https://api.docsaid.org/docaligner-public-predict")
req = Net::HTTP::Post.new(url)
req["Authorization"] = "Bearer <Your-Public-Token>"

form_data = [
  ["file", File.open("/path/to/your/document.jpg")],
  ["do_center_crop", "true"]
]
req.set_form form_data, "multipart/form-data"

res = Net::HTTP.start(url.hostname, url.port, use_ssl: url.scheme == "https") do |http|
  http.request(req)
end

puts res.body
`}
          />
        )
      }
    ]
  };
}
