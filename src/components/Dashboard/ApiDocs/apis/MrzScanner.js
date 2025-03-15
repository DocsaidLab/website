// src/components/Dashboard/ApiDocs/apis/MrzScanner.jsx
import React from "react";
import CodeBlock from "../CodeBlock";

const MrzScannerI18n = {
  "zh-hant": {
    mrzScannerTitle: "MRZScanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      這個 API 用於掃描並解析 MRZ 區域，可選擇先對齊影像 (do_doc_align)，再決定是否後處理 (do_postprocess)、中心裁切等。
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
    mrzScannerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "要上傳的影像檔 (jpg, png 等)。"
      },
      {
        name: "do_doc_align",
        type: "bool",
        required: false,
        default: "false",
        desc: "是否先使用 DocAligner 對齊影像。"
      },
      {
        name: "do_postprocess",
        type: "bool",
        required: false,
        default: "false",
        desc: "是否在辨識後進行後處理 (去雜訊)。"
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "true",
        desc: "是否在掃描前做中心裁切。"
      }
    ]
  },
  en: {
    mrzScannerTitle: "MRZScanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      This API scans and parses the MRZ zone, optionally aligns the document first (do_doc_align),
      then you can decide whether to post-process (do_postprocess) or center-crop.
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
    mrzScannerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "Uploaded image file (jpg, png, etc.)."
      },
      {
        name: "do_doc_align",
        type: "bool",
        required: false,
        default: "false",
        desc: "Whether to align the document first."
      },
      {
        name: "do_postprocess",
        type: "bool",
        required: false,
        default: "false",
        desc: "Whether to apply post-processing (noise removal)."
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "true",
        desc: "Whether to apply center cropping before scanning."
      }
    ]
  },
  ja: {
    mrzScannerTitle: "MRZScanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      このAPIはMRZ領域をスキャンして解析します。事前にドキュメントアライメント(do_doc_align)を
      行うことも可能で、後処理(do_postprocess)や中心トリミング(do_center_crop)なども選択できます。
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
    mrzScannerParams: [
      {
        name: "file",
        type: "File",
        required: true,
        default: "-",
        desc: "アップロードする画像ファイル(jpg, pngなど)。"
      },
      {
        name: "do_doc_align",
        type: "bool",
        required: false,
        default: "false",
        desc: "先にDocAlignerで画像を整列させるかどうか。"
      },
      {
        name: "do_postprocess",
        type: "bool",
        required: false,
        default: "false",
        desc: "スキャン後にノイズ除去などの後処理を行うかどうか。"
      },
      {
        name: "do_center_crop",
        type: "bool",
        required: false,
        default: "true",
        desc: "スキャン前に中心部分のトリミングを行うかどうか。"
      }
    ]
  }
};

export default function MrzScanner(locale = "en") {
  const text = MrzScannerI18n[locale] || MrzScannerI18n.en;
  return {
    key: "mrzscanner",
    text,
    title: text.mrzScannerTitle,
    route: text.mrzScannerPath,
    overview: text.mrzScannerOverview,
    params: text.mrzScannerParams,
    codeExamples: [
      {
        label: text.curlLabel,
        key: "curl",
        children: (
          <CodeBlock
            codeStr={`curl -X POST https://api.docsaid.org/mrzscanner-public-predict \\
  -H "Authorization: Bearer <Your-Public-Token>" \\
  -F "file=@/path/to/your/document.jpg" \\
  -F "do_doc_align=true" \\
  -F "do_postprocess=false" \\
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

url = "https://api.docsaid.org/mrzscanner-public-predict"
headers = {"Authorization": "Bearer <Your-Public-Token>"}
files = {"file": open("/path/to/your/document.jpg", "rb")}
data = {
  "do_doc_align": "true",
  "do_postprocess": "false",
  "do_center_crop": "true"
}

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
form.append('do_doc_align', 'true');
form.append('do_postprocess', 'false');
form.append('do_center_crop', 'true');

axios.post('https://api.docsaid.org/mrzscanner-public-predict', form, {
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
form.append("do_doc_align", "true");
form.append("do_postprocess", "false");
form.append("do_center_crop", "true");

fetch("https://api.docsaid.org/mrzscanner-public-predict", {
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
  .addFormDataPart("do_doc_align", "true")
  .addFormDataPart("do_postprocess", "false")
  .addFormDataPart("do_center_crop", "true")
  .build();

Request request = new Request.Builder()
  .url("https://api.docsaid.org/mrzscanner-public-predict")
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

url = URI("https://api.docsaid.org/mrzscanner-public-predict")
req = Net::HTTP::Post.new(url)
req["Authorization"] = "Bearer <Your-Public-Token>"

form_data = [
  ["file", File.open("/path/to/your/document.jpg")],
  ["do_doc_align", "true"],
  ["do_postprocess", "false"],
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
