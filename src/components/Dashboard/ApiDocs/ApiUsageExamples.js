// src/components/Dashboard/ApiDocs/ApiUsageExamples.jsx
import styles from "./index.module.css";

import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Card, Collapse, Divider, List, Tabs, Tag, Typography } from "antd";
import React from "react";

const { Panel } = Collapse;
const { Title, Paragraph, Text } = Typography;

/** i18n 字串 */
const apiKeyLocale = {
  "zh-hant": {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      這個 API 用於身分證、證件等影像的自動裁切與修正，可選擇是否進行中心裁切（do_center_crop）。
    `,
    mrzScannerTitle: "MRZ Scanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      這個 API 用於掃描並解析 MRZ 區域，可選擇先對齊影像 (do_doc_align)，再決定是否後處理 (do_postprocess)、中心裁切等。
    `,
    parameters: "參數說明",
    codeExamples: "程式碼範例",
    nameLabel: "參數",
    typeLabel: "型態",
    descLabel: "說明",
    requiredLabel: "必填",
    defaultLabel: "預設值",
    requiredYes: "是",
    requiredNo: "否",
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
    ],
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
    ],
    // 語言
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby"
  },
  en: {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      This API automatically crops and aligns document images (e.g., ID cards).
      Optionally apply center cropping (do_center_crop).
    `,
    mrzScannerTitle: "MRZ Scanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      This API scans and parses the MRZ zone, optionally aligns the document first (do_doc_align),
      then you can decide whether to post-process (do_postprocess) or center-crop.
    `,
    parameters: "Parameters",
    codeExamples: "Code Examples",
    nameLabel: "Name",
    typeLabel: "Type",
    descLabel: "Description",
    requiredLabel: "Required",
    defaultLabel: "Default",
    requiredYes: "Yes",
    requiredNo: "No",
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
    ],
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
    ],
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby"
  },
  ja: {
    docAlignerTitle: "DocAligner",
    docAlignerPath: "POST /docaligner-public-predict",
    docAlignerOverview: `
      このAPIはIDカードやパスポートなどの画像を自動トリミング・補正します。
      必要に応じて中心部分のトリミング(do_center_crop)を行います。
    `,
    mrzScannerTitle: "MRZ Scanner",
    mrzScannerPath: "POST /mrzscanner-public-predict",
    mrzScannerOverview: `
      このAPIはMRZ領域をスキャンして解析します。事前にドキュメントアライメント(do_doc_align)を
      行うことも可能で、後処理(do_postprocess)や中心トリミング(do_center_crop)なども選択できます。
    `,
    parameters: "パラメータ",
    codeExamples: "コード例",
    nameLabel: "名前",
    typeLabel: "型",
    descLabel: "説明",
    requiredLabel: "必須",
    defaultLabel: "デフォルト",
    requiredYes: "はい",
    requiredNo: "いいえ",
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
    ],
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
    ],
    curlLabel: "cURL",
    pythonLabel: "Python",
    nodeLabel: "Node.js",
    jsLabel: "JavaScript",
    javaLabel: "Java",
    rubyLabel: "Ruby"
  }
};

/**
 * 簡易程式碼區塊
 * 可考慮再加 prismjs 或 highlight.js 作語法上色
 */
function CodeBlock({ codeStr }) {
  const style = {
    background: "#f5f5f5",
    borderRadius: 4,
    padding: 12,
    whiteSpace: "pre-wrap",
    fontFamily: "Consolas, Menlo, monospace",
    marginTop: 8
  };
  return <div className={styles.codeBlock}>{codeStr}</div>;
}

/** List 方式顯示參數 (模仿 FastAPI docs) */
function ParamList({ data, text }) {
  return (
    <List
      style={{ marginTop: 8 }}
      itemLayout="vertical"
      dataSource={data}
      renderItem={(item) => (
        <List.Item className={styles.paramListItem}>
          <List.Item.Meta
            title={
              <span className={styles.paramTitle}>
                <Text code>{item.name}</Text>{" "}
                {item.type && <Tag color="blue">{item.type}</Tag>}
                {item.required && <Tag color="red">{text.requiredLabel}</Tag>}
                {item.default && item.default !== "-" && (
                  <Tag color="green">
                    {text.defaultLabel}: {item.default}
                  </Tag>
                )}
              </span>
            }
            description={<span className={styles.paramDesc}>{item.desc}</span>}
          />
        </List.Item>
      )}
    />
  );
}


export default function ApiUsageExamples() {
  const {
    i18n: { currentLocale }
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  /**
   * 未來要擴充到 100 個 API，可以在這裡持續增加。
   * 資料結構說明：
   * {
   *   key: unique key (供 Collapse.Panel 用),
   *   title: 顯示在 Card / Panel 標題,
   *   route: API 路由 (e.g. POST /xxx),
   *   overview: API 簡述,
   *   params: [ { name, type, required, default, desc } ... ],
   *   codeExamples: [ { label: 'cURL', key: 'curl', children: <CodeBlock>... } ... ]
   * }
   */

  const apiDefinitions = [
    {
      key: "docaligner",
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
    },
    {
      key: "mrzscanner",
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
    }
  ];

  return (
    <Collapse
      style={{ marginTop: 24 }}
      accordion
      defaultActiveKey={[apiDefinitions[0].key]}
    >
      {apiDefinitions.map((apiDef) => (
        <Panel
          header={<Text strong>{apiDef.title}</Text>}
          key={apiDef.key}
          // 可在這裡也加上 route
          extra={<Tag color="blue">{apiDef.route}</Tag>}
        >
          <Card bodyStyle={{ padding: "16px 24px" }}>
            <Paragraph>{apiDef.overview}</Paragraph>

            <Divider orientation="left" style={{ marginTop: 24 }}>
              {text.parameters}
            </Divider>
            <ParamList data={apiDef.params} text={text} />

            <Divider orientation="left" style={{ marginTop: 32 }}>
              {text.codeExamples}
            </Divider>
            <Tabs
              defaultActiveKey={apiDef.codeExamples[0].key}
              size="small"
              items={apiDef.codeExamples}
            />
          </Card>
        </Panel>
      ))}
    </Collapse>
  );
}
