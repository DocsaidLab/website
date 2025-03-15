// src/components/Dashboard/ApiKey/ApiUsageExamples.jsx
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Card, Tabs } from "antd";
import React from "react";

const { TabPane } = Tabs;

const apiKeyLocale = {
  "zh-hant": {
    apiUsageExampleTitle: "API 使用範例",
  },
  en: {
    apiUsageExampleTitle: "API Usage Examples",
  },
  ja: {
    apiUsageExampleTitle: "API利用例",
  },
};

export default function ApiUsageExamples() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  const text = apiKeyLocale[currentLocale] || apiKeyLocale.en;

  const examples = {
    docaligner: {
      curl: `curl -X POST https://api.docsaid.org/docaligner-public-predict \\
  -H "Authorization: Bearer <Your-Public-Token>" \\
  -F "file=@/path/to/your/document.jpg"`,
      python: `import requests

url = "https://api.docsaid.org/docaligner-public-predict"
headers = {"Authorization": "Bearer <Your-Public-Token>"}
files = {"file": open("/path/to/your/document.jpg", "rb")}
response = requests.post(url, headers=headers, files=files)
print(response.json())`,
      node: `const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('/path/to/your/document.jpg'));

axios.post('https://api.docsaid.org/docaligner-public-predict', form, {
  headers: {
    ...form.getHeaders(),
    'Authorization': 'Bearer <Your-Public-Token>'
  }
})
.then(response => {
  console.log(response.data);
})
.catch(error => {
  console.error(error);
});`,
      javascript: `fetch("https://api.docsaid.org/docaligner-public-predict", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <Your-Public-Token>"
  },
  body: new FormData(document.getElementById("uploadForm"))
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error(error));`,
      java: `OkHttpClient client = new OkHttpClient();

MediaType mediaType = MediaType.parse("multipart/form-data");
RequestBody body = new MultipartBody.Builder()
  .setType(MultipartBody.FORM)
  .addFormDataPart("file", "/path/to/your/document.jpg",
      RequestBody.create(new File("/path/to/your/document.jpg"), MediaType.parse("image/jpeg")))
  .build();

Request request = new Request.Builder()
  .url("https://api.docsaid.org/docaligner-public-predict")
  .post(body)
  .addHeader("Authorization", "Bearer <Your-Public-Token>")
  .build();

Response response = client.newCall(request).execute();
System.out.println(response.body().string());`,
      ruby: `require 'net/http'
require 'uri'

uri = URI.parse("https://api.docsaid.org/docaligner-public-predict")
request = Net::HTTP::Post.new(uri)
request["Authorization"] = "Bearer <Your-Public-Token>"
form_data = [['file', File.open('/path/to/your/document.jpg')]]
request.set_form form_data, 'multipart/form-data'

response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
  http.request(request)
end

puts response.body`,
    },
    mrzscanner: {
      curl: `curl -X POST https://api.docsaid.org/mrzscanner-public-predict \\
  -H "Authorization: Bearer <Your-Public-Token>" \\
  -F "file=@/path/to/your/document.jpg"`,
      python: `import requests

url = "https://api.docsaid.org/mrzscanner-public-predict"
headers = {"Authorization": "Bearer <Your-Public-Token>"}
files = {"file": open("/path/to/your/document.jpg", "rb")}
response = requests.post(url, headers=headers, files=files)
print(response.json())`,
      node: `const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const form = new FormData();
form.append('file', fs.createReadStream('/path/to/your/document.jpg'));

axios.post('https://api.docsaid.org/mrzscanner-public-predict', form, {
  headers: {
    ...form.getHeaders(),
    'Authorization': 'Bearer <Your-Public-Token>'
  }
})
.then(response => {
  console.log(response.data);
})
.catch(error => {
  console.error(error);
});`,
      javascript: `fetch("https://api.docsaid.org/mrzscanner-public-predict", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <Your-Public-Token>"
  },
  body: new FormData(document.getElementById("uploadForm"))
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error(error));`,
      java: `OkHttpClient client = new OkHttpClient();

MediaType mediaType = MediaType.parse("multipart/form-data");
RequestBody body = new MultipartBody.Builder()
  .setType(MultipartBody.FORM)
  .addFormDataPart("file", "/path/to/your/document.jpg",
      RequestBody.create(new File("/path/to/your/document.jpg"), MediaType.parse("image/jpeg")))
  .build();

Request request = new Request.Builder()
  .url("https://api.docsaid.org/mrzscanner-public-predict")
  .post(body)
  .addHeader("Authorization", "Bearer <Your-Public-Token>")
  .build();

Response response = client.newCall(request).execute();
System.out.println(response.body().string());`,
      ruby: `require 'net/http'
require 'uri'

uri = URI.parse("https://api.docsaid.org/mrzscanner-public-predict")
request = Net::HTTP::Post.new(uri)
request["Authorization"] = "Bearer <Your-Public-Token>"
form_data = [['file', File.open('/path/to/your/document.jpg')]]
request.set_form form_data, 'multipart/form-data'

response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
  http.request(request)
end

puts response.body`,
    },
  };

  // 建立「第二層 Tabs」的 items - 例如 DocAligner
  const makeInnerTabs = (dataObj) => {
    // dataObj: examples.docaligner 或 examples.mrzscanner
    return [
      {
        label: "cURL",
        key: "curl",
        children: (
          <pre style={preStyle}>
            {dataObj.curl}
          </pre>
        ),
      },
      {
        label: "Python",
        key: "python",
        children: (
          <pre style={preStyle}>
            {dataObj.python}
          </pre>
        ),
      },
      {
        label: "Node.js",
        key: "node",
        children: (
          <pre style={preStyle}>
            {dataObj.node}
          </pre>
        ),
      },
      {
        label: "JavaScript",
        key: "javascript",
        children: (
          <pre style={preStyle}>
            {dataObj.javascript}
          </pre>
        ),
      },
      {
        label: "Java",
        key: "java",
        children: (
          <pre style={preStyle}>
            {dataObj.java}
          </pre>
        ),
      },
      {
        label: "Ruby",
        key: "ruby",
        children: (
          <pre style={preStyle}>
            {dataObj.ruby}
          </pre>
        ),
      },
    ];
  };

  // 第一層 Tabs items
  const topItems = [
    {
      label: "DocAligner",
      key: "docaligner",
      children: (
        <Tabs defaultActiveKey="curl" items={makeInnerTabs(examples.docaligner)} />
      ),
    },
    {
      label: "MRZ Scanner",
      key: "mrzscanner",
      children: (
        <Tabs defaultActiveKey="curl" items={makeInnerTabs(examples.mrzscanner)} />
      ),
    },
  ];

  return (
    <Card title={text.apiUsageExampleTitle} size="small">
      <Tabs defaultActiveKey="docaligner" items={topItems} />
    </Card>
  );
}

// 也可另行定義
const preStyle = {
  background: "#f5f5f5",
  padding: 12,
  whiteSpace: "pre-wrap",
};