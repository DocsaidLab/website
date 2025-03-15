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

  return (
    <Card title={text.apiUsageExampleTitle} size="small">
      <Tabs defaultActiveKey="docaligner">
        <TabPane tab="DocAligner" key="docaligner">
          <Tabs defaultActiveKey="curl">
            <TabPane tab="cURL" key="curl">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.curl}
              </pre>
            </TabPane>
            <TabPane tab="Python" key="python">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.python}
              </pre>
            </TabPane>
            <TabPane tab="Node.js" key="node">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.node}
              </pre>
            </TabPane>
            <TabPane tab="JavaScript" key="javascript">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.javascript}
              </pre>
            </TabPane>
            <TabPane tab="Java" key="java">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.java}
              </pre>
            </TabPane>
            <TabPane tab="Ruby" key="ruby">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.docaligner.ruby}
              </pre>
            </TabPane>
          </Tabs>
        </TabPane>
        <TabPane tab="MRZ Scanner" key="mrzscanner">
          <Tabs defaultActiveKey="curl">
            <TabPane tab="cURL" key="curl">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.curl}
              </pre>
            </TabPane>
            <TabPane tab="Python" key="python">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.python}
              </pre>
            </TabPane>
            <TabPane tab="Node.js" key="node">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.node}
              </pre>
            </TabPane>
            <TabPane tab="JavaScript" key="javascript">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.javascript}
              </pre>
            </TabPane>
            <TabPane tab="Java" key="java">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.java}
              </pre>
            </TabPane>
            <TabPane tab="Ruby" key="ruby">
              <pre style={{ background: "#f5f5f5", padding: 12, whiteSpace: "pre-wrap" }}>
                {examples.mrzscanner.ruby}
              </pre>
            </TabPane>
          </Tabs>
        </TabPane>
      </Tabs>
    </Card>
  );
}
