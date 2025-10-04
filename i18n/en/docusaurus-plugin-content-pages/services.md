# Technical Services

**Docsaid** is a deep learning studio specializing in **model development and long-term maintenance**.

We transform **real-world business needs** into **maintainable, deployable, and evolvable** AI model modules—focusing on the **quality and reliability of the models themselves**.

Through a **small, dedicated engineering team** and a **stable delivery workflow**, we **work side-by-side with your team**—integrating into your existing systems and processes, filling resource gaps, and supporting model deployment and continuous optimization.

> Note: Our frontend/backend work serves as **lightweight support for model demo, evaluation, and integration**—it is **not** our primary service offering.

---

## Why Work With Us?

- **Model-Centric Focus** – Task-oriented performance goals (mAP, F1, TPR@FPR, Latency, etc.) ensure time and resources are invested in **model quality and stability**.
- **Fast Start, Low Risk** – No team expansion required. We provide ready-to-use scaffolds for data governance, experiment tracking, and benchmarking.
- **Embedded Collaboration** – We integrate with your existing data, product, and engineering teams, reusing your toolchain and minimizing disruption.
- **Measurable Results** – Benchmark dashboards with PR curves, confusion matrices, and multi-model/data comparisons.
- **Inference Optimization** – ONNX/TensorRT, quantization, and **latency budget management**; supports on-premise or private deployment with monitoring and replay analysis.
- **Security & Compliance** – NDA-ready, PII masking, full data and experiment traceability meeting audit requirements.
- **Transparent & Reversible** – Clear iteration goals, change logs, and rollback strategies.

**Flagship Projects**

- **[DocAligner](https://github.com/DocsaidLab/DocAligner)** – Builds metric systems, data versioning, and visualization reports, making every iteration **quantifiable, traceable, and explainable**.
- **[MRZScanner](https://github.com/DocsaidLab/MRZScanner)** – An end-to-end pipeline covering preprocessing, localization, recognition, and verification, emphasizing **practical balance between accuracy and latency**.

---

## Our Expertise

### A. Document Understanding Models

- **A1 Text Detection** – Locating document/scene text regions, rotation correction, and noise suppression.
- **A2 Text Recognition** – OCR for Chinese, English, numeric, and special domains (including **MRZ**); with error correction and dictionary constraints.
- **A3 Layout Understanding** – Block classification, hierarchical parsing, table extraction, and key–value pair parsing (Form/Invoice/ID).
- **A4 Document Alignment** – Multi-template matching, geometric/semantic alignment, and quality measurement (based on the **DocAligner** methodology).

### B. Object Detection Models

- **B1 Training Pipelines** – Data governance/annotation workflows, augmentation strategies, and experiment tracking (mAP, F1, Latency).
- **B2 Framework Re-engineering** – Experience **modularizing frameworks like Ultralytics** into project-specific components (custom Head/Loss/Augmentation/Training commands).
- **B3 Inference Optimization** – ONNX/TensorRT, quantization, batch and stream inference.

### C. Face-Related Models

- **C1 Face Detection** – Robust multi-face and small-face detection under pose variations.
- **C2 Landmark Localization** – 5/68+ points for alignment and pose estimation (roll/pitch/yaw).
- **C3 Face Recognition** – Feature embedding, database management, threshold calibration, and deduplication.
- **C4 Liveness Detection** – Static/dynamic anti-spoofing; extensible to RGB/IR.

---

## Inference, Evaluation & Lightweight Tooling

- **Deployment & Inference**

  - **E1** Model packaging, CI/CD, versioning, rollback, and regression testing workflows.
  - **E2** ONNX deployment: graph optimization, operator alignment, throughput/latency trade-offs, and offline batch processing.
  - **E3** C++ inference engine: on-prem acceleration, resource monitoring, memory and queue management (optimized for air-gapped networks).

- **Benchmarking & Visualization**

  - **F1** PR curves, confusion matrices, accuracy–speed surfaces, and cross-version comparisons (multi-model, multi-dataset).
  - **F2** Periodic reports and milestone reviews for internal discussions and decision-making.

> All these components are **designed to serve the model**, enabling easier **testing, demonstration, and integration**, not large-scale frontend/backend development.

---

## Collaboration Models

- **Model Module Maintenance** – Focused on a **single model module**, delivering improvements periodically (metric gains, performance optimization, data updates).
- **Short-Term Project Engagements** – Targeted tasks such as tracking module integration, inference optimization, or benchmark system design—clear goals, controllable timelines.
- **Long-Term Advisory Support** – Embedding best practices and methodologies into your internal team to build sustainable in-house capability.

> **Already have a model or dataset?** We can take over maintenance and establish benchmarks.
> **Still exploring?** We’ll help you quickly build a **quantifiable starting line** before scaling steadily.

---

## Collaboration Process

1. **Requirement Discussion** – Clarify business objectives, current status, and constraints (approx. 30–60 mins).
2. **Proposal** – Present a breakdown, timeline, and risk assessment; define deliverables and success metrics.
3. **Iterative Execution** – Work in 1–2 week sprints, continuously delivering results and difference reports.
4. **Acceptance & Handover** – Once targets are achieved, hand over code, documentation, and deployment scripts; maintenance renewals available.

---

## Service Overview

Below is a quick summary of our main services—you can select the items most relevant to your needs:

import ServicesAccordion from "@site/src/components/ServicePage/ServicesAccordion";

<ServicesAccordion />;

---

## Frequently Asked Questions

import QnAAccordion from "@site/src/components/ServicePage/QnAAccordion";

<QnAAccordion />;

---

## When We’re a Good Fit (or Not)

- ✅ **Good fit**: You need production-ready models, value long-term maintenance and versioning, and prefer working directly with a small, specialized team.
- ⚠️ **Not a good fit**: You need rapid large-scale manpower deployment, or plan to train large language models (LLMs) requiring massive cloud compute.

---

## Contact

To begin collaboration:

1. **Submit the requirement form** – Whether it’s optimizing an existing model, building a new workflow, or assessing feasibility, feel free to reach out.
2. **Follow-up discussion** – You’ll receive a reply within 1–2 business days; we may schedule a short call if needed.
3. **Project kickoff** – Once both sides confirm the scope and deliverables, the engagement officially begins.

- 📮 Email: **[docsaidlab@gmail.com](mailto:docsaidlab@gmail.com)**
- 🌐 Technical articles & project records: [**https://docsaid.org**](https://docsaid.org/)

---

## Collaboration Form

import CooperationForm from "@site/src/components/ServicePage/CooperationForm";

<CooperationForm />;

---

## Additional Notes

- For **LLM / RAG / Chatbot** NLP projects: we can provide preliminary technical consulting and system evaluations. However, due to compute constraints, **we do not offer full-scale LLM training or large-scale language model development**.
- For **non-local (non-Taiwan) or non-English markets**: timelines may vary due to time zone, NDA, and compliance considerations—please contact us for discussion.
