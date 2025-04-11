# Crowd Flow and Demographic Analysis System | Modular Proposal Report

Through this proposal, we aim to assist you in building a **real-time and flexible** crowd flow and demographic analysis system.

This system allows for continuous monitoring of crowd density at any given location, while also providing structural information such as age, gender, etc. This enables more data-driven decisions for **space planning, operational strategies, or marketing adjustments**.

The system is suitable for various environments such as malls, retail stores, exhibition halls, public spaces, or smart cities. In the case of privacy or regulatory concerns (such as personal data protection or regional compliance issues), the system can also be customized to meet legal requirements through **de-identification** and **customized settings**. Additionally, depending on your budget and environmental conditions, it offers flexible deployment options (local, edge devices, or internal network environments), and the modular design allows for easy upgrades to minimize the risk of redundant development or "vendor lock-in."

> **Additional Notes:**
> To enhance the feasibility of this project across various practical applications, we will consider factors such as environmental lighting, camera positioning, privacy conditions, and network infrastructure. Early-stage small-scale pilots will be conducted to ensure the system undergoes practical verification and adjustments before full deployment.

## 1. Project Background and Goals

- **Precise Space Allocation**
  Through peak period and crowd distribution analysis, optimize personnel scheduling and flow planning.

- **Enhancing Operational Efficiency**
  Use demographic data such as age and gender to create more tailored promotional or service strategies for target groups.

- **Implementing Privacy Protection**
  Supports de-identification and customized privacy settings to minimize the risk of personal data leaks.

- **Flexible Deployment Across Multiple Environments**
  If you already have camera equipment or are considering adding cameras, the system allows you to select features based on your specific needs.

- **Compliance and Scalability**
  In response to regional regulations (e.g., GDPR) or large-scale multi-camera integrations, the system can be modularly adjusted.

> **Additional Notes:**
> To enable decision-makers and on-site managers to quickly grasp changes, we can also provide real-time notifications or graphical interfaces to assist with immediate crowd fluctuations, ensuring that operational decisions and space allocation are based on the most timely information.

---

## 2. Modules and Functional Details

To meet customization needs, the system adopts a "modular" design. Below is a list of core modules and their respective time estimates, allowing you to flexibly choose necessary components based on your budget and project stages. If your budget is limited, you can opt for essential modules for a Minimum Viable Product (MVP) or Proof of Concept (POC), and gradually expand later.

---

### 2.1 Model Modules

| Submodule Name                       | Function Description                                                                                                          | Common Technical Challenges                                        | Time Estimate (days) | Standalone Operation      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------- | ------------------------- |
| Pedestrian Detection & Preprocessing | Real-time detection of all pedestrians in the frame, with bounding box/coordinate marking                                     | Lighting variations, occlusion, and resolution affecting stability | 6                    | ✅                        |
| Pedestrian Tracking                  | Assigning an ID to each pedestrian and continuously tracking their movement, supports single and multiple camera integrations | Cross-entrance ID confusion, multi-camera overlap                  | 5                    | Requires Detection Module |
| Age and Gender Prediction            | Estimation of the pedestrian's age group and gender, supports profile and blurry face handling                                | Blurry faces, side-angle reduces accuracy                          | 6                    | Can be cropped            |
| Facial De-identification (Optional)  | Blurring/obfuscating personal features to avoid retaining identifiable characteristics                                        | Needs to obscure without affecting statistical accuracy            | 3                    | Optional                  |
| Hotspot Analysis (Optional)          | Identifying high-traffic areas through pedestrian movement patterns                                                           | Aggregation across time axes and converting coordinate hotspots    | 5                    | Optional                  |
| Multi-Camera Merging Strategy        | Integrating personnel identification across multiple angles to avoid duplicate calculations                                   | No GPS, path overlap strategy required                             | 7                    | Optional                  |
| Model Performance Optimization       | Ensuring inference speed and stability in high-density environments                                                           | Stress testing and parameter fine-tuning                           | 4                    | Requires Detection Module |
| Model Version Management             | Supports multiple inference formats (ONNX, TensorRT) for flexible upgrades and replacements                                   | Format conversion stability and inference consistency              | 4                    | Optional                  |

---

#### 📌 Common Field Challenges and Solutions: High-Angle Camera Perspectives

If the system is deployed on the ceiling or high up, using a downward or near-vertical angle for imaging will present the following technical impacts. We recommend considering these factors during the planning phase:

1. **Facial Features Limited, Attribute Prediction (Age/Gender) Accuracy Decreases**
   Since most facial features (such as eyes, nose, mouth) are difficult to identify in a downward view, this impacts facial detection and subsequent attribute prediction. For these environments, it’s suggested to either disable relevant modules or use body outline estimation (though this comes with lower accuracy).

2. **Pedestrian Detection and Tracking Accuracy Affected by Body Distortion**
   High-angle views cause body proportions to distort (e.g., enlarging the head), which may affect boundary detection and ID stability. In this case, it's recommended to use optimized detection models for downward angles and track based on footpoints to improve hotspot analysis' spatial accuracy.

3. **Difficulty in Multi-Camera Merging and Cross-Area Tracking**
   Overlap between high-angle cameras is minimal, making cross-camera ID matching and merging difficult. For multi-camera setups, we suggest splitting the image into segments for independent counting and later combining outputs or introducing environmental markers (e.g., indoor maps) to aid integration.

> **Additional Notes:**
> If high-angle cameras are deployed alongside cameras from other angles, integration and scene reference points can be synchronized across different views to enhance overall coverage and detection accuracy. If your deployment covers a large area, we recommend defining camera overlap and boundaries in the initial planning phase to avoid overly complex merging strategies later.

---

### 2.2 Backend Modules

| Submodule Name                                  | Function Description                                                                                                                    | Common Technical Challenges                                                        | Time Estimate (days) | Standalone Operation    |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------- | ----------------------- |
| API Architecture & Development                  | Provides data read/write services for frontend and third-party systems, supports extensions, OAuth2 authentication, and version control | Data consistency from multiple sources, API error handling, and version iterations | 4                    | Requires DB Module      |
| Data Storage Design                             | Establish SQL/NoSQL architecture and indexing, handle high-frequency writes and queries                                                 | Balancing write load and query performance                                         | 4                    | ✅                      |
| Scheduling & Statistical Computation (Optional) | Automatically generate reports and backups, supports flexible daily/weekly/monthly scheduling with CSV/JSON/PDF output formats          | Error retry in scheduling and data integrity                                       | 4                    | Depends on Data Storage |
| Authorization System (Optional)                 | Multiple role login (admin/viewer/operational), controls access to sensitive data, supports role and feature-based permissions          | Permission layers and module expansion                                             | 3                    | Optional                |
| Demographic Statistics (Optional)               | Statistics on age and gender distribution, with time segmentation aggregation and data partitioning                                     | Synchronizing and merging data from multiple fields                                | 3                    | Optional                |
| Multi-Field Merging Strategy (Optional)         | Aggregates data from multiple locations and generates cross-field reports, supports timezone and data field alignment                   | Aligning inconsistent field standards across different environments                | 3                    | Optional                |
| System Logs & Alerts (Optional)                 | Records errors and abnormal events, supports automatic notification mechanisms like Email/Slack/Webhook                                 | Performance monitoring and alert categorization                                    | 3                    | Optional                |
| Testing & API Documentation                     | Writes Swagger/Postman documentation to ensure maintainability and consistency                                                          | Field version synchronization and test environment setup                           | 4                    | Requires API Module     |
| Dockerization & Deployment (Optional)           | Uses containers for rapid deployment and environment encapsulation, supports GitHub Actions/GitLab CI/CD processes                      | Multi-environment compatibility and dependency management                          | 5                    | Optional                |
| Asynchronous Task Queue (Optional)              | Uses Celery/Kafka to handle asynchronous tasks and distribute high-frequency data/event processing                                      | Task stability and message queue management                                        | 4                    | Optional                |
| Hotspot Data Caching (Optional)                 | Uses Redis to cache frequently queried statistical data, improving frontend query performance and response speed                        | Cache update strategy and data consistency                                         | 3                    | Optional                |

> **Additional Notes:**
> Due to the wide scope and extensibility of backend tasks, the actual implementation will need to be flexibly integrated with the existing system architecture (such as existing databases, API gateways, etc.). If you already have a mature internal system, we will aim to minimize redundant construction and prioritize integrating with the existing environment.

---

#### 📌 Module Integration and Flexible Expansion Recommendations

To balance system flexibility and performance stability, we recommend adopting a "microservices + asynchronous processing + cache optimization" three-layer architecture for the backend modules:

- **Microservice-based API Architecture**: Each module can be independently deployed and maintained, supporting future field expansions and hot updates.
- **Asynchronous Queue Processing**: For large-scale inference and cross-field data aggregation, we recommend using Celery task queues to prevent blocking of the main system.
- **Real-Time Caching Strategy**: Frequently queried statistical data will be cached in Redis to improve response speed and reduce database load.

This architecture also supports Docker deployment and CI/CD processes, making it easier for businesses to deploy in the cloud or private environments.

> **Additional Notes:**
> If future expansion involves multiple countries, multiple data centers, or high-concurrency scenarios, the flexibility of backend scalability becomes crucial. We can design horizontal scaling mechanisms in advance based on predicted crowd volumes or cross-timezone needs, including automated deployment scripts and load balancing configurations to ensure the system remains stable under high traffic.

---

### 2.3 Frontend Modules

| Submodule Name                           | Function Description                                                                                                             | Common Technical Challenges                                                | Time Estimate (days) | Standalone Operation    |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------- | ----------------------- |
| Dashboard Design                         | Design the main UI, including the login homepage, overview page, module entry, real-time alert bar, etc.                         | Component performance, chart rendering, and interaction smoothness         | 5                    | Requires API Interface  |
| Login & Role Permission Management       | Frontend interfaces for different roles, managing access to various views and actions                                            | Permission display logic and data security settings                        | 3                    | Optional                |
| Traffic Trend Visualization              | Line/bar charts displaying traffic trends, supports day/week/month switching, zooming, and tooltips                              | Handling missing data and frontend performance                             | 4                    | Requires DB & API       |
| Demographic Distribution Charts          | Pie charts displaying gender and age distribution, supports filtering by area and range estimation                               | Maintaining visual accuracy when data is incomplete                        | 3                    | Optional                |
| Peak Time Annotation                     | Automatically marks peak traffic periods, provides on-screen annotations and exportable data                                     | Peak time misidentification and standard definition                        | 2                    | Optional                |
| API Integration                          | Integrates frontend and backend data, supports timestamp alignment, error retries, token authorization, and cross-domain control | Error synchronization and security settings                                | 5                    | ✅ (Depends on Backend) |
| Multi-Camera Switching                   | Displays live feeds from multiple locations, supports RTSP streaming, snapshot cycling, and manual switching                     | Screen latency and synchronization control                                 | 3                    | Optional                |
| Responsive Design                        | Adapts for mobile/tablet/desktop screens, supports basic PWA architecture to improve mobile device experience                    | CSS configuration, cross-device testing                                    | 4                    | Optional                |
| Report Export                            | Exports statistical charts and data to Excel/CSV/PDF, supports column selection and format conversion                            | Font/format compatibility and export chart rendering                       | 3                    | Optional                |
| Real-Time Alert Mechanism                | Customizable threshold alerts, displaying pop-up warnings, sound prompts, and supports Email/SMS notification options            | Threshold definition, notification frequency, and multi-channel management | 3                    | Optional                |
| Customizable Dashboard Module (Optional) | Users can drag and combine dashboard blocks to customize content and configuration                                               | Block dependencies and configuration saving                                | 4                    | Optional                |
| User Behavior Tracking (Optional)        | Records user clicks and module usage, providing UI optimization and usage analysis data                                          | Anonymity protection, frontend and backend data separation                 | 3                    | Optional                |

> **Additional Notes:**
> The frontend interface not only provides a visual dashboard but also allows different departments to set the visibility or functionality of various items based on their roles. If you already have an internal BI platform, we can limit the integration to just the necessary API connections and report export functions to reduce redundant development costs.

---

#### 📌 Frontend Module Configuration Recommendations and Use Case Scenarios

The frontend module design follows the core principles of "flexible toggles," "role-based access," and "high interactivity," allowing different departments to adjust the visual interface and functionality modules based on their needs. Common use cases are as follows:

1. **Operations and Strategy Team**

   - Recommended modules: `Traffic Trend Visualization`, `Demographic Distribution Charts`, `Peak Time Annotation`, `Report Export`
   - Purpose: Monitor customer traffic changes, assess marketing effectiveness, and generate weekly or monthly operational reports.

2. **On-Site Management Staff**

   - Recommended modules: `Dashboard Design`, `Real-Time Alert Mechanism`, `Multi-Camera Switching`
   - Purpose: Quickly monitor if crowd density exceeds limits, review spatial distribution and live camera feeds.

3. **IT Department or Internal System Integrators**

   - Recommended modules: `API Integration`, `Login & Role Permission Management`, `User Behavior Tracking (Optional)`
   - Purpose: Integrate internal account permissions, connect with existing systems, or monitor data traffic.

4. **Cross-Department Management & Senior Executives**
   - Recommended module: `Customizable Dashboard Module (Optional)`, customize views based on key areas of concern.
   - Purpose: Create personalized visual overviews without interfering with other departments' settings.

> **Additional Notes:**
> If some users only need key data (e.g., total daily foot traffic or changes in specific time slots), the frontend can provide a simplified "Quick View" mode for fast chart or data output. For deep analysis needs, a full dashboard will allow multi-dimensional filtering and exploration.

---

### 2.4 Edge Deployment Modules

| Submodule Name                            | Function Description                                                                                                        | Common Technical Challenges                                                      | Time Estimate (days) | Standalone Operation |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------- | -------------------- |
| Hardware Evaluation & Selection           | Recommends devices like Jetson/Coral based on site conditions, considering outdoor shielding, power, and PoE configurations | Device variability, space limitations, and cost estimation                       | 3                    | Optional             |
| Model Conversion & Acceleration           | Converts models to ONNX/TensorRT/FP16, adjusts batch size to enhance inference performance                                  | Balancing accuracy and speed, adjusting TRT compatibility                        | 5                    | Requires Base Model  |
| Local Caching & Resend Mechanism          | Caches statistical results/images when network is unstable, automatically resends once connection is restored               | Offline data synchronization verification and concurrency consistency management | 4                    | Optional             |
| Remote Management & Monitoring            | Provides WebUI and SSH interfaces for device status checking, parameter setting, and rebooting                              | Security of control and interface integration                                    | 4                    | Optional             |
| Thermal & Power Testing                   | Evaluates if devices overheat or power off during prolonged operation, adjusts fan or UPS configurations                    | Site condition variability, test durability, and unstable measurement points     | 4                    | Optional             |
| Batch Update & Reporting                  | Supports OTA model/software batch updates, provides update results and error logs                                           | Multi-node synchronization control and error recovery                            | 4                    | Optional             |
| Watchdog Auto-Restart                     | Detects GPU anomalies, memory overflow, or service interruptions, automatically restarts and logs the status                | Defining error conditions and fault tolerance                                    | 3                    | Optional             |
| Bandwidth & Image Optimization            | Returns statistical data or JPEG compressed images, supports frame rate and image quality adjustments                       | Real-time vs image quality trade-off, difficult bandwidth estimation             | 3                    | Optional             |
| Edge Device Dashboard (Optional)          | Displays real-time status, upload rates, error alarms, and resource usage for each node, with manual restart option         | Multi-node data integration and frontend presentation                            | 4                    | Optional             |
| Device Environment Diagnosis (Optional)   | Automatically checks camera stream stability, latency, and FPS, provides pre-deployment warnings                            | Stream protocol compatibility, real-time diagnostic flexibility                  | 3                    | Optional             |
| Regional Deployment Scheduling (Optional) | Centralizes task distribution, device settings, and deployment scheduling for multi-site deployments                        | Task assignment and concurrent deployment strategies                             | 4                    | Optional             |

> **Additional Notes:**
> Edge device selection and deployment are often limited by onsite power and network conditions, as well as hardware costs. If bandwidth is limited, it's recommended to perform basic inference at the edge and only send useful statistical results to reduce image transmission. If power or space isn't an issue, a higher-end GPU server may be considered to achieve better inference performance and support for multiple cameras.

---

#### 📌 Edge Deployment Strategy Recommendations: Single-Node vs Multi-Node Scalable Deployment

This system supports a variety of edge deployment strategies, which can be flexibly adjusted based on site scale, network conditions, and operational maintenance. Below are module selection recommendations for two common scenarios:

##### ✅ **Scenario 1: Single-Node Deployment, Emphasizing Stability and Auto-Recovery**

Suitable for: Small to medium-sized stores, independent kiosks, temporary exhibition areas, etc.

- **Recommended Modules**:
  - `Hardware Evaluation & Selection`
  - `Model Conversion & Acceleration`
  - `Local Caching & Resend Mechanism`
  - `Watchdog Auto-Restart`
  - `Bandwidth & Image Optimization`
- **Deployment Focus**:
  - Enhance offline tolerance
  - Reduce manual maintenance burden
  - Enable auto-recovery and network compensation

##### ✅ **Scenario 2: Multi-Node Deployment, Requires Centralized Control and Remote Maintenance**

Suitable for: Chain stores, large exhibition halls, multi-location malls, smart buildings, etc.

- **Recommended Modules**:
  - `Remote Management & Monitoring`
  - `Batch Update & Reporting`
  - `Edge Device Dashboard`
  - `Regional Deployment Scheduling (Optional)`
  - `Device Environment Diagnosis (Optional)`
- **Deployment Focus**:
  - Centralized control and OTA updates
  - Real-time monitoring of multi-node statuses
  - Reduce on-site inspections and deployment errors

> **Additional Notes:**
> Whether deploying a single-node or multi-node setup, if cross-departmental or cross-system data integration is expected, it's recommended to plan the data synchronization method (e.g., via API or message queues) from the start and leave room for fast expansion later.

---

### 2.5 Development Process Recommendations

This system recommends adopting a "**Progressive Implementation + Agile Development**" approach to reduce one-time investment risks.

The development process will be divided into three phases by modules, each phase will produce a concrete, testable MVP (Minimum Viable Product), and modules can be flexibly adjusted or split based on budget and schedule.

---

- **Phase 1 | Core Model Construction**

  - **Main Content**: Complete the base model and data pipeline construction, establish the backend API framework
  - **Related Modules**: Pedestrian Detection, Tracking, Age/Gender Prediction (2.1), CLI, API Architecture, Data Storage (2.2)
  - **Deliverables**:
    - CLI test scripts
    - API interface documentation
    - Image input → structured data output process
  - **Estimated Duration**: About 30 days

---

- **Phase 2 | Visualization and Statistical Module Construction**

  - **Main Content**: Create the first version of frontend dashboards and chart interfaces, along with scheduling and data aggregation
  - **Related Modules**: Traffic Trend Charts, Demographic Distribution Charts (2.3), Scheduling & Statistical Modules, Multi-Site Merging (2.2)
  - **Deliverables**:
    - Frontend screen prototypes
    - Report templates
    - Automatic scheduling and daily statistical data
  - **Estimated Duration**: About 24 days

---

- **Phase 3 | Integration Deployment and Edge Module Testing**

  - **Main Content**: Establish the permission system, integrate frontend and backend, Dockerize, and conduct edge device on-site testing and optimization
  - **Related Modules**: Permission Management, Alert Mechanisms, Docker (2.2 – 2.3), Edge Selection and Management Modules (2.4)
  - **Deliverables**:
    - Deployable version
    - Edge device connection test records
    - Multi-camera stitching test reports
  - **Estimated Duration**: About 42 days

---

> - Each phase can be independently executed and accepted, suitable for planning sprints on a weekly basis.
> - If the initial focus is only on verifying the model and data flow, you can execute **Phase 1** as a POC (Proof of Concept).
> - Development can be adjusted for **concurrent frontend and backend development** (e.g., frontend integration as soon as the API is completed).
> - If Phase 3 requires more privacy and regulatory testing, the testing duration can be extended to ensure compliance and stability before full launch.

---

## 3. Solutions and Feature Description

To address the diverse needs of different environments, budgets, and development stages, we have planned **four flexible solutions** that can be combined through the "core feature modules," allowing you to expand or upgrade at any time.

### 3.1 Solution Overview

The table below outlines the overall solutions, from early concept verification (POC) to complete system implementation, along with reference pricing for licensing and perpetual purchase models (actual costs depend on functionality and environment):

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '1rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

| Solution Level | Target Audience                                             | Feature Focus                                                                                                                 | Licensing Fee (USD) | Perpetual Purchase Fee (USD) |
| -------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------------- |
| **POC**        | Initial exploration, short-term concept validation          | Uses default model combinations, quickly deployable on a single machine to validate system effectiveness and data feasibility | USD 5,000           | —                            |
| **L1**         | Single-node deployment, formal long-term use                | Adjustable models and tracking modules, supports scheduling, automatic storage, and future expansions                         | USD 20,000          | USD 60,000                   |
| **L2**         | Internal analysis and reporting needs                       | Adds visualization and statistical modules, supports backend browsing and trend monitoring                                    | USD 32,000          | USD 96,000                   |
| **L3**         | Multi-department, multi-site, requires external integration | Complete backend system, API integration, multi-camera and edge device deployment                                             | USD 50,000          | USD 150,000                  |

</div>

> **Licensing Model**: Grants system usage rights, with source code and intellectual property retained by our team. If upgrades or maintenance are needed, additional service plans can be negotiated.

> **Perpetual Purchase Model**: Grants full ownership of the source code and intellectual property, allowing for in-house development and maintenance; technical support can be arranged through a maintenance contract.

> **Additional Notes**:
> If you start with a POC and verify its effectiveness, you can flexibly upgrade to L1 / L2 / L3 without incurring duplicate or wasted development costs. We use a modular program architecture, allowing seamless integration of previous POC results, maximizing project development efficiency and continuity.

### 3.2 Module Function Comparison

The table below compares the core feature modules of each version, enabling flexible selection based on actual needs or future upgrades:

1. **Model and Detection Capabilities**

   <div style={{
   whiteSpace: 'nowrap',
   overflowX: 'auto',
   fontSize: '1rem',
   lineHeight: '0.8',
   justifyContent: 'center',
   display: 'flex',
   }}>

   | Module Item                                                   | Simplified Version | Level 1          | Level 2 | Level 3 |
   | ------------------------------------------------------------- | ------------------ | ---------------- | ------- | ------- |
   | 🎯 Crowd Detection Model                                      | ✅ (Fixed Model)   | ✅ (Adjustable)  | ✅      | ✅      |
   | 🔁 Multi-Person Tracking Module                               | ❌                 | ✅               | ✅      | ✅      |
   | 🧠 Age/Gender Prediction Model                                | ✅ (Default Model) | ✅ (Upgradeable) | ✅      | ✅      |
   | ⚙️ Model Parameter Settings (Confidence Threshold, IOU, etc.) | ❌                 | ✅               | ✅      | ✅      |
   | 🔄 Model Upgrade/Replacement (ONNX/PyTorch)                   | ❌                 | ✅               | ✅      | ✅      |

   </div>

2. **System Modules and Deployment Flexibility**

   <div style={{
   whiteSpace: 'nowrap',
   overflowX: 'auto',
   fontSize: '1rem',
   lineHeight: '0.8',
   justifyContent: 'center',
   display: 'flex',
   }}>

   | Module Item                                 | Simplified Version | Level 1     | Level 2     | Level 3     |
   | ------------------------------------------- | ------------------ | ----------- | ----------- | ----------- |
   | 📦 CLI Interface                            | ✅                 | ✅          | ✅          | ✅          |
   | 💾 Data Storage (CSV / SQLite)              | ✅ (CSV)           | ✅ (SQLite) | ✅          | ✅          |
   | 🔁 Auto Scheduling and Daily Output         | ❌                 | ✅          | ✅          | ✅          |
   | 🐳 Docker Deployment                        | ❌                 | ☑️ Optional | ☑️ Optional | ✅          |
   | 🧱 Edge Deployment Modules (Jetson / Coral) | ❌                 | ☑️ Optional | ☑️ Optional | ☑️ Optional |

   </div>

3. **Visualization and Backend Modules**

   <div style={{
   whiteSpace: 'nowrap',
   overflowX: 'auto',
   fontSize: '1rem',
   lineHeight: '0.8',
   justifyContent: 'center',
   display: 'flex',
   }}>

   | Module Item                               | Simplified Version | Level 1 | Level 2 | Level 3 |
   | ----------------------------------------- | ------------------ | ------- | ------- | ------- |
   | 🧪 Data Aggregation and Daily Statistics  | ❌                 | ✅      | ✅      | ✅      |
   | 📊 Simple Reporting Interface (Streamlit) | ❌                 | ❌      | ✅      | ❌      |
   | 🧱 Frontend Dashboard (React)             | ❌                 | ❌      | ❌      | ✅      |
   | 🔐 Login and Permission Management        | ❌                 | ❌      | ❌      | ✅      |
   | 🛠 API Query Interface (FastAPI)           | ❌                 | ❌      | ❌      | ✅      |

   </div>

> - **Simplified Version (POC)**: Focuses on rapid testing, not expandable, no multi-person tracking or model parameter adjustments.
> - **Level 1**: Ready for formal deployment, supports scheduling and modular expansion.
> - **Level 2**: Enhanced visualization reports, making internal presentations and strategy analysis more intuitive.
> - **Level 3**: Complete architecture, supporting multi-department collaboration and external integrations.

> **Additional Notes**:
> You can start directly with L1 / L2 if your budget and department needs align, without necessarily going through the POC phase. If your organization already has a mature backend and frontend team, you can focus more on the model and deployment layers.

---

## 4. Data Privacy and Regulatory Compliance

The system was designed with careful consideration of privacy and regulatory risks across various environments. It provides flexible settings, a data minimization strategy, and a comprehensive operation log mechanism to help you safely operate within legal boundaries.

### 4.1 Personal Data and Privacy Protection

- The system adheres to regulations like GDPR and personal data protection laws, with the default setting not storing full facial images.
- A "de-identification" mode can be enabled, outputting only statistical results such as the number of people and age ranges.
- Customers can set up masking areas (such as counters or resting areas) and detection zones based on their needs.
- The system supports data minimization design, allowing facial modules to be turned off or replaced with lower-precision contour models.

### 4.2 Data Responsibility Boundaries

- Provides templates for Data Protection Impact Assessment (DPIA) or Data Processing Agreement (DPA) for partnerships.
- Clearly defines data responsibility boundaries: original image retention is managed by the customer, while our team processes only anonymized inference outputs.
- The system has built-in account hierarchy design: administrators can access device and system settings, while analysts can only read statistical data.

### 4.3 Security and Auditing

- Supports HTTPS/TLS secure communication, JWT authentication, and full operation log retention.
- Sensitive data, such as age range, can be tokenized or hashed.
- All access and modification actions are recorded in system logs and can be exported for archival.
- If needed, model version signatures and hash records can be provided for future reporting or verification.

### 4.4 Risk Control Mechanisms

- Offers a "data retention period" module, which automatically deletes historical data on a regular basis.
- Can integrate with a "blockchain recording module" to create immutable inference and query records (suitable for government or financial institutions).
- Supports third-party security validation (e.g., DNV, SGS) or integration with customers' internal security audits.

> **Additional Notes:**
> For high-privacy-sensitive fields (e.g., finance, healthcare, government), we recommend collaborating with legal and security departments early in the project to clearly define the scope and storage periods of the data. The "minimization" principle should be enforced at the technical level to reduce future compliance risks.

---

## 5. Conclusion

The above is the **Modular Proposal Report** for the **Crowd Flow and Demographic Analysis System**. The content clearly distinguishes between different implementation phases and demand types, while also emphasizing the importance of privacy and regulatory compliance. If you have additional specific functional requirements (e.g., VIP identification, behavior analysis, regional hotspot maps, etc.), we can further expand the system based on this modular framework, ensuring the system’s true implementation and maximum benefit. At the same time, we can flexibly adjust project budgets and cooperation models to ensure that the core goals are met even with limited resources, while leaving room for future upgrades.
