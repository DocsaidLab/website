# Let's Encrypt

## What is Let's Encrypt?

:::info
Readers who are already familiar with Let's Encrypt can skip this section.
:::

Let's Encrypt is a free, open, and automated SSL/TLS certificate authority (CA) launched by the Internet Security Research Group (ISRG). It officially entered public beta in December 2015 and quickly grew to become one of the largest certificate issuers globally.

Its core goal is to improve website security, make HTTPS encryption more accessible, and lower the technical and cost barriers for certificate management. Let's Encrypt only offers Domain Validation (DV) type SSL/TLS certificates, and it automates issuance and renewal using the ACME (Automatic Certificate Management Environment) protocol, allowing website administrators to avoid manually handling complex certificate signing and renewal processes.

Let's Encrypt has garnered support from major tech companies, including Mozilla, Google, Cisco, Akamai, the Electronic Frontier Foundation (EFF), and many others. It has become an important driver of HTTPS adoption.

:::tip
A **Domain Validation (DV)** SSL/TLS certificate is a basic encryption certificate that only verifies a website's control over its domain, without including business or personal identity information. Its primary purpose is to ensure that communications between the website and visitors are encrypted, preventing traffic interception or tampering.

DV certificates are suitable for personal websites, blogs, small businesses, or any application that requires basic HTTPS encryption. If the site doesn't need to display organizational information (as provided by OV or EV certificates), a DV certificate is a quick and simple choice.
:::

## Features of Let's Encrypt

- **1. Free**

  Traditional SSL/TLS certificates often require payment, and the price varies depending on the certificate type (DV, OV, EV) and trust level. Let's Encrypt is completely free, enabling businesses, personal websites, and developers to obtain trusted SSL/TLS certificates at no cost, significantly lowering the barrier to deploying HTTPS.

---

- **2. Automated**

  Let's Encrypt uses the ACME (Automatic Certificate Management Environment) protocol to automate the process of certificate issuance, verification, signing, installation, and renewal. By using an ACME client (such as Certbot, acme.sh, etc.), website administrators do not need to manually submit a CSR (Certificate Signing Request) or negotiate with the CA, greatly simplifying certificate management.

---

- **3. Open and Transparent**

  Let's Encrypt operates with full openness and transparency:

  - **Open Source**: The core software, ACME protocol, verification mechanisms, etc., are open source, allowing anyone to review and improve them.
  - **Public Logs**: All issued certificates are recorded in Certificate Transparency (CT) public logs, which are searchable and auditable, preventing malicious abuse or unauthorized certificates.

---

- **4. Wide Support and Compatibility**

  Let's Encrypt certificates are trusted by all major browsers and operating systems, including:

  - **Desktop**: Chrome, Firefox, Safari, Microsoft Edge
  - **Mobile**: Android (7.1.1 and above), iOS
  - **Servers/Operating Systems**: Linux (Ubuntu, Debian, CentOS, RHEL, etc.), Windows Server, macOS

  :::warning
  Very old versions of Android (below 7.1.1) may not trust the default Let's Encrypt root certificate ISRG Root X1, but compatibility can be maintained by enabling the trust for the legacy root DST Root CA X3.
  :::

---

- **5. Validity Period of 90 Days**

  Let's Encrypt certificates have a validity period of 90 days (traditional commercial CA certificates usually last 1 to 2 years). The advantages of this design include:

  - **Enhanced Security**: Even if a certificate's private key is compromised, the short validity period reduces the risk.
  - **Encourages Automated Renewal**: It is recommended to use automated renewal mechanisms like `certbot renew` or scheduled ACME clients to avoid the risks of manual updates.
  - **In line with modern best practices**: Many browsers and security agencies have pushed for shorter certificate validity periods to enhance security and flexibility.

## How Let's Encrypt Works

Let's Encrypt uses the ACME (Automatic Certificate Management Environment) protocol to automatically issue and renew SSL/TLS certificates without human intervention, enabling website administrators to easily deploy HTTPS.

The ACME process is as follows:

1. **The user installs an ACME client** (such as `Certbot`, `acme.sh`, or other ACME-compatible tools).
2. **The ACME client sends a certificate request to Let's Encrypt**, requesting to encrypt a specific domain name (e.g., `example.com`).
3. **Let's Encrypt performs domain validation**, confirming that the user controls the domain. The validation methods include:
   - **HTTP Validation (HTTP-01)**: Let's Encrypt provides a random verification file, and the ACME client must place it in the `.well-known/acme-challenge/` directory of the website server. Let's Encrypt's server will try to access this file to verify domain control.
   - **DNS Validation (DNS-01)**: Used for wildcard certificates or when HTTP validation is not possible. The user needs to add a specific TXT record to the DNS settings, and Let's Encrypt will query the DNS record to complete the validation.
   - **TLS-ALPN Validation (TLS-ALPN-01)**: Suitable for environments that do not use an HTTP server, such as applications that only provide TLS services. This method creates a temporary TLS certificate on port 443 of the server to complete the validation.
4. **After successful validation, Let's Encrypt issues the SSL/TLS certificate**, and the certificate will be stored in the `/etc/letsencrypt/live/yourdomain/` directory (depending on the ACME client).
5. **The ACME client performs automatic renewal**: Typically, the client checks every 60 days to see if a renewal is needed, ensuring that the certificate does not expire (with a 90-day validity period).

## How to Install and Use?

Let's Encrypt supports various ACME clients, with the most popular being **Certbot**, developed by the Electronic Frontier Foundation (EFF).

- **1. Install Certbot**

  On Ubuntu/Debian:

  ```bash
  sudo apt update
  sudo apt install certbot python3-certbot-nginx
  ```

  On CentOS/RHEL:

  ```bash
  sudo yum install certbot python3-certbot-nginx
  ```

- **2. Request an SSL Certificate**

  Assuming the website is running on Nginx, use the following command to request the certificate:

  ```bash
  sudo certbot --nginx -d example.com -d www.example.com
  ```

  This will automatically modify the Nginx configuration and enable HTTPS.

- **3. Set Up Automatic Renewal**

  Certbot automatically renews certificates by default, but it's recommended to manually test the renewal process:

  ```bash
  sudo certbot renew --dry-run
  ```

  If successful, the system will automatically renew certificates regularly.

## The Drawbacks

While Let's Encrypt offers free and automated SSL certificates, there are also some limitations:

- **1. Only Provides Domain Validation (DV) Certificates**

  Let's Encrypt only offers DV certificates, meaning it cannot provide OV (Organization Validation) or EV (Extended Validation) certificates. This means business names can't be displayed (some banks, e-commerce sites, or large enterprises might need EV certificates to boost trust).

  Furthermore, it’s not suitable for high-trust business websites like financial institutions or government sites, as these organizations typically require OV or EV certificates to prove their legitimacy.

---

- **2. Certificate Validity is Only 90 Days**

  Compared to traditional CA (Certificate Authorities) that offer certificates with validity periods of 1-2 years, Let's Encrypt certificates are valid for only 90 days. If the automatic renewal mechanism fails, the website may become inaccessible to users.

---

- **3. Does Not Support IP Address SSL Certificates**

  Let's Encrypt does not provide SSL certificates for IP addresses; it only supports domain-based certificates (Fully Qualified Domain Name - FQDN).

---

- **4. Relies on ACME Client and Server Configuration**

  While Let's Encrypt supports automation, users need to install and configure an ACME client (such as `Certbot`). For Windows servers or custom web service architectures, this may add to the technical complexity.

---

- **5. Lower Trust in Certain Situations**

  Because Let's Encrypt certificates can be quickly and automatically obtained, malicious websites may also use them to gain the HTTPS lock icon, leading users to mistakenly believe the site is secure. In fact, phishing sites or malicious sites may also use Let's Encrypt, lowering visitors' guard.

---

- **6. No Technical Support**

  Finally, Let's Encrypt does not offer dedicated customer support; users must rely on official documentation and community forums for solutions, which may be challenging for less tech-savvy users.

---

While Let's Encrypt lowers the barrier for SSL/TLS encryption and helps popularize HTTPS, it is not the best solution for all websites. Users should choose the appropriate SSL certificate based on their specific needs.

## Conclusion

Let's Encrypt, through its free, automated, and open approach, has contributed to the popularization of HTTPS, advancing internet security. For personal websites, blogs, and small to medium-sized businesses, it is an ideal choice for easily obtaining SSL/TLS encryption protection.

After understanding the features and operation of Let's Encrypt, we will move on to configuring HTTPS in Nginx and using Let's Encrypt certificates to secure website data transmission.
