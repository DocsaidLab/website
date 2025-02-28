---
sidebar_position: 1
---

# Nginx Introduction

Starting from scratch can often seem dull and uninspiring.

So, let’s first define a scenario, and from there, all the learning will be aimed at completing this scenario.

## Scenario Description

We have set up an API for a model inference service on the server. It allows us to send data via HTTP to the server and receive a response.

The current scenario is: **We need to expose this API endpoint to the outside world.**

For example, our API endpoint is `https://temp_api.example.com/test`, and we expect to be able to retrieve the response using `curl`, like this:

```bash
API_URL="https://temp_api.example.com/test"

curl -X GET $API_URL
```

The response might be a string:

```json
{
  "message": "API is running!"
}
```

:::warning
The above API endpoint is hypothetical and does not actually exist.
:::

## Prerequisites

In this scenario, we will use Let's Encrypt to obtain an SSL certificate to provide HTTPS service.

Since Let's Encrypt requires domain name resolution and does not accept IP addresses, please make sure you have a domain name available if you want to follow along.

## Learning Goals

We expect to learn the following:

1. ✅ Nginx Introduction
2. ✅ Nginx Reverse Proxy
3. ✅ Nginx HTTPS Configuration
4. ✅ Nginx Security
5. ✅ Nginx Monitoring
6. ✅ Nginx Serving Static Resources
7. [ ] Set Up Nginx Load Balancing

## References

- [**Nginx Official Documentation**](https://nginx.org/en/docs/)
