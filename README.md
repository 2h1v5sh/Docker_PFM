# 💼 Personal Finance Manager (PFM) – AI-Powered Financial Management & Advisory System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-316192.svg)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📘 Overview

The **Personal Finance Manager (PFM)** is a **modular, AI-powered FinTech backend system** designed to help users **manage, analyze, and optimize** their financial lives.  
It leverages **FastAPI microservices**, **Docker containers**, and **AI-driven analytics** to provide **personalized financial insights** and **interactive advisory support** via an **AI Financial Advisor Chatbot**.

---

## 🎯 Project Goals

- ✅ Automate income, expense, and investment tracking  
- ✅ Deliver real-time analytics and financial insights  
- ✅ Provide conversational financial advisory via AI  
- ✅ Maintain high security and modular scalability  
- ✅ Enable seamless integration with modern frontends (React, Next.js, etc.)

---

## 🧩 System Architecture

      ┌──────────────────────────┐
      │        API Gateway       │
      └────────────┬─────────────┘
                   │
 ┌───────────────────────────────────────────┐
 │         Microservices (FastAPI)           │
 │───────────────────────────────────────────│
 │ Auth | User | Transaction | Budget | Chat │
 │ Invest | Debt | Analytics | Notify | AI/ML│
 └───────────────────────────────────────────┘
                   │
┌────────┬────────┬────────┬────────┐
│ PostgreSQL │ Redis │ RabbitMQ │ MinIO │
│ ClickHouse │ Milvus │           │       │
└──────────────────────────────────────────┘


Each service runs in its own container and communicates through REST APIs and RabbitMQ queues.

---

## ✨ Features

✅ JWT-based authentication & RBAC  
✅ Budgeting, spending, and investment tracking  
✅ Debt & EMI management  
✅ AI-powered financial advisor chatbot  
✅ ClickHouse-based analytics dashboard  
✅ Secure file storage via MinIO  
✅ Event-driven communication using RabbitMQ  

---

## 🧰 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Backend Framework** | FastAPI (Python) |
| **Databases** | PostgreSQL, Redis, ClickHouse |
| **Vector Database** | Milvus |
| **Message Broker** | RabbitMQ |
| **Object Storage** | MinIO (S3 Compatible) |
| **Containerization** | Docker & Docker Compose |
| **AI/ML Stack** | Gemini API, Transformers, Scikit-learn |
| **Security** | JWT, AES-256 Encryption, RBAC |
| **Monitoring** | Prometheus, Grafana |

---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/2h1v5sh/Docker_PFM.git
cd Docker_PFM
