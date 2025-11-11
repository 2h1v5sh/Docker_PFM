# Personal Finance Manager (PFM)

A modular, AI-powered financial management system built with FastAPI microservices and Docker.

## Overview

Personal Finance Manager is a containerized backend platform for tracking expenses, analyzing investments, and providing AI-driven financial advice through secure REST APIs.

## Features

- JWT-based authentication
- Transaction tracking (income, expenses, transfers)
- Budget management with alerts
- Investment portfolio tracking
- Debt and EMI management
- AI-powered financial advisor chatbot
- Real-time analytics dashboard
- Email/SMS notifications
- Secure file storage

## Architecture

```
                    ┌─────────────────────┐
                    │   API Gateway       │
                    │   (Port 8000)       │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Auth Service  │     │ User Service  │     │ Transaction   │
│   (8101)      │     │   (8102)      │     │   (8103)      │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Budget Service│     │ Investment    │     │ Debt Service  │
│   (8104)      │     │   (8105)      │     │   (8106)      │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐    ┌──────────────┐
│ Analytics     │     │ Notification  │    │  AI/ML       │
│   (8107)      │     │   (8108)      │    │  (8109)      │
└───────┬───────┘     └───────┬───────┘    └──────┬───────┘
        │                     │                    │
        └─────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Chat Service     │
                    │  AI Advisor       │
                    │    (8110)         │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│ PostgreSQL   │      │   Redis      │     │  RabbitMQ    │
│   (5432)     │      │   (6379)     │     │   (5672)     │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│ ClickHouse   │      │   Milvus     │     │   MinIO      │
│   (8123)     │      │  (19530)     │     │   (9000)     │
└──────────────┘      └──────────────┘     └──────────────┘
```

### Microservices

- **Auth Service** (8101) - User authentication & JWT tokens
- **User Service** (8102) - Profile management & KYC
- **Transaction Service** (8103) - Income/expense tracking
- **Budget Service** (8104) - Budget planning & alerts
- **Investment Service** (8105) - Portfolio tracking
- **Debt Service** (8106) - Loan & EMI management
- **Analytics Service** (8107) - Financial insights & reports
- **Notification Service** (8108) - Email/SMS alerts
- **AI/ML Service** (8109) - Predictions & anomaly detection
- **Chat Service** (8110) - AI financial advisor chatbot

### Data Layer

- **PostgreSQL** - Primary transactional database
- **Redis** - Caching & session management
- **ClickHouse** - Analytics & time-series data
- **Milvus** - Vector database for AI embeddings
- **RabbitMQ** - Message queue for async communication
- **MinIO** - Object storage for files

## Tech Stack

- **Backend**: FastAPI (Python)
- **Databases**: PostgreSQL, Redis, ClickHouse
- **Vector DB**: Milvus
- **Message Queue**: RabbitMQ
- **Storage**: MinIO
- **AI**: Gemini API
- **Container**: Docker & Docker Compose

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+ (for local development)

## Installation

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/2h1v5sh/Docker_PFM.git
cd Docker_PFM
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Configure environment variables in `.env`:
```env
POSTGRES_USER=pfm_user
POSTGRES_PASSWORD=your_password
REDIS_PASSWORD=your_redis_pass
RABBITMQ_USER=pfm_queue
RABBITMQ_PASSWORD=your_queue_pass
GEMINI_API_KEY=your_gemini_key
JWT_SECRET_KEY=your_jwt_secret
```

4. Start all services:
```bash
docker-compose up --build
```

5. Verify services are running:
```bash
docker-compose ps
```

### Alternative: Using Setup Scripts

The project includes two shell scripts for easy deployment:

#### 1. API Services Setup (`API.sh`)

Starts all backend microservices and infrastructure:

```bash
chmod +x API.sh
./API.sh
```

This script will:
- Start PostgreSQL, Redis, RabbitMQ, ClickHouse, Milvus, MinIO
- Launch all 10 microservices
- Initialize databases
- Set up API Gateway

#### 2. Chat Service Setup (`chat.sh`)

Starts only the AI chatbot service with required dependencies:

```bash
chmod +x chat.sh
./chat.sh
```

This script will:
- Start Chat Service (port 8110)
- Initialize Milvus vector database
- Load AI models and embeddings
- Connect to Gemini API

**Usage Examples:**
```bash
# Start all services
./API.sh

# In another terminal, start chat service separately
./chat.sh

# Or start everything together
./API.sh && ./chat.sh
```

## Usage

### Access Service Documentation

- API Gateway: http://localhost:8000
- Auth Service: http://localhost:8101/docs
- Chat Service: http://localhost:8110/docs
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- RabbitMQ Dashboard: http://localhost:15672

### Example API Calls

**Register User:**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "pass123"}'
```

**Add Transaction:**
```bash
curl -X POST http://localhost:8000/transactions/add \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"type": "expense", "amount": 1000, "category": "food"}'
```

**Chat with AI Advisor:**
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"message": "How can I save money?"}'
```

## Project Structure

```
Docker_PFM/
├── docker-compose.yml
├── .env
├── backend/
│   ├── auth_service/
│   ├── user_service/
│   ├── transaction_service/
│   ├── budget_service/
│   ├── investment_service/
│   ├── debt_service/
│   ├── analytics_service/
│   ├── notification_service/
│   ├── ai_ml_service/
│   ├── chat_service/
│   └── common/
├── scripts/
│   ├── init_db.py
│   └── setup_milvus.py
└── docker/
    ├── base.Dockerfile
    └── nginx.conf
```

## Development

### Run Individual Service
```bash
docker-compose up auth_service
```

### View Logs
```bash
docker-compose logs -f chat_service
```

### Run Database Migrations
```bash
docker-compose exec auth_service python scripts/init_db.py
```

### Stop Services
```bash
docker-compose down
```

## Configuration

Key configuration files:

- `docker-compose.yml` - Service orchestration
- `.env` - Environment variables
- `backend/common/config.py` - Application settings

## Security

- JWT authentication with refresh tokens
- AES-256 encryption for sensitive data
- RBAC (Role-Based Access Control)
- Rate limiting on API endpoints
- Secure password hashing (bcrypt)

## Monitoring

Services include health check endpoints:

```bash
curl http://localhost:8101/health
```

## License

MIT License - See LICENSE file for details

## Author

**Bhavesh R**  
GitHub: [@2h1v5sh](https://github.com/2h1v5sh)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
