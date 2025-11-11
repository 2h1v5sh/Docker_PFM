#!/bin/bash
# setup.sh - Setup script for Personal Finance Manager
# Run this script to create all necessary files and directories

set -e  # Exit on error

echo "🚀 Setting up Personal Finance Manager..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"

mkdir -p services/{auth-service,user-service,transaction-service,budget-service}
mkdir -p services/{investment-service,debt-service,analytics-service,notification-service,ai-ml-service}
mkdir -p shared
mkdir -p scripts
mkdir -p gateway
mkdir -p monitoring

echo -e "${GREEN}✓ Directories created${NC}"

# Create Dockerfile for standard services
echo -e "${BLUE}Creating Dockerfiles...${NC}"

for service in auth-service user-service transaction-service budget-service investment-service debt-service notification-service; do
  cat > services/$service/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
EOF

  # Create requirements.txt for each service
  cat > services/$service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.0
pydantic-settings==2.1.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
aio-pika==9.3.1
python-dotenv==1.0.0
httpx==0.25.2
cryptography==41.0.7
EOF
done

# AI/ML Service Dockerfile
cat > services/ai-ml-service/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models

EXPOSE 8009

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8009/health || exit 1

CMD ["python", "main.py"]
EOF

cat > services/ai-ml-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.0
python-jose[cryptography]==3.3.0
aio-pika==9.3.1
python-dotenv==1.0.0
httpx==0.25.2
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
pymilvus==2.3.4
EOF

# Analytics Service Dockerfile
cat > services/analytics-service/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8007/health || exit 1

CMD ["python", "main.py"]
EOF

cat > services/analytics-service/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.0
python-jose[cryptography]==3.3.0
aio-pika==9.3.1
python-dotenv==1.0.0
httpx==0.25.2
pandas==2.1.4
numpy==1.26.2
clickhouse-driver==0.2.6
EOF

echo -e "${GREEN}✓ Dockerfiles created${NC}"

# Create placeholder main.py files if they don't exist
echo -e "${BLUE}Creating placeholder main.py files...${NC}"

for service in auth-service user-service transaction-service budget-service investment-service debt-service analytics-service notification-service ai-ml-service; do
  if [ ! -f services/$service/main.py ]; then
    cat > services/$service/main.py << 'EOF'
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": os.getenv("SERVICE_NAME", "unknown")}

@app.get("/")
async def root():
    return {"message": "Service is running"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF
  fi
done

echo -e "${GREEN}✓ Placeholder files created${NC}"

# Create shared/__init__.py
cat > shared/__init__.py << 'EOF'
# Shared modules for Personal Finance Manager
EOF

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo -e "${BLUE}Creating .env file...${NC}"
  cat > .env << 'EOF'
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=pfm_user
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=pfm_database

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_secure_pass

# ClickHouse Configuration
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=9000
CLICKHOUSE_DB=analytics

# Milvus Configuration
MILVUS_HOST=milvus
MILVUS_PORT=19530

# RabbitMQ Configuration
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=pfm_queue
RABBITMQ_PASSWORD=queue_password

# MinIO/S3 Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=pfm-documents

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Service Ports
AUTH_SERVICE_PORT=8001
USER_SERVICE_PORT=8002
TRANSACTION_SERVICE_PORT=8003
BUDGET_SERVICE_PORT=8004
INVESTMENT_SERVICE_PORT=8005
DEBT_SERVICE_PORT=8006
ANALYTICS_SERVICE_PORT=8007
NOTIFICATION_SERVICE_PORT=8008
AI_ML_SERVICE_PORT=8009

# Encryption (generate new key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
ENCRYPTION_KEY=your-32-byte-encryption-key-here-change-this

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@pfm.com

# SMS Configuration (Twilio - Optional)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
EOF
  echo -e "${GREEN}✓ .env file created${NC}"
fi

# Create init_db.sql if it doesn't exist
if [ ! -f scripts/init_db.sql ]; then
  echo -e "${BLUE}Creating init_db.sql...${NC}"
  cat > scripts/init_db.sql << 'EOF'
-- PostgreSQL Initialization Script
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, date DESC) WHERE user_id IS NOT NULL;

-- Success message
SELECT 'Database initialized successfully' AS status;
EOF
  echo -e "${GREEN}✓ init_db.sql created${NC}"
fi

# Create gateway config placeholder
if [ ! -f gateway/kong.yml ]; then
  cat > gateway/kong.yml << 'EOF'
_format_version: "3.0"
services: []
EOF
fi

# Create monitoring config placeholder
if [ ! -f monitoring/prometheus.yml ]; then
  cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'pfm-services'
    static_configs:
      - targets: ['localhost:9090']
EOF
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your credentials"
echo "2. Add your service code to services/*/main.py"
echo "3. Run: docker-compose up --build -d"
echo ""
echo "⚠️  IMPORTANT: Change these in .env:"
echo "   - JWT_SECRET_KEY"
echo "   - ENCRYPTION_KEY"
echo "   - POSTGRES_PASSWORD"
echo "   - REDIS_PASSWORD"
echo ""