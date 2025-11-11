-- This script is run automatically by the postgres container on first launch.
-- It creates the database user and the database itself.
-- Tables are created by SQLAlchemy (Base.metadata.create_all) when the services start.

-- Create user and database (if they don't exist from docker-compose env vars)
-- The docker-compose setup handles this, but this is good for manual setup.

-- CREATE USER pfm_user WITH PASSWORD 'secure_password_123';
-- CREATE DATABASE pfm_database OWNER pfm_user;
-- \c pfm_database pfm_user

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- You can pre-create tables here, but it's better to let SQLAlchemy manage it
-- to keep your models.py as the source of truth.
-- If you let SQLAlchemy create them, this file can be almost empty.

-- Grant permissions
-- GRANT ALL PRIVILEGES ON DATABASE pfm_database TO pfm_user;

-- Set timezone
SET timezone = 'UTC';