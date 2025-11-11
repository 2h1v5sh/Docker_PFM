import redis
import aio_pika
import json
from typing import Any, Optional
import os
import httpx
from datetime import datetime

# --- Redis Client ---

class RedisClient:
    """A singleton wrapper for the Redis client."""
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD'),
                decode_responses=True # Decode from bytes to str
            )
        return cls._client

    def get(self, key: str) -> Optional[str]:
        return self.get_client().get(key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None):
        """Sets a key-value pair, serializing if dict/list."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self.get_client().set(key, value, ex=ex)
        
    def delete(self, key: str):
        self.get_client().delete(key)
    
    def exists(self, key: str) -> bool:
        return self.get_client().exists(key) > 0

# Create a single instance for import
redis_client = RedisClient()


# --- RabbitMQ Manager (Async) ---

class RabbitMQManager:
    """A wrapper for async RabbitMQ connections and operations."""
    def __init__(self):
        self.connection_url = (
            f"amqp://{os.getenv('RABBITMQ_USER', 'guest')}:"
            f"{os.getenv('RABBITMQ_PASSWORD', 'guest')}@"
            f"{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )
        self.connection = None
        self.channel = None

    async def connect(self):
        """Establishes a robust connection and channel."""
        if not self.connection or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(self.connection_url)
            self.channel = await self.connection.channel()
            print("RabbitMQ connected successfully.")

    async def close(self):
        """Closes the channel and connection."""
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        print("RabbitMQ connection closed.")

    async def publish(self, queue_name: str, message: dict):
        """Declares a queue and publishes a JSON message to it."""
        await self.connect() # Ensure connection
        
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        await self.channel.default_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key=queue_name
        )

    async def consume(self, queue_name: str, callback):
        """Consumes messages from a queue and passes them to a callback."""
        await self.connect() # Ensure connection
        
        queue = await self.channel.declare_queue(queue_name, durable=True)
        await queue.consume(callback)
        print(f"Started consuming from queue: {queue_name}")

# Create a single instance for import
rabbitmq_manager = RabbitMQManager()


# --- HTTP Client (for inter-service communication) ---

class ServiceClient:
    """
    An async HTTP client wrapper for service-to-service calls.
    (Not used in the refactored code, but good to have)
    """
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get(self, service_url: str, endpoint: str, headers: dict = None):
        url = f"{service_url}{endpoint}"
        response = await self.client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    async def post(self, service_url: str, endpoint: str, data: dict, headers: dict = None):
        url = f"{service_url}{endpoint}"
        response = await self.client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        await self.client.aclose()