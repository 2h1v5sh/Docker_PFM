from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="AI/ML Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-ml-service"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8109))
    uvicorn.run(app, host="0.0.0.0", port=port)
