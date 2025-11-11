from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="debt Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "debt-service"}

@app.get("/")
async def root():
    return {"message": "debt service is running"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
