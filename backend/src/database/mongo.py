import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, Request

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "lumicare")

client: AsyncIOMotorClient | None = None

def init_mongo(app: FastAPI):
    @app.on_event("startup")
    async def startup():
        global client
        client = AsyncIOMotorClient(
            MONGO_URL,
            tls=True,
            tlsAllowInvalidCertificates=False,
        )
        app.state.db = client[MONGO_DB_NAME]
        print("✅ MongoDB connected:", MONGO_DB_NAME)

    @app.on_event("shutdown")
    async def shutdown():
        global client
        if client:
            client.close()
        print("🔌 MongoDB connection closed")

def get_db(request : Request):
    return request.app.state.db