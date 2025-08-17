from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from .database.mongo import init_mongo
from .routes import users

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


init_mongo(app)
app.include_router(users.router)

