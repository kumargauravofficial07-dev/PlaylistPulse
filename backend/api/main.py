from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.analytics import router as analytics_router
from api.routes.all_results import router as results_router
from api.routes.tracks import router as tracks_router

app = FastAPI(
    title="PlaylistPulse API",
    description="Spotify Track Intelligence ML API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "PlaylistPulse API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(analytics_router, prefix="/api", tags=["Analytics"])
app.include_router(results_router)
app.include_router(tracks_router)
