
print("STARTUP CHECK 1: entering app.main", flush=True)

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

print("STARTUP CHECK 2: FastAPI imports complete", flush=True)

from app.api.routes import router
print("STARTUP CHECK 3: routes imported", flush=True)

from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_db

print("STARTUP CHECK 4: application imports complete", flush=True)

configure_logging()
settings = get_settings()

print("STARTUP CHECK 5: initializing audit database", flush=True)
init_db()
print("STARTUP CHECK 6: audit database initialized", flush=True)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

print("STARTUP CHECK 7: FastAPI app configured", flush=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.app_name},
    )
