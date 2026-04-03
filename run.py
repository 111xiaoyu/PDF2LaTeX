import uvicorn
import os


if __name__ == "__main__":
    reload_env = os.getenv("UVICORN_RELOAD", "0").strip().lower()
    use_reload = reload_env in {"1", "true", "yes", "on"}
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=use_reload)
