from fastapi import FastAPI


app = FastAPI(title="Project October API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
