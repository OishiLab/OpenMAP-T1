from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/health")
def health():
    return Response(content="OK", status_code=200)
