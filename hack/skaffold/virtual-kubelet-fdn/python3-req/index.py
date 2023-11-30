from typing import Optional

from function.utils.reqmodel import RequestModel

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html

from function.handler import handle_head, handle_tail

app = FastAPI(
  title="Request Handler",
  description="Handles the requests for the model",
  version="0.1.0",
)

@app.get("/docs")
def custom_swagger_ui_html(req: Request):
  root_path = req.scope.get("root_path", "").rstrip("/")
  openapi_url = root_path + app.openapi_url
  return get_swagger_ui_html(
    openapi_url=openapi_url,
    title="API",
  )

@app.post("/")
@app.get("/")
def live_probe():
  return {"message": "hello-world"}

@app.post("/head", status_code=200)
def handle_model(params: RequestModel):
  """
    Handles the head of the model

    This function gets called independently of the cluster type
  """
  
  output = handle_head(params)
  print("Success:", output)
  return {"result": output}

@app.post("/tail")
def handle_second(params: RequestModel):
  """
    Handles the tail of the model

    This function gets called only if the cluster type if CLOUD and
    the EDGE did not finish on time.
  """
  output = handle_tail(params)
  return {"result": output}