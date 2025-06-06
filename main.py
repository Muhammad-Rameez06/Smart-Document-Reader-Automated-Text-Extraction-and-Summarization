# main.py
# FastAPI for OCR
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model import extract_text_from_image_bytes

app = FastAPI()

@app.post("/extract/")
async def extract_text(file: UploadFile = File(...)):
    contents = await file.read()
    result = extract_text_from_image_bytes(contents)
    return JSONResponse(content={"extracted_text": result})
