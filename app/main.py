from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from app.gfpgan_infer import restore_image  # <--- This should now work
import uuid
import os

app = FastAPI()

@app.post("/restore")
async def restore_face(file: UploadFile = File(...)):
    contents = await file.read()
    temp_input = f"input_{uuid.uuid4().hex}.png"
    original_filename = file.filename
    restored_filename = f"restored_{original_filename}"
    temp_output = f"output_{uuid.uuid4().hex}.png"

    with open(temp_input, "wb") as f:
        f.write(contents)

    restore_image(input_path=temp_input, output_path=temp_output)

    def cleanup_files():
        os.remove(temp_input)
        os.remove(temp_output)

    return FileResponse(
        path=temp_output,
        media_type="image/png",
        filename=restored_filename,
        background=BackgroundTask(cleanup_files)
    )
