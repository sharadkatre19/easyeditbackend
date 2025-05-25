from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from app.enhancer import load_model, enhance_image_bytes
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    print("Model loaded successfully.")


def load_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        pre_pad=0,
        half=False
    )
    return upsampler

@app.post("/enhance-image/")
async def enhance_image(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Try restarting the server."}
    image_bytes = await file.read()
    output_image = enhance_image_bytes(model, image_bytes)
    return StreamingResponse(output_image, media_type="image/png")