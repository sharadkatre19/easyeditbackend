import io
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_model():
    model = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=RRDBNet(num_in_ch=3, num_out_ch=3, nf=64, nb=23,
                      gc=32, scale=4, group=1, norm_type=None,
                      act_type='leakyrelu', upsample_mode='upconv')
    )
    return model

def enhance_image_bytes(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(image)
    output, _ = model.enhance(np_img, outscale=4)
    result_image = Image.fromarray(output)
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return buf
