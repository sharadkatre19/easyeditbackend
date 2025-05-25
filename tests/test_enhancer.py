import pytest
from app.enhancer import load_model, enhance_image_bytes
from PIL import Image
import io

@pytest.fixture(scope="module")
def model():
    return load_model()

def test_enhance_image_bytes(model):
    # Create a dummy image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    result = enhance_image_bytes(model, buf.read())
    assert isinstance(result, io.BytesIO)
    assert result.getbuffer().nbytes > 0
