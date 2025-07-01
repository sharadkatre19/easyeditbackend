# GFPGAN API

A FastAPI service to restore facial images using Tencent's GFPGAN.

## Usage

### Run locally

1. Clone the repo
2. Download the model: `GFPGANv1.4.pth` into `models/`
   - https://github.com/TencentARC/GFPGAN#model-zoo
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
4. Start Virticual environment
```bash
source new_venv/bin/activate 

5. Run App
```bash
uvicorn app.main:app --reload