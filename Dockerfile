FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

RUN pip install --user \
    'ultralytics==8.0.183' \
    pyyaml \
    'pillow>=9.5.0' \
    'scikit-image==0.21.0' \
    'numpy==1.23.5'