import os
import torch
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
import sys
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream, style_transfer, super_resolution, inpainting
import torch.nn.functional as F
import random
import torchvision.transforms as T
import numpy as np
import requests
from PIL import Image
import torch
import re
import requests
from PIL import Image
import torch
import http.client
import json
import base64
import requests
from huggingface_hub import login
from potassium import Potassium, Request, Response
from urllib.request import urlretrieve

app = Potassium("my_app")

@app.init
def init():
    login('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')

    device = "cuda:0"
    if_I = IFStageI("IF-I-XL-v1.0", device=device)
    if_II = IFStageII("IF-II-L-v1.0", device=device)
    if_III = StableStageIII("stable-diffusion-x4-upscaler", device=device)
    t5 = T5Embedder(device=device)

    return {
        "if_I": if_I,
        "if_II": if_II,
        "if_III": if_III,
        "t5": t5
    }

@app.handler()
def handler(context: dict, request: Request) -> Response:
    # Extract the models from the context
    if_I = context.get("if_I")
    if_II = context.get("if_II")
    if_III = context.get("if_III")
    t5 = context.get("t5")

    # Extract the inputs from the request
    inputs = request.json
    original_image_url = inputs["original_image"]
    prompt = inputs["prompt"]

    # Download the image from the URL
    urlretrieve(original_image_url, "/tmp/original_image.png")
    original_image = Image.open("/tmp/original_image.png").convert("RGB")

    seed = 2

    # Generate the style transferred image
    result = style_transfer(
        t5=t5,
        if_I=if_I,
        if_II=if_II,
        if_III=if_III,
        disable_watermark=True,
        support_pil_img=original_image,
        prompt=[prompt] * num_outputs,
        seed=seed,
        if_I_kwargs={
            "guidance_scale": guidance_scale,
            "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
            "support_noise_less_qsample_steps": 25,
        },
        if_II_kwargs={
            "guidance_scale": 4.0,
            "sample_timestep_respacing": "smart50",
            "support_noise_less_qsample_steps": 25,
        },
        if_III_kwargs={
            "guidance_scale": 9.0,
            "noise_level": 20,
            "sample_timestep_respacing": "75",
        },
    )

    # Save the generated images to temporary files and collect their paths
    paths = []
    for n, image in enumerate(result["III"]):
        image.save(f"/tmp/out-{n}.png")
        paths.append(f"/tmp/out-{n}.png")

    # Return the paths of the generated images
    return Response(json={"outputs": paths}, status=200)
