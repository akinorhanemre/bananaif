from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import style_transfer
from PIL import Image
from potassium import Potassium, Request, Response
import huggingface_hub
import random
import json

app = Potassium("my_app")

@app.init
def init():
    huggingface_hub.login('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')

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
    # Extract the model from the context
    if_I = context.get("if_I")
    if_II = context.get("if_II")
    if_III = context.get("if_III")
    t5 = context.get("t5")

    # Extract the inputs from the request
    inputs = request.json
    original_image = inputs["original_image"]
    prompt = inputs["prompt"]
    negative_prompt = inputs.get("negative_prompt", "")
    style_prompt = inputs.get("style_prompt", "")
    num_outputs = inputs.get("num_outputs", 1)
    seed = inputs.get("seed", 0)
    guidance_scale = inputs.get("guidance_scale", 10.0)

    from urllib.request import urlretrieve

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
    negative_prompt = inputs.get("negative_prompt", "")
    style_prompt = inputs.get("style_prompt", "")
    num_outputs = inputs.get("num_outputs", 1)
    seed = inputs.get("seed", 0)
    guidance_scale = inputs.get("guidance_scale", 10.0)

    # Download the image from the URL
    urlretrieve(original_image_url, "/tmp/original_image.png")
    original_image = Image.open("/tmp/original_image.png").convert("RGB")

    seed = random.randint(0, 2**32 - 1) if seed == 0 else seed

    # Generate the style transferred image
    result = style_transfer(
        t5=t5,
        if_I=if_I,
        if_II=if_II,
        if_III=if_III,
        disable_watermark=True,
        support_pil_img=original_image,
        negative_prompt=[negative_prompt] * num_outputs,
        style_prompt=[style_prompt] * num_outputs,
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