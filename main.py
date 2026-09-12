from configs import FactorPrompts, FactorInferenceParameters
from inpainters import FactorCNetComposerInpainter
from composer import Composer
from PIL import Image
from pathlib import Path
from cutter import BirefNetCutter

prompts = FactorPrompts(prompt="A realistic photo of a beautiful young woman in front of a soviet factory",
                        negative_prompt="3d render, anime, collage")
params = FactorInferenceParameters(
    extra={"strength": 0.8,
            "controlnet_conditioning_scale": 0.8,
            "control_guidance_end": 0.8
            })

cutter = BirefNetCutter()
cutter.remove_background(image= Image.open("comrade_6.png"),
                         save_path=Path("sporous.png"))

with Composer(verbose=False) as compo:
    composition = compo.compose(background=Image.open("factory.png"),
                  figure=Image.open("sporous.png"),
                  scale=1)

with FactorCNetComposerInpainter() as inpy:
    inpy.inpaint_image(prompts=prompts,
                            config=params, 
                            composition=composition,
                            save_path=Path("outCNet.png"))