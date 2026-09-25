import json
from importlib.metadata import PackageNotFoundError, version
from configs import FactorPrompts, FactorInferenceParameters
from inpainters import FactorCNetComposerInpainter
from composer import Composer
from PIL import Image
from pathlib import Path
from cutter import BirefNetCutter


def segmentation_onnx_providers(cutter):
    """Return providers from rembg's underlying ONNX session when available."""
    session = getattr(cutter, "session", None)
    inner_session = getattr(session, "inner_session", None)
    for candidate in (inner_session, session):
        get_providers = getattr(candidate, "get_providers", None)
        if callable(get_providers):
            return get_providers()
    return None


def main() -> None:
    prompts = FactorPrompts(
        prompt="A realistic photo of a beautiful young woman in front of a soviet factory",
        negative_prompt="3d render, anime, collage",
    )
    params = FactorInferenceParameters(
        extra={
            "strength": 0.8,
            "controlnet_conditioning_scale": 0.8,
            "control_guidance_end": 0.8,
        },
        seed=1729,
    )

    cutter = BirefNetCutter()
    cutter.remove_background(
        image=Image.open("portrait_source.png"),
        save_path=Path("portrait_cutout.png"),
    )

    composition = Composer(verbose=False).compose(
        background=Image.open("factory.png"),
        figure=Image.open("portrait_cutout.png"),
        scale=1,
    )

    with FactorCNetComposerInpainter() as inpy:
        inpy.inpaint_image(
            prompts=prompts,
            config=params,
            composition=composition,
            save_path=Path("outCNet.png"),
        )

        package_versions = {}
        for package in (
            "accelerate",
            "diffusers",
            "numpy",
            "onnxruntime",
            "opencv-python",
            "pillow",
            "rembg",
            "torch",
            "transformers",
        ):
            try:
                package_versions[package] = version(package)
            except PackageNotFoundError:
                package_versions[package] = None
        metadata = {
            "inputs": {
                "background": "factory.png",
                "foreground_source": "portrait_source.png",
                "cutout": "portrait_cutout.png",
            },
            "outputs": {
                "cutout": "portrait_cutout.png",
                "inpainted_image": "outCNet.png",
                "metadata": "outCNet.metadata.json",
            },
            "prompt": prompts._as_dict(),
            "seed": inpy.last_seed,
            "inference": {
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                **params.extra,
                "width": composition.collage.width,
                "height": composition.collage.height,
            },
            "composition": {"mask_mode": "full", "placement": [0, 0], "scale": 1},
            "models": {
                "segmentation": "birefnet-general",
                "base": inpy.base_model_id,
                "controlnet": inpy.controlnet_model_id,
                "vae": inpy.vae_model_id,
                "revision": None,
            },
            "execution": {
                "device": inpy.device,
                "dtype": str(inpy.dtype).removeprefix("torch."),
                "scheduler": "DPMSolverMultistepScheduler",
                "scheduler_algorithm": "dpmsolver++",
                "use_karras_sigmas": True,
                "segmentation_onnx_providers": segmentation_onnx_providers(cutter),
            },
            "dependency_versions": package_versions,
            "reproducibility_note": (
                "The explicit seed and settings support repeatable runs where the libraries "
                "permit; bitwise-identical output is not guaranteed across hardware or versions."
            ),
        }
        Path("outCNet.metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("[Demo] Saved image outCNet.png and metadata outCNet.metadata.json")


if __name__ == "__main__":
    main()
