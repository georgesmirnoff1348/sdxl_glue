from configs import factortimeinference
import random, gc, torch
from diffusers import (AutoencoderKL, 
                        StableDiffusionXLInpaintPipeline, 
                        ControlNetModel,
                        StableDiffusionXLControlNetInpaintPipeline
                        )
from composer import Composer, CompositionResult
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from configs import FactorInferenceParameters, FactorPrompts
from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod


def select_device_and_dtype(preferred: Optional[str] = None):
    """Choose an available inference backend and a compatible model dtype."""
    supported = {"mps", "cuda", "cpu"}
    if preferred is not None and preferred not in supported:
        raise ValueError(f"Unsupported device {preferred!r}; expected one of {sorted(supported)}")

    available = []
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    # PyTorch's CPU backend is always available and uses float32 below.
    available.append("cpu")

    if preferred is not None:
        if preferred not in available:
            raise RuntimeError(f"Requested device {preferred!r} is unavailable")
        device = preferred
    else:
        device = next(
            candidate for candidate in ("mps", "cuda", "cpu") if candidate in available
        )
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    return device, dtype


class FactorInpainter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def inpaint_image(
        self,
        prompts: FactorPrompts,
        config: FactorInferenceParameters,
        composition: Optional[CompositionResult] = None,
        background: Optional[Image.Image] = None,
        figure: Optional[Image.Image] = None,
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        pass

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def unload(self) -> None:
        "You need to free your memory because diffusors are too heavy"
        print("--- СИСТЕМА ФАКТОР: НАЧАТО ИЗВЛЕЧЕНИЕ МОДЕЛИ ИЗ ОПЕРАТИВНОЙ ПАМЯТИ ---")

        if hasattr(self, "pipe"):
            del self.pipe

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "mps":
            torch.mps.empty_cache()

        print(
            "--- СИСТЕМА ФАКТОР: ОПЕРАТИВНАЯ ПАМЯТЬ УСПЕШНО ОСВОБОЖДЕНА ---"
        )
    


class FactorComposerInpainter(FactorInpainter):
    def __init__(
        self, 
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.vae_model_id = "madebyollin/sdxl-vae-fp16-fix"
        self.device, self.dtype = select_device_and_dtype(device)
        self._default_composer = Composer(verbose=False)

        print("[Inpainter] Загрузка VAE...")
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"[Inpainter] Загрузка основной модели {model_id}...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config( 
                self.pipe.scheduler.config,
                use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
                )
        self.pipe.scheduler.algorithm_type = "dpmsolver++"
        

        print(f"[Inpainter] Перевод моделей на {self.device}...")
        self.pipe.to(self.device)

        # Оставляем ТОЛЬКО slicing
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()

    @factortimeinference
    def inpaint_image(
        self,
        prompts: FactorPrompts,
        config: FactorInferenceParameters,
        composition: Optional[CompositionResult] = None,
        background: Optional[Image.Image] = None,
        figure: Optional[Image.Image] = None,
        save_path: Optional[Path] = None
        ) -> Image.Image:

        prompts_dict = prompts._as_dict()
        config_dict = config._as_dict()

        actual_seed = config_dict.pop("seed", None)

        # Implementing seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)
        self.last_seed = actual_seed

        print(f"--- СИСТЕМА ФАКТОР: ИСПОЛЬЗУЕТСЯ ЯДРО СЛУЧАЙНОГО ЧИСЛА: {actual_seed} ---")
        config_dict["generator"] = torch.Generator(device="cpu").manual_seed(actual_seed)


        if composition is not None:
            collage = composition.collage
            mask = composition.mask
        elif background is not None and figure is not None:
            comp_res = self._default_composer.compose(background, figure)
            collage = comp_res.collage
            mask = comp_res.mask
        else:
            raise ValueError("Передайте либо результат работы Composer, либо пару 'background' и 'figure'!")


        if self.device == "mps":
            torch.mps.empty_cache()

        print(f"[Inpainter] Генерация (Steps: {config_dict['num_inference_steps']}, Strength: {config_dict['strength']})...")

        # The composition dimensions are authoritative for this pipeline call.
        config_dict.pop("height", None)
        config_dict.pop("width", None)
        
        with torch.inference_mode():
            image_out = self.pipe(
                **config_dict,
                **prompts_dict,
                image=collage.convert("RGB"),
                mask_image=mask.convert("L"),
                height=collage.height,
                width=collage.width
            ).images[0]

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_out.save(save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image_out



class FactorCNetComposerInpainter(FactorInpainter):
    """
    Inpaint class with ControlNet (any type, don't forget 
    to bring it in in configs as control_image)
    extra-params in config: strength: float,
            controlnet_conditioning_scale: float,
            control_guidance_end: float,
    """
    def __init__(
        self,
        base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_model_id: str = "xinsir/controlnet-tile-sdxl-1.0",
        vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
        device: Optional[str] = None,
    ):
        self.base_model_id = base_model_id
        self.controlnet_model_id = controlnet_model_id
        self.vae_model_id = vae_model_id
        self.device, self.dtype = select_device_and_dtype(device)
        self._default_composer = Composer(verbose=False)
        print(f"--- СИСТЕМА ЦЕНЗОР: ПЕРЕХОД НА ЭВМ {self.device.upper()}...")

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ВАРИАЦИОННОГО АВТОКОДЕРА... ---")
        vae = AutoencoderKL.from_pretrained(
            vae_model_id, 
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА КОНТРОЛЬНОЙ СЕТИ {controlnet_model_id.upper()}... ---")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ОСНОВНОЙ МОДЕЛИ {base_model_id.upper()} С КОНТРОЛИРУЮЩИМИ СЕТЯМИ ---")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        self.pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config( 
                self.pipe.scheduler.config,
                use_karras_sigmas = True 
                )
        self.pipe.scheduler.algorithm_type = "dpmsolver++"

        # only slicing
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()
        print(f"--- СИСТЕМА ЦЕНЗОР: ИНИЦИАЛИЗАЦИЯ КОНВЕЙЕРА ДОРИСОВКИ НА {self.device.upper()} ЗАВЕРШЕНА ---")

    @factortimeinference
    def inpaint_image(
        self,
        prompts: FactorPrompts,
        config: FactorInferenceParameters,
        composition: Optional[CompositionResult] = None,
        background: Optional[Image.Image] = None,
        figure: Optional[Image.Image] = None,
        save_path: Optional[Path] = None
        ) -> Image.Image:

        prompts_dict = prompts._as_dict()
        config_dict = config._as_dict()
        
        actual_seed = config_dict.pop("seed", None)

        # Implementing seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)
        self.last_seed = actual_seed

        print(f"--- СИСТЕМА ФАКТОР: ИСПОЛЬЗУЕТСЯ ЯДРО СЛУЧАЙНОГО ЧИСЛА: {actual_seed} ---")
        config_dict["generator"] = torch.Generator(device="cpu").manual_seed(actual_seed)


        if composition is not None:
            collage = composition.collage
            mask = composition.mask
        elif background is not None and figure is not None:
            comp_res = self._default_composer.compose(background, figure)
            collage = comp_res.collage
            mask = comp_res.mask
        else:
            raise ValueError("Передайте либо результат работы Composer, либо пару 'background' и 'figure'!")


        if self.device == "mps":
            torch.mps.empty_cache()

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
        print(f"--- СИСТЕМА ЦЕНЗОР: ИСПОЛЬЗУЮТСЯ ДАННЫЕ: {prompts_dict['prompt'].upper()} ---")

        control_image = config_dict.pop("control_image", collage)
        config_dict.pop("height", None)
        config_dict.pop("width", None)
        image_inpainted = self.pipe(
            **prompts_dict,
            **config_dict,
            image=collage.convert("RGB"),
            mask_image=mask,
            control_image=control_image,
            height=collage.height,
            width=collage.width,
        ).images[0]

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_inpainted.save(save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image_inpainted
