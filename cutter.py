from PIL import Image
import onnxruntime
from rembg import new_session, remove
from pathlib import Path
from typing import Optional
from abc import abstractmethod, ABC
from configs import factortimeinference


_ACCELERATOR_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
)


def select_onnx_providers(available_providers):
    """Return installed, supported accelerators first and CPU as fallback."""
    available = set(available_providers)
    providers = [provider for provider in _ACCELERATOR_PROVIDERS if provider in available]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError("ONNX Runtime reports no supported execution providers")
    return providers


class FactorCutter(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def remove_background(self, image: Image.Image, save_path: Optional[Path] = "output.png") -> Image.Image:
        pass


class Cutter (FactorCutter):
    def __init__(self, model_name: str = "u2net"):
        providers = select_onnx_providers(onnxruntime.get_available_providers())
        try:
            self.session = new_session(model_name=model_name, providers=providers)
        except Exception as accelerator_error:
            accelerator_providers = [
                provider for provider in providers if provider != "CPUExecutionProvider"
            ]
            if not accelerator_providers or "CPUExecutionProvider" not in providers:
                raise
            print("[Cutter] Accelerator provider initialization failed; retrying with CPU.")
            try:
                self.session = new_session(
                    model_name=model_name, providers=["CPUExecutionProvider"]
                )
            except Exception as cpu_error:
                raise RuntimeError(
                    "ONNX session initialization failed with accelerator providers "
                    f"{accelerator_providers!r} ({accelerator_error!r}); CPU retry also "
                    f"failed ({cpu_error!r})"
                ) from accelerator_error

    @factortimeinference
    def remove_background(
            self, 
            image: Image.Image, 
            save_path: Optional[Path] = None):

        output_image = remove(image, session=self.session)

        if save_path is not None:
            output_image.save(save_path)

        return output_image


# Ярлык для U2net
class U2netCutter(Cutter):
    def __init__(self):
        # Прокидываем название модели в конструктор BaseRembgCutter через super()
        super().__init__(model_name="u2net")


# Ярлык для BiRefNet
class BirefNetCutter(Cutter):
    def __init__(self):
        # То же самое для birefnet
        super().__init__(model_name="birefnet-general")
