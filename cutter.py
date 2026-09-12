from PIL import Image
from rembg import new_session, remove
from pathlib import Path
from typing import Optional
from abc import abstractmethod, ABC
from functools import wraps
import time
from configs import factortimeinference

class FactorCutter(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def remove_background(self, image: Image.Image, save_path: Optional[Path] = "output.png") -> Image.Image:
        pass


class Cutter (FactorCutter):
    def __init__(self, model_name: str = "u2net"):
        providers = ["CUDAExecutionProvider", "MPSExecutionProvider", "CPUExecutionProvider"]
        self.session = new_session(model_name=model_name, providers=providers)

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

