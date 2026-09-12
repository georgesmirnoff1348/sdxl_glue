from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict
from collections.abc import Mapping
import torch
from functools import wraps
import time


@dataclass
class FactorPrompts(Mapping):
    """
    Class for easy prompts transporting
    """
    prompt: str = ""
    negative_prompt: str = ""
    # Для SDXL: второй текстовый энкодер или промпт для рефайнера
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    
    def _as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self):
        return len(self._as_dict())


@dataclass
class FactorInferenceParameters(Mapping):
    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    seed: Optional[int] = None
    height: int = 1024
    width: int = 1024
    extra: Dict[str, Any] = field(default_factory=dict)

    def _as_dict(self) -> Dict[str, Any]:
        d = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "height": self.height,
            "width": self.width,
            "seed": self.seed
        }
        d.update(self.extra)
        return d

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self):
        return len(self._as_dict())


def factortimeinference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Если декорируется метод класса, извлекаем имя класса из args[0]
        class_name = args[0].__class__.__name__ if args else ""
        algo_name = f"{class_name}.{func.__name__}"
        
        print(f"--- СИСТЕМА Ф.А.К.Т.О.Р.: ЗАПУСК АЛГОРИТМА {algo_name} ---")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        print(f"--- СИСТЕМА ФАКТОР: {algo_name} ОТРАБОТАЛА ЗА {execution_time:.4f} СЕК ---")
        print(f"--- СИСТЕМА ФАКТОР: ОТЧЕТ О ПОТРАЧЕННОМ ЭВМ-ВРЕМЕНИ НАПРАВЛЕН РУКОВОДСТВУ НИИ 112-10 ---")
        return result
    return wrapper

# Usage:
# pipe(**prompts, **params)
# where `prompts` is an instance of FactorPrompts
# and `params` is an instance of FactorInferenceParameters