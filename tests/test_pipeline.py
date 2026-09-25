import contextlib
import importlib
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import composer


class _FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float16 = object()
        self.float32 = object()
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.Generator = _FakeGenerator
        self.inference_mode = contextlib.nullcontext


_fake_torch = _FakeTorch()
_fake_diffusers = types.ModuleType("diffusers")
for _name in (
    "AutoencoderKL",
    "ControlNetModel",
    "DPMSolverMultistepScheduler",
    "StableDiffusionXLControlNetInpaintPipeline",
    "StableDiffusionXLInpaintPipeline",
):
    setattr(_fake_diffusers, _name, type(_name, (), {}))

_fake_onnxruntime = types.ModuleType("onnxruntime")
_fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
_fake_rembg = types.ModuleType("rembg")
_fake_rembg.new_session = lambda **kwargs: object()
_fake_rembg.remove = lambda image, session=None: image

with patch.dict(
    sys.modules,
    {
        "torch": _fake_torch,
        "diffusers": _fake_diffusers,
        "onnxruntime": _fake_onnxruntime,
        "rembg": _fake_rembg,
    },
):
    import configs
    import cutter
    import inpainters


class ConfigurationTests(unittest.TestCase):
    def test_prompt_and_inference_config_conversion(self):
        prompts = configs.FactorPrompts(
            prompt="place subject", prompt_2=None, negative_prompt="blur"
        )
        self.assertEqual(
            prompts._as_dict(),
            {"prompt": "place subject", "negative_prompt": "blur"},
        )

        defaults = configs.FactorInferenceParameters()._as_dict()
        self.assertEqual(defaults["num_inference_steps"], 20)
        self.assertEqual(defaults["guidance_scale"], 7.0)
        self.assertEqual((defaults["height"], defaults["width"]), (1024, 1024))
        self.assertIsNone(defaults["seed"])

        configured = configs.FactorInferenceParameters(
            seed=42, extra={"strength": 0.7}
        )
        self.assertEqual(configured._as_dict()["seed"], 42)
        self.assertEqual(configured._as_dict()["strength"], 0.7)


class CompositionTests(unittest.TestCase):
    def test_compose_dimensions_and_full_mask(self):
        background = Image.new("RGB", (130, 129), "navy")
        figure = Image.new("RGBA", (64, 64), (220, 20, 20, 255))

        result = composer.Composer(verbose=False).compose(background, figure)

        self.assertEqual(result.collage.size, (128, 128))
        self.assertEqual(result.collage.mode, "RGB")
        self.assertEqual(result.mask.size, result.collage.size)
        self.assertEqual(result.mask.mode, "L")
        self.assertEqual(result.mask.getextrema(), (255, 255))

    def test_edge_mask_is_localized_and_rejects_non_rgba(self):
        background = Image.new("RGB", (128, 128), "white")
        figure = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(figure).rectangle((16, 16, 47, 47), fill=(200, 20, 20, 255))
        result = composer.Composer(verbose=False, mask_inflate=10).compose(
            background, figure, position=(32, 32), mode="edge"
        )
        self.assertEqual(result.mask.getextrema()[0], 0)
        self.assertGreater(result.mask.getextrema()[1], 0)
        self.assertEqual(result.mask.getpixel((120, 120)), 0)
        with self.assertRaisesRegex(ValueError, "альфа-каналом"):
            composer.Composer(verbose=False).compose(
                background, Image.new("RGB", (64, 64)), mode="edge"
            )


class BackendSelectionTests(unittest.TestCase):
    def test_automatic_device_selection_priority_and_dtype(self):
        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=True),
            patch.object(_fake_torch.cuda, "is_available", return_value=True),
        ):
            self.assertEqual(
                inpainters.select_device_and_dtype(), ("mps", _fake_torch.float16)
            )

        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=False),
            patch.object(_fake_torch.cuda, "is_available", return_value=True),
        ):
            self.assertEqual(
                inpainters.select_device_and_dtype(), ("cuda", _fake_torch.float16)
            )

        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=False),
            patch.object(_fake_torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(
                inpainters.select_device_and_dtype(), ("cpu", _fake_torch.float32)
            )

    def test_explicit_unavailable_accelerators_raise(self):
        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=True),
            patch.object(_fake_torch.cuda, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "cuda.*unavailable"):
                inpainters.select_device_and_dtype("cuda")

        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=False),
            patch.object(_fake_torch.cuda, "is_available", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "mps.*unavailable"):
                inpainters.select_device_and_dtype("mps")

    def test_explicit_cpu_is_selected_when_accelerator_is_available(self):
        with (
            patch.object(_fake_torch.backends.mps, "is_available", return_value=True),
            patch.object(_fake_torch.cuda, "is_available", return_value=True),
        ):
            self.assertEqual(
                inpainters.select_device_and_dtype("cpu"),
                ("cpu", _fake_torch.float32),
            )

    def test_unsupported_explicit_device_raises_value_error(self):
        with self.assertRaises(ValueError):
            inpainters.select_device_and_dtype("tpu")

    def test_onnx_providers_only_include_available_compatible_providers(self):
        providers = cutter.select_onnx_providers(
            ["CPUExecutionProvider", "CUDAExecutionProvider", "MPSExecutionProvider"]
        )
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(
            cutter.select_onnx_providers(
                [
                    "CPUExecutionProvider",
                    "CUDAExecutionProvider",
                    "TensorrtExecutionProvider",
                ]
            ),
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.assertEqual(
            cutter.select_onnx_providers(["CPUExecutionProvider"]),
            ["CPUExecutionProvider"],
        )
        with self.assertRaises(RuntimeError):
            cutter.select_onnx_providers(["MPSExecutionProvider"])

    def test_cutter_retries_cpu_after_accelerator_session_failure(self):
        with (
            patch.object(
                cutter.onnxruntime,
                "get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch.object(
                cutter,
                "new_session",
                side_effect=[RuntimeError("accelerator unavailable"), "cpu-session"],
            ) as new_session,
        ):
            instance = cutter.Cutter()

        self.assertEqual(instance.session, "cpu-session")
        self.assertEqual(new_session.call_args_list[0].kwargs["providers"], [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ])
        self.assertEqual(new_session.call_args_list[1].kwargs["providers"], [
            "CPUExecutionProvider",
        ])

    def test_cutter_preserves_accelerator_error_if_cpu_retry_fails(self):
        accelerator_error = RuntimeError("accelerator initialization failed")
        cpu_error = RuntimeError("CPU initialization failed")
        with (
            patch.object(
                cutter.onnxruntime,
                "get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch.object(
                cutter,
                "new_session",
                side_effect=[accelerator_error, cpu_error],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU initialization failed") as raised:
                cutter.Cutter()

        self.assertIs(raised.exception.__cause__, accelerator_error)


class InferenceContractTests(unittest.TestCase):
    @staticmethod
    def _fake_inpainter(inpainter_type):
        inpainter = inpainter_type.__new__(inpainter_type)
        inpainter.device = "cpu"
        inpainter.dtype = _fake_torch.float32
        inpainter._default_composer = composer.Composer(verbose=False)

        class FakePipeline:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

        inpainter.pipe = FakePipeline()
        return inpainter

    def test_standard_inpainter_passes_composition_dimensions_once(self):
        inpainter = self._fake_inpainter(inpainters.FactorComposerInpainter)
        composition = composer.CompositionResult(
            Image.new("RGB", (64, 128)), Image.new("L", (64, 128), 255)
        )
        output = inpainter.inpaint_image(
            configs.FactorPrompts(prompt="test"),
            configs.FactorInferenceParameters(
                seed=321,
                height=999,
                width=999,
                extra={"strength": 0.6},
            ),
            composition=composition,
        )
        kwargs = inpainter.pipe.kwargs
        self.assertEqual(output.size, (64, 64))
        self.assertEqual((kwargs["height"], kwargs["width"]), (128, 64))
        self.assertEqual(kwargs["image"].mode, "RGB")
        self.assertEqual(kwargs["mask_image"].mode, "L")
        self.assertEqual(kwargs["generator"].seed, 321)
        self.assertEqual(kwargs["strength"], 0.6)
        self.assertEqual(inpainter.last_seed, 321)

    def test_controlnet_inpainter_preserves_control_image_contract(self):
        inpainter = self._fake_inpainter(inpainters.FactorCNetComposerInpainter)
        composition = composer.CompositionResult(
            Image.new("RGB", (96, 128)), Image.new("L", (96, 128), 255)
        )
        inpainter.inpaint_image(
            configs.FactorPrompts(prompt="control test"),
            configs.FactorInferenceParameters(
                seed=17,
                height=999,
                width=999,
                extra={"strength": 0.8, "controlnet_conditioning_scale": 0.5},
            ),
            composition=composition,
        )
        kwargs = inpainter.pipe.kwargs
        self.assertEqual((kwargs["height"], kwargs["width"]), (128, 96))
        self.assertEqual(sum(key == "height" for key in kwargs), 1)
        self.assertEqual(sum(key == "width" for key in kwargs), 1)
        self.assertEqual(kwargs["control_image"].mode, "RGB")
        self.assertEqual(kwargs["mask_image"].mode, "L")
        self.assertEqual(kwargs["generator"].seed, 17)
        self.assertEqual(kwargs["controlnet_conditioning_scale"], 0.5)
        self.assertEqual(inpainter.last_seed, 17)

    def test_importing_main_does_not_run_demo(self):
        mocked_modules = {
            "torch": _fake_torch,
            "diffusers": _fake_diffusers,
            "onnxruntime": _fake_onnxruntime,
            "rembg": _fake_rembg,
        }
        with patch.dict(sys.modules, mocked_modules):
            with patch.object(_fake_rembg, "new_session") as new_session:
                module = importlib.import_module("main")
                self.assertTrue(callable(module.main))
                new_session.assert_not_called()

    def test_main_reads_providers_from_rembg_inner_session_or_outer_session(self):
        mocked_modules = {
            "torch": _fake_torch,
            "diffusers": _fake_diffusers,
            "onnxruntime": _fake_onnxruntime,
            "rembg": _fake_rembg,
        }
        with patch.dict(sys.modules, mocked_modules):
            module = importlib.import_module("main")

        inner_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cutter_with_inner = types.SimpleNamespace(
            session=types.SimpleNamespace(
                inner_session=types.SimpleNamespace(
                    get_providers=lambda: inner_providers
                ),
                get_providers=lambda: ["outer-provider"],
            )
        )
        self.assertEqual(
            module.segmentation_onnx_providers(cutter_with_inner), inner_providers
        )
        outer_providers = ["CPUExecutionProvider"]
        cutter_with_outer = types.SimpleNamespace(
            session=types.SimpleNamespace(get_providers=lambda: outer_providers)
        )
        self.assertEqual(
            module.segmentation_onnx_providers(cutter_with_outer), outer_providers
        )


if __name__ == "__main__":
    unittest.main()
