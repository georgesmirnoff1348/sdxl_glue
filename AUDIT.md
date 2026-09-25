# Repository Audit — SDXL Glue / Image Harmonization

## Executive summary

This is a compact, understandable image-harmonization prototype: it segments a foreground portrait, alpha-composites it over a background, builds an inpainting mask, then calls an SDXL inpainting pipeline augmented by Tile ControlNet. The repository includes a runnable-looking example and visual assets, but the current example has a deterministic inference-call failure: image dimensions are passed twice. Other notable issues are device selection that has no CPU fallback, model cleanup that targets the wrong attribute, incomplete alternative inpainter implementation, and mismatched dependency manifests.

The audit is based on static inspection of the repository and included images. No dependency installation or model inference was run; results and runtime behavior beyond the statically evident issues are not claimed as tested. No files other than this audit are intended to be changed.

## A. Current architecture

1. [`main.py`](main.py:8) defines an example prompt and inference parameters, extracts a portrait foreground through [`BirefNetCutter`](cutter.py:45), composites it with a background through [`Composer.compose()`](composer.py:158), and sends the resulting image and mask to [`FactorCNetComposerInpainter.inpaint_image()`](inpainters.py:213).
2. [`cutter.py`](cutter.py:18) wraps `rembg` sessions, with convenience subclasses selecting U2Net or BiRefNet model names. The example uses BiRefNet.
3. [`composer.py`](composer.py:7) resizes the background and RGBA foreground to dimensions intended to be multiples of 64, pastes the foreground onto an RGB canvas, and produces an L-mode mask. It supports several mask modes plus an optional console preview / interactive placement loop.
4. [`inpainters.py`](inpainters.py:157) loads the SDXL inpainting checkpoint, a VAE, and Tile ControlNet; it configures a DPM-Solver scheduler and invokes Diffusers. A second SDXL inpainting class without ControlNet also exists at [`FactorComposerInpainter`](inpainters.py:53).
5. [`configs.py`](configs.py:9) defines prompt and inference-parameter dataclasses plus a timing decorator. [`README.md`](README.md:4) describes the goal and shows one example; the top-level PNGs include the background, source portrait, cutout, and output.

The separation is adequate for a small experiment: composition/masks, segmentation, model inference, and example orchestration are distinct. The main architecture issue is not missing layers; it is inconsistent contracts and lifecycle behavior between the two inpainters and their shared base class.

## Findings

### P0 — The example inference call passes `height` and `width` twice

- **Concrete problem:** [`FactorInferenceParameters._as_dict()`](configs.py:42) always includes `height` and `width`. The ControlNet path expands that dictionary into the pipeline call and also supplies explicit dimensions derived from the composition at [`inpainters.py`](inpainters.py:253). The example constructs the config with its default dimensions at [`main.py`](main.py:10).
- **Why it matters:** Python raises a `TypeError` for duplicate keyword arguments before Diffusers runs, so the checked-in end-to-end example cannot complete inference as written.
- **Smallest reasonable fix:** Make the pipeline call pass each dimension exactly once. Prefer deriving width and height from the actual collage and removing those keys from the expanded config; add a mocked-pipeline test for the call arguments.

### P1 — ControlNet device selection fails on CPU-only systems

- **Concrete problem:** [`FactorCNetComposerInpainter.__init__()`](inpainters.py:165) selects MPS when available and otherwise unconditionally selects CUDA. It does not check CUDA availability and never selects CPU. The class also always requests float16 weights.
- **Why it matters:** On a machine without MPS and without a working CUDA device, model transfer to `cuda` fails. This undermines portability and means the code does not implement the CPU option implied by the requested device audit; CPU float16 execution also cannot be assumed to work for this pipeline.
- **Smallest reasonable fix:** Select among MPS, CUDA, and CPU based on availability, choose a compatible dtype for the selected device, and fail early with an actionable message if the chosen backend cannot support the requested model/resolution. Keep backend selection in the existing constructor rather than introducing a device-management framework.

### P1 — Context-manager cleanup does not release the loaded diffusion pipeline

- **Concrete problem:** Both inpainter implementations store the model as `self.pipe` ([`inpainters.py`](inpainters.py:72), [`inpainters.py`](inpainters.py:192)), while [`FactorInpainter.unload()`](inpainters.py:32) checks for and deletes `self.pipeline`. That attribute is not set, so leaving the context does not drop the strong reference to the loaded pipeline. Cache-clearing calls cannot free weights still referenced by the object.
- **Why it matters:** The context-manager API promises cleanup but can retain many gigabytes of model memory for as long as the inpainter object remains referenced. In [`main.py`](main.py:25), the object remains bound after leaving the `with` block.
- **Smallest reasonable fix:** Delete or clear the actual `pipe` attribute in `unload()` before collecting garbage and clearing the selected backend cache. Avoid claiming that cache clearing alone releases live model objects.

### P1 — The non-ControlNet inpainter cannot be instantiated through its declared interface

- **Concrete problem:** [`FactorInpainter`](inpainters.py:17) declares `inpaint_image()` abstract, but [`FactorComposerInpainter`](inpainters.py:53) implements a method named `inpaint()` instead ([`inpainters.py`](inpainters.py:93)).
- **Why it matters:** The class remains abstract under Python's ABC rules and cannot be instantiated, despite containing a complete-looking implementation. This is a broken alternative public path and makes the base interface misleading.
- **Smallest reasonable fix:** Rename the implementation to the interface method, or change the base contract and both subclasses to use one consistent method name. If this alternative is intentionally unsupported, remove it rather than retaining an unusable implementation.

### P1 — The two dependency manifests do not describe the same install

- **Concrete problem:** [`pyproject.toml`](pyproject.toml:7) declares `rembg` and `onnxruntime`, which are needed by [`cutter.py`](cutter.py:1), but [`requirements.txt`](requirements.txt:1) omits both. The README does not state which manifest is authoritative or provide an install procedure.
- **Why it matters:** Installing from the requirements file alone can leave the segmentation import unavailable; following the repository therefore has no unambiguous, reliable setup path. The lock file corresponds to the project metadata, not to the incomplete requirements list.
- **Smallest reasonable fix:** Choose one source of truth. For example, document the existing project/lock workflow and generate any pip requirements export from it, or add the omitted runtime requirements and keep the two files synchronized. Add a clean-environment install check.

### P1 — The demo is not reproducible from the README

- **Concrete problem:** [`README.md`](README.md:4) explains the intended result and displays images, but provides no Python/environment setup, run command, hardware and memory expectations, model-download/cache behavior, or pointer to the exact example invocation. [`main.py`](main.py:8) hard-codes paths, prompt, and output names and currently contains the P0 call defect.
- **Why it matters:** A reviewer cannot reliably reproduce the showcased result or distinguish an environment/setup failure from a pipeline failure. A portfolio project should make its example reproducible without guessing the intended Python command, accelerator, or dependency source.
- **Smallest reasonable fix:** After fixing the inference-call failure, add concise setup and run steps for the supported environment, identify required model downloads / hardware expectations, and explain the sample assets and output. Use a fixed example seed and state that exact pixels can vary by backend and library version.

### P2 — `rembg` is given an unsupported MPS execution-provider name

- **Concrete problem:** [`Cutter.__init__()`](cutter.py:18) requests `MPSExecutionProvider` alongside CUDA and CPU. ONNX Runtime does not provide a general MPS execution provider; provider availability depends on the installed ONNX Runtime build.
- **Why it matters:** The requested provider may be ignored with a warning or cause confusing fallback behavior. The list also does not verify that a requested accelerator provider is actually available.
- **Smallest reasonable fix:** Build the provider list from `onnxruntime.get_available_providers()` and use only providers available in the installed runtime. Do not label MPS support unless using a real supported execution provider such as CoreML where appropriate.

### P2 — The default composition mask repaints the entire canvas

- **Concrete problem:** [`Composer.compose()`](composer.py:158) defaults to `mode="full"`; that mode eventually replaces the mask with an all-white canvas at [`composer.py`](composer.py:207). The example does not override it.
- **Why it matters:** Inpainting is permitted to change the entire background and foreground, not merely soften the pasted boundary. This may be intentional—the project describes global harmonization—but it can change scene details or subject identity and is an important quality/reproducibility trade-off.
- **Smallest reasonable fix:** Document the full-repaint semantics and show how to select a localized mask mode when scene preservation is desired. Keep the current default only if global regeneration is the deliberate product behavior; otherwise choose the intended mode explicitly in the example.

### P2 — Dimension normalization has an unhandled small-image edge case

- **Concrete problem:** [`Composer.compose()`](composer.py:168) rounds each background dimension down to a multiple of 64. A positive input dimension below 64 becomes zero, which is not a valid PIL resize size. The foreground helper has a minimum size, but the background path does not.
- **Why it matters:** Small images fail with a low-level resize error instead of a clear input constraint, and floor-rounding can also discard a substantial border for dimensions just above a multiple of 64.
- **Smallest reasonable fix:** Validate minimum dimensions and state the policy, or resize/pad to a valid multiple of 64 while preserving the intended aspect ratio. Add tests around dimensions below and just above 64-pixel boundaries.

### P2 — Model identity and run settings are not recorded with outputs

- **Concrete problem:** Model IDs are configurable but loaded without pinned Hub revisions at [`inpainters.py`](inpainters.py:165); the sample does not set an explicit seed ([`main.py`](main.py:8)). When unset, a random seed is printed by [`FactorCNetComposerInpainter.inpaint_image()`](inpainters.py:226), but is not saved with the resulting image.
- **Why it matters:** Re-running later may fetch changed model revisions or produce a different sample, and an image file alone does not preserve enough information to reconstruct its prompt, seed, scheduler, model revisions, and backend. A seed does not guarantee bitwise-identical output across different accelerators and library versions.
- **Smallest reasonable fix:** Set an explicit seed in the checked-in example and write a small adjacent metadata record containing the seed, prompt, model IDs/revisions, key inference parameters, device, and relevant package versions. Pin model revisions for the showcased result.

### P2 — There is no model-free test suite for the image-processing contract

- **Concrete problem:** The repository contains no test files. The image/mask code in [`composer.py`](composer.py:110) can be exercised without loading SDXL, while the inference dispatch can be tested with a fake pipeline; neither currently has regression coverage.
- **Why it matters:** The P0 duplicate-argument defect and mask/alpha/size regressions are inexpensive to catch without GPU hardware, but a manual full-model run is the only apparent end-to-end check.
- **Smallest reasonable fix:** Add a small test suite for (1) RGBA composition and placement, (2) mask modes, dimensions, and invalid inputs, (3) parameter conversion and seed handling, and (4) inference keyword dispatch against a mocked pipeline. Keep actual model-download/GPU inference as a documented optional smoke test, not a required unit test.

### P2 — The current configuration mapping protocol is not used by its consumers

- **Concrete problem:** [`FactorPrompts`](configs.py:9) and [`FactorInferenceParameters`](configs.py:33) implement `Mapping` and duplicate dictionary behavior through `__getitem__`, `__iter__`, and `__len__`, but the inference classes call `_as_dict()` explicitly at [`inpainters.py`](inpainters.py:223).
- **Why it matters:** This adds a second API surface without a current call site that benefits from it, and makes it less obvious which representation is authoritative. It does not validate keys added through `extra`, so misspelled or conflicting Diffusers arguments still fail late.
- **Smallest reasonable fix:** Either use the mapping contract consistently or remove the unused protocol methods and retain ordinary dataclasses plus explicit conversion. Keep the flexible extra-parameter escape hatch if it is useful, but validate conflicts with the explicitly managed keys.

### P2 — The composer exposes a misleading model-unload context manager

- **Concrete problem:** [`Composer.__exit__()`](composer.py:216) calls [`Composer.unload()`](composer.py:222), which only runs garbage collection and prints that memory has been freed. The composer owns no diffusion model or other large persistent allocation to unload.
- **Why it matters:** The context manager and status messages suggest model lifecycle management where none exists. [`main.py`](main.py:20) uses it, adding ceremony around a lightweight image utility and potentially confusing the real inpainter cleanup behavior.
- **Smallest reasonable fix:** Remove the composer context-manager/unload methods and instantiate it directly, or make it own a real resource before retaining this API. Do not use garbage collection as a substitute for releasing an actual model reference.

### P3 — Several small leftovers add noise but do not block the pipeline

- **Concrete problem:** There are unused imports in [`cutter.py`](cutter.py:1) and [`inpainters.py`](inpainters.py:1), initial foreground-size assignments in [`Composer._resize()`](composer.py:75) are overwritten before use, and [`FactorCutter.__init__()`](cutter.py:10) only contains `pass`.
- **Why it matters:** These are minor maintenance distractions, not material architectural defects. They make it harder to distinguish active logic from abandoned implementation work.
- **Smallest reasonable fix:** Delete only confirmed-unused imports and assignments, and omit the empty base initializer. Avoid broad formatting or structural refactors as part of this cleanup.

## Other audit areas

### Repository structure and responsibility boundaries

The top-level, single-file-per-concern structure is reasonable for this project's scale. [`composer.py`](composer.py:1), [`cutter.py`](cutter.py:1), and [`inpainters.py`](inpainters.py:1) have recognizable responsibilities, and the image assets make the example tangible. There is no need to introduce a package hierarchy merely for appearance. The concern is behavior at the boundaries: a shared inpainter interface that its alternative implementation does not satisfy, resource cleanup targeting the wrong attribute, and an example script that mixes reusable logic with import-time execution.

### Code correctness, robustness, and error handling

The P0 duplicate keyword failure is the clearest deterministic break. Other predictable input errors include missing/invalid image paths, a non-RGBA foreground rejected by the composer, and dimensions that normalize to zero. The composer checks RGBA mode and mask-mode names, which is useful, but the example otherwise relies on raw PIL, Diffusers, and ONNX Runtime exceptions. Prefer small boundary validations with actionable messages; do not catch and suppress arbitrary model exceptions.

The example orchestration is executed at module import in [`main.py`](main.py:1): importing it performs segmentation, downloads/loads large models, and writes image files. This is awkward for tests and reuse. The minimal improvement is to put orchestration behind a `main()` function and a standard main guard; a full CLI is not necessary.

### Device handling and model memory

The ControlNet implementation assumes an accelerator, defaults to half precision, and has no CPU fallback. The non-ControlNet implementation hard-codes its default device to MPS at [`FactorComposerInpainter.__init__()`](inpainters.py:53). Device policy is inconsistent between implementations. The `rembg` ONNX provider list is separately hard-coded and includes the unsupported MPS provider name noted above.

SDXL + VAE + ControlNet is a large in-memory stack; the code loads these components and moves the assembled pipeline to one device. VAE slicing is enabled, while tiling is disabled. Those choices may be adequate for the included 1024-pixel example, but memory expectations should be documented. Fix actual object-reference cleanup first. Do not add offload frameworks or intricate memory policies without a demonstrated need.

### Dependency management

[`pyproject.toml`](pyproject.toml:1) is the project metadata source and [`uv.lock`](uv.lock:1) provides a lock. However, the project manifest lists many low-level/transitive packages as direct, exactly pinned dependencies, while [`requirements.txt`](requirements.txt:1) is a second, non-equivalent list. This increases maintenance and platform friction for a project whose direct code imports are a much smaller set. Prefer declaring actual direct dependencies in project metadata and using the lock for exact environment resolution; keep an exported requirements file only if it is generated or deliberately supported. Document the tested Python version and hardware path rather than implying that one universal environment is equally supported everywhere.

### Documentation and visual evidence

The English and Russian README explains the high-level idea and shows a source composition/result sequence ([`README.md`](README.md:6)). The supplied example output makes the project easier to understand than a code-only prototype. The README does not explain the mask mode, exact run steps, setup, device/memory requirements, model downloads, seed, or the roles/licensing of the sample assets. Its generic image alt text also does not help screen-reader users. Address reproducibility first; improve labels/captions and input/output descriptions as a small documentation polish pass.

### Signs of overengineering—and what is not overengineering

- The configuration classes' custom `Mapping` implementations and the composer's garbage-collection context manager do not currently solve a demonstrated use case; simplify these instead of adding more abstractions.
- The abstract cutter/inpainter bases should only remain if their implementations actually honor the contracts. The two segmentation model aliases and the separation of preprocessing from model inference are otherwise proportionate to the project.
- The interactive ASCII placement preview is optional and small. It is not worth deleting solely because it is not used by the demo.
- There is no case for services, queues, containers/orchestration platforms, a plugin system, a generalized configuration framework, or elaborate CI/CD in this repository.

## B. Five most important improvements

1. **Fix the inference call** so composition dimensions are passed exactly once; add a mocked-pipeline regression test. This unblocks the current example.
2. **Make device/dtype selection explicit and safe**, including a CPU fallback or a clear early rejection when the selected pipeline cannot run on CPU.
3. **Release actual model references** when leaving the inpainter context and align the alternative inpainter with its declared interface.
4. **Make installation unambiguous** by reconciling project metadata and requirements, then documenting the tested setup and supported accelerator.
5. **Make the sample reproducible and testable**: guard script execution, set a sample seed, record run metadata, add model-free tests for composition/masks/dispatch, and document the exact run steps.

## C. Things that should remain unchanged

- Keep the project focused on one concrete CV task: harmonizing an alpha-composited foreground and background with SDXL + ControlNet.
- Keep the image-processing and diffusion-inference responsibilities separated; the current modules are small and understandable.
- Keep `CompositionResult` carrying both collage and mask as a direct, useful boundary between composition and inference.
- Keep the included input/output assets and a concise runnable example; they communicate the project better than a large framework would.
- Keep optional local experimentation features such as the ASCII placement preview unless maintenance shows they are harmful.
- Do not add infrastructure, abstraction layers, configuration machinery, or performance optimizations without a concrete need demonstrated by the example.

## D. Proposed scope for a focused cleanup pass

Keep the cleanup limited to the following deliverables:

1. Correct the duplicated inference dimensions and add a mocked-pipeline regression test.
2. Fix backend selection and dtype policy, and make the ONNX provider list reflect installed providers.
3. Correct the inpainter cleanup/interface contract; remove the composer's no-op cleanup context manager and confirmed dead code.
4. Choose one dependency source of truth, ensure segmentation dependencies are included, and document one tested install/run path.
5. Put the demo behind a main guard, set an example seed, record essential output metadata, and add model-free tests for composition/masks/config dispatch.

Do not expand this pass into a package redesign, GUI, generalized command-line interface, broad style rewrite, model benchmark suite, or infrastructure project. Keep a real SDXL run as a manual, hardware-dependent smoke test with its requirements clearly stated.
