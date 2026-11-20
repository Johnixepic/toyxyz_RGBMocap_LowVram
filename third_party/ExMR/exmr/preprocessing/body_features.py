"""SAM 3D Body image features: the body decoder's pose token, one per frame."""

import contextlib
import io
import logging
import re
import sys
import warnings
from pathlib import Path

from mocap.batch_prefetch import batch_source

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from exmr.preprocessing.image_io import read_rgb_frame

# SAM 3D Body is not pip-installable; it is a git submodule of this repository
SAM_3D_BODY_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sam-3d-body"

# Parameters that SAM 3D Body's checkpoint lacks by design: the MHR body model's buffers (loaded
# from mhr_model.pt), the hand PCA copies (set in SAM3DBody.__init__) and DINOv3's mask token
# (zero-initialized; unused at inference)
_EXPECTED_MISSING_KEY = re.compile(
    r"backbone\.encoder\.mask_token|head_pose(_hand)?\.(hand_pose_comps_ori|mhr\..+)"
)
_MISMATCH = "The model and loaded state dict do not match exactly"
_MISSING = "missing keys in source state_dict: "


def _drop_expected_mismatch(record):
    """Logging filter: drop the checkpoint-mismatch warning if it lists only expected keys."""
    msg = record.getMessage()
    if not msg.startswith(_MISMATCH) or "unexpected key" in msg or _MISSING not in msg:
        return True
    keys = msg.split(_MISSING, 1)[1].strip().split(", ")
    return not all(_EXPECTED_MISSING_KEY.fullmatch(k) for k in keys)


def _drop_ignored_kwargs(record):
    """Logging filter: DINOv3 ignores the config's drop_path, which only matters in training."""
    return not record.getMessage().startswith("Ignored kwargs: {'drop_path'")


@contextlib.contextmanager
def _quiet_model_loading():
    """Hide SAM 3D Body's expected messages while loading the model (real problems still show)."""
    filters = [
        (logging.getLogger("sam_3d_body.utils.checkpoint"), _drop_expected_mismatch),
        (logging.getLogger("dinov3"), _drop_ignored_kwargs),
    ]
    for logger, f in filters:
        logger.addFilter(f)
    try:
        # load_sam_3d_body prints "Loading SAM 3D Body model..."
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for logger, f in filters:
            logger.removeFilter(f)


def _import_sam_3d_body():
    try:
        import sam_3d_body  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(SAM_3D_BODY_ROOT))
    with warnings.catch_warnings():
        # MHR head: optional Momentum dependency, unused by the PyTorch MHR model
        warnings.filterwarnings("ignore", message="Momentum is (not )?enabled")
        import sam_3d_body
        from sam_3d_body.data import transforms
        from sam_3d_body.data.utils.io import load_image
        from sam_3d_body.data.utils.prepare_batch import prepare_batch
        from sam_3d_body.utils import recursive_to

    return sam_3d_body, transforms, load_image, prepare_batch, recursive_to


class SamBodyFeatures:
    """Per-frame SAM 3D Body pose tokens (1024-d) of one person, given its bounding boxes.

    The token is read from the body decoder with a forward hook (it is not part of the model's
    output). The hand-detection tokens stay enabled as in the original model config, since
    they influence the body token through self-attention.

    Args:
        checkpoint_dir: directory with model.ckpt, model_config.yaml and assets/mhr_model.pt of
            facebook/sam-3d-body-dinov3
        fp16: run the backbone in float16 instead of the checkpoint's bfloat16 (SAM 3D Body's own
            FP16_TYPE option). Faster only with FP16 accumulation
            (torch.backends.cuda.matmul.allow_fp16_accumulation); tokens change at the 1e-3 level
    """

    TOKEN_DIM = 1024

    def __init__(self, checkpoint_dir, fp16=False, batch_size=32, num_workers=8, device="cuda", low_memory=False):
        sam_3d_body, self._transforms, self._load_image, self._prepare_batch, self._to = (
            _import_sam_3d_body()
        )
        checkpoint_dir = Path(checkpoint_dir)
        # Fast loading initializes on GPU; Low Memory accepts slower CPU
        # initialization to avoid holding temporary checkpoint/model copies in VRAM.
        load_device = "cpu" if low_memory else device
        with _quiet_model_loading(), torch.device(load_device):
            model, model_cfg = sam_3d_body.load_sam_3d_body(
                str(checkpoint_dir / "model.ckpt"),
                device=load_device,
                mhr_path=str(checkpoint_dir / "assets" / "mhr_model.pt"),
            )
        assert model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS
        if fp16:
            for t in [*model.backbone.parameters(), *model.backbone.buffers()]:
                if t.dtype == torch.bfloat16:
                    t.data = t.data.half()
            model.backbone_dtype = torch.float16
        self.device = load_device if low_memory else device
        self.model = model.eval().to(self.device)

        # Optional hybrid CPU/GPU DINOv3 execution for Low Memory mode.
        gpu_blocks = 0
        if low_memory and torch.cuda.is_available():
            gpu_blocks = max(0, min(
                int(__import__("os").environ.get("MOCAP_DINO_GPU_BLOCKS", "32")),
                32,
            ))

        encoder = getattr(self.model.backbone, "encoder", None)
        if encoder is not None and gpu_blocks:
            gpu_blocks = min(gpu_blocks, len(encoder.blocks))
            split = len(encoder.blocks) - gpu_blocks
            for block in encoder.blocks[split:]:
                block.to("cuda")
            encoder._mocap_gpu_block_start = split
            print(f"DINOv3 hybrid mode: {len(encoder.blocks) - split} blocks on CUDA")
        elif encoder is not None:
            encoder._mocap_gpu_block_start = None

        self.input_size = model_cfg.MODEL.IMAGE_SIZE
        self.batch_size, self.num_workers = batch_size, num_workers

        # the decoder returns (tokens, pose_output); token 0 is the pose token
        self._tokens = None
        self._token_hook = self.model.decoder.register_forward_hook(self._store_tokens)

    def close(self):
        self._token_hook.remove()
        self._tokens = None

    def _store_tokens(self, _module, _inputs, output):
        self._tokens = output[0].detach()

    @torch.no_grad()
    def __call__(self, frame_dir, bbx_xyxy, K_fullimg, checkpoint_path=None, checkpoint_identity=None):
        """Extract one body token per frame, optionally resuming an identity-checked checkpoint."""
        import os
        import uuid

        self.model._mocap_features_only=True
        self.model.decoder._mocap_features_only=True
        t = self._transforms
        transform = t.Compose([
            t.GetBBoxCenterScale(),
            t.TopdownAffine(input_size=self.input_size, use_udp=False),
            t.VisionTransformWrapper(ToTensor()),
        ])
        dataset = _FrameDataset(
            Path(frame_dir), bbx_xyxy, K_fullimg, transform, self._load_image, self._prepare_batch
        )
        tokens = torch.zeros((len(dataset), self.TOKEN_DIM), dtype=torch.float32)
        start = 0

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            try:
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                saved_tokens = saved["tokens"]
                completed = int(saved["completed"])
                if (saved["identity"] == checkpoint_identity
                        and 0 <= completed <= len(dataset)
                        and tuple(saved_tokens.shape) == (completed, self.TOKEN_DIM)
                        and bool(torch.isfinite(saved_tokens).all())):
                    tokens[:completed] = saved_tokens
                    start = completed
                    print(f"Resuming SAM body features at frame {start}/{len(dataset)}")
            except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError):
                pass

        def save_checkpoint(completed):
            if checkpoint_path is None:
                return
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_name(checkpoint_path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                torch.save({
                    "identity": checkpoint_identity,
                    "completed": completed,
                    "tokens": tokens[:completed].clone(),
                }, temporary)
                os.replace(temporary, checkpoint_path)
            finally:
                temporary.unlink(missing_ok=True)

        if start < len(dataset):
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=range(start, len(dataset)),
                num_workers=self.num_workers,
                pin_memory=True,
            )
            i = start
            profile = os.environ.get("MOCAP_EXMR_PROFILE", "").strip().lower() in ("1", "true", "yes")
            profile_times = {name: 0.0 for name in ("batch_wait", "host_to_device", "initialize", "forward", "device_to_host", "checkpoint")}
            profile_batches = 0

            def profile_sync():
                if profile and self.device != "cpu":
                    torch.cuda.synchronize(self.device)

            with batch_source(loader, batch_bytes=self.batch_size*3*1024*1024*4) as batches, tqdm(
                total=len(dataset), initial=start, desc="SAM 3D Body features", unit="frame"
            ) as pbar:
                batch_iter = iter(batches)
                while True:
                    t0 = time.perf_counter()
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    profile_times["batch_wait"] += time.perf_counter() - t0
                    bs = batch["img"].shape[0]

                    t0 = time.perf_counter()
                    batch = self._to(batch, self.device)
                    profile_sync()
                    profile_times["host_to_device"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    self.model._initialize_batch(batch)
                    profile_sync()
                    profile_times["initialize"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    self.model.forward_step(batch, decoder_type="body")
                    profile_sync()
                    profile_times["forward"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    tokens[i:i + bs] = self._tokens[:, 0].cpu()
                    profile_sync()
                    profile_times["device_to_host"] += time.perf_counter() - t0

                    i += bs
                    profile_batches += 1
                    pbar.update(bs)
                    if i % 10 == 0 or i == len(dataset):
                        t0 = time.perf_counter()
                        save_checkpoint(i)
                        profile_times["checkpoint"] += time.perf_counter() - t0

            if profile:
                elapsed = sum(profile_times.values())
                print(
                    "SAM body profile (s): "
                    + ", ".join(f"{name}={seconds:.2f}" for name, seconds in profile_times.items())
                    + f", measured_total={elapsed:.2f}, batches={profile_batches}, "
                    + f"frames={i - start}, batch_size={self.batch_size}",
                    flush=True,
                )
                if self.device != "cpu":
                    print(
                        "SAM body CUDA peak (MiB): "
                        f"allocated={torch.cuda.max_memory_allocated(self.device) / 1024**2:.0f}, "
                        f"reserved={torch.cuda.max_memory_reserved(self.device) / 1024**2:.0f}",
                        flush=True,
                    )

        if not torch.isfinite(tokens).all():
            raise RuntimeError("Non-finite SAM 3D Body features; refusing to save invalid output")
        return tokens


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, bbx_xyxy, K_fullimg, transform, load_image, prepare_batch):
        assert len(bbx_xyxy) == len(K_fullimg)
        self.frame_dir, self.bbx_xyxy, self.K_fullimg = frame_dir, bbx_xyxy, K_fullimg
        self.transform, self.load_image, self.prepare_batch = transform, load_image, prepare_batch

    def __len__(self):
        return len(self.bbx_xyxy)

    def __getitem__(self, idx):
        img = read_rgb_frame(self.frame_dir / f"{idx:06d}.jpg")
        batch = self.prepare_batch(
            img,
            self.transform,
            self.bbx_xyxy[idx].reshape(1, 4),
            masks=None,
            masks_score=None,
            cam_int=self.K_fullimg[idx],
        )
        del batch["img_ori"]
        # prepare_batch makes a batch of persons per image; there is exactly one
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                batch[k] = v[0]
        return batch
