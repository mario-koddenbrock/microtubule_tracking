from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import shutil
import json

from .stardist import StarDist


class AnyStar(StarDist):
    """
    2D AnyStar wrapper built on top of the generic StarDist (2D) class.

    Behavior:
    - If `model_dir` is provided, that folder is used (must be 2D).
    - Else, if `pretrained` (case-insensitive) == "anystar", this class ensures
      a local folder exists at 'models/AnyStar/anystar-mix/' and uses it.
      If the downloaded content is 3D, it raises with a clear message.
    - Else, falls back to StarDist built-in 2D pretrained names.

    Notes:
    - Requires `gdown` to be installed if it needs to download the weights.
    """

    # Google Drive folder for AnyStar release (provided earlier)
    _ANYSTAR_DRIVE_ID = "1yiY_vBR2GQW9zJzgUPRWeIecN4ZnCi3c"

    def __init__(
        self,
        *,
        pretrained: Optional[str] = "AnyStar",          # trigger auto-managed weights
        model_dir: Optional[Union[str, Path]] = None,   # explicit local model folder (must be 2D)
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        normalize: bool = True,
        norm_percentiles: Tuple[float, float] = (1.0, 99.8),
        channel: Optional[int] = None,
        n_tiles: Optional[Union[int, Tuple[int, int]]] = None,
        dest_root: Union[str, Path] = "models/AnyStar",
        model_name_on_disk: str = "anystar-mix",
    ):
        self._dest_root = Path(dest_root)
        self._model_name_on_disk = model_name_on_disk
        super().__init__(
            model_name="AnyStar",
            pretrained=pretrained,
            model_dir=model_dir,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            normalize=normalize,
            norm_percentiles=norm_percentiles,
            channel=channel,
            n_tiles=n_tiles,
        )

    # ---- hook implementation ----
    def _prepare_model_dir(self) -> Optional[Path]:
        # If user provided a folder, use it (the base class will validate it's 2D)
        if self.model_dir is not None:
            return self.model_dir

        # If the user asked for "AnyStar", ensure/download the local 2D weights
        if (self.pretrained or "").lower() == "anystar":
            return self._ensure_anystar_2d_weights()

        # Otherwise, fall back to built-in pretrained StarDist2D (handled by base)
        return None

    # ---- internal helpers ----
    def _ensure_anystar_2d_weights(self) -> Path:
        dest_root = self._dest_root
        model_dir = dest_root / self._model_name_on_disk
        cfg = model_dir / "config.json"

        if not cfg.exists():
            # Download the whole Drive folder to a temp and move the inner model folder in place
            import gdown  # assumed installed
            dest_root.mkdir(parents=True, exist_ok=True)
            tmp = dest_root / f"__tmp_{self._model_name_on_disk}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)

            url = f"https://drive.google.com/drive/folders/{self._ANYSTAR_DRIVE_ID}"
            gdown.download_folder(url=url, output=str(tmp), quiet=False, use_cookies=False)

            candidates = list(tmp.rglob("config.json"))
            if not candidates:
                raise RuntimeError("AnyStar download did not contain a StarDist model (missing config.json).")

            src = candidates[0].parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.move(str(src), str(model_dir))
            shutil.rmtree(tmp, ignore_errors=True)

        # Validate 2D
        with cfg.open("r") as f:
            config = json.load(f)
        grid = config.get("grid", [])
        n_dim = int(config.get("n_dim", 2))

        if (isinstance(grid, (list, tuple)) and len(grid) == 3) or n_dim == 3:
            raise RuntimeError(
                f"The downloaded AnyStar model at '{model_dir}' appears to be 3D (grid={grid}, n_dim={n_dim}). "
                f"This project is 2D-only. Please use a 2D StarDist model "
                f"(e.g., pretrained='2D_versatile_fluo') or provide a 2D model_dir."
            )

        return model_dir
