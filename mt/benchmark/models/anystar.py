from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import shutil

from .stardist import StarDist


class AnyStar(StarDist):
    """
    AnyStar wrapper built on top of the generic StarDist (3D) class.

    Behavior:
    - If `model_dir` is provided, that folder is used.
    - Else, if `pretrained` (case-insensitive) == "anystar", this class ensures
      a local folder exists at 'models/AnyStar/anystar-mix/' and uses it.
    - Else, falls back to StarDist built-in 3D pretrained names.

    Notes:
    - Requires `gdown` to be installed if it needs to download the weights.
    """

    # Google Drive folder for AnyStar release (provided earlier)
    _ANYSTAR_DRIVE_ID = "1yiY_vBR2GQW9zJzgUPRWeIecN4ZnCi3c"

    def __init__(
        self,
        *,
        pretrained: Optional[str] = "AnyStar",          # trigger auto-managed weights
        model_dir: Optional[Union[str, Path]] = None,   # explicit local model folder
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        normalize: bool = True,
        norm_percentiles: Tuple[float, float] = (1.0, 99.8),
        channel: Optional[int] = None,
        n_tiles: Optional[Union[int, Tuple[int, int, int]]] = None,
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
        # If user provided a folder, use it
        if self.model_dir is not None:
            return self.model_dir

        # If the user asked for "AnyStar", ensure/download the local weights
        if (self.pretrained or "").lower() == "anystar":
            return self._ensure_anystar_weights()

        # Otherwise, fall back to built-in pretrained StarDist3D (handled by base)
        return None

    # ---- internal helpers ----
    def _ensure_anystar_weights(self) -> Path:
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


        return model_dir
