import json
import random
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from data_layer import metadata_store

_ANNOTATIONS_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
_ANNOTATIONS_FILE   = "instances_val2017.json"


class COCOBenchmarkLoader:

    def __init__(self, data_dir: str, max_images: int = 500) -> None:
        self.data_dir         = Path(data_dir)
        self.max_images       = max_images
        self.images_dir       = self.data_dir / "val2017"
        self.annotations_path = self.data_dir / _ANNOTATIONS_FILE

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Annotations: download and extract once
    # ------------------------------------------------------------------

    def _ensure_annotations(
        self, progress_callback: Optional[Callable] = None
    ) -> Path:
        if self.annotations_path.exists():
            return self.annotations_path

        zip_path = self.data_dir / "annotations_trainval2017.zip"

        if not zip_path.exists():
            msg = (
                "Downloading COCO val2017 annotations (~241 MB, one-time). "
                "This may take a few minutes..."
            )
            print(msg)
            if progress_callback:
                progress_callback("status", msg)

            def _reporthook(block: int, block_size: int, total: int) -> None:
                if total > 0 and progress_callback:
                    pct = min(100, block * block_size * 100 // total)
                    progress_callback("progress", pct * 0.4)
                    progress_callback("status", f"Downloading COCO annotations: {pct}%")

            urllib.request.urlretrieve(_ANNOTATIONS_ZIP_URL, zip_path, _reporthook)
            print(f"Download complete: {zip_path}")

        if progress_callback:
            progress_callback("status", "Extracting COCO annotations...")
            progress_callback("progress", 45)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            target  = (
                _ANNOTATIONS_MEMBER
                if _ANNOTATIONS_MEMBER in members
                else next((m for m in members if _ANNOTATIONS_FILE in m), None)
            )
            if target is None:
                raise FileNotFoundError(
                    f"{_ANNOTATIONS_FILE} not found inside the annotations zip."
                )
            zf.extract(target, self.data_dir)
            extracted = self.data_dir / target
            if extracted != self.annotations_path:
                extracted.rename(self.annotations_path)
            # Remove the now-empty sub-directory left by extraction
            try:
                extracted.parent.rmdir()
            except OSError:
                pass

        print(f"Annotations ready: {self.annotations_path}")
        return self.annotations_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load_data(
        self, progress_callback: Optional[Callable] = None
    ) -> Tuple[List[str], List[List[float]]]:
        ann_path = self._ensure_annotations(progress_callback)

        if progress_callback:
            progress_callback("status", "Parsing COCO annotations...")
            progress_callback("detail", f"Annotations file: {ann_path.name}")
            progress_callback("progress", 50)

        with open(ann_path, "r", encoding="utf-8") as f:
            coco: Dict[str, Any] = json.load(f)

        # category_id → name
        cat_id_to_name: Dict[int, str] = {
            c["id"]: c["name"] for c in coco["categories"]
        }
        # Sorted list for consistent multi-label vector ordering
        sorted_cats  = sorted(cat_id_to_name.values())
        cat_to_idx   = {name: i for i, name in enumerate(sorted_cats)}
        n_cats       = len(sorted_cats)

        # image_id → {category_name, ...}
        img_cats: Dict[int, set] = {}
        for ann in coco["annotations"]:
            name = cat_id_to_name.get(ann["category_id"])
            if name:
                img_cats.setdefault(ann["image_id"], set()).add(name)

        # image_id → image info dict
        img_index: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in coco["images"]
        }

        # Select a reproducible subset
        candidates = [iid for iid in img_cats if iid in img_index]
        if len(candidates) > self.max_images:
            random.seed(42)
            candidates = random.sample(candidates, self.max_images)

        if progress_callback:
            progress_callback(
                "status", f"Downloading / verifying {len(candidates)} images..."
            )
            progress_callback("detail", f"Categories: {n_cats}  |  Images to load: {len(candidates)}")

        image_paths:  List[str]         = []
        multi_labels: List[List[float]] = []
        total = len(candidates)

        for i, img_id in enumerate(candidates):
            local = self._ensure_image(img_index[img_id])
            if local is None:
                continue

            vec = [0.0] * n_cats
            for cat in img_cats[img_id]:
                idx = cat_to_idx.get(cat)
                if idx is not None:
                    vec[idx] = 1.0

            image_paths.append(local)
            multi_labels.append(vec)

            if progress_callback and (i % 20 == 0 or i == total - 1):
                pct = 50 + (i + 1) / total * 35
                progress_callback("progress", pct)
                progress_callback("status", f"Images ready: {i + 1}/{total}")
                progress_callback("detail", f"Images ready: {i + 1}/{total}")

        if not image_paths:
            print("No COCO images could be loaded.")
            return [], []

        if progress_callback:
            progress_callback("status", "Seeding database with COCO ground truth...")
            progress_callback("detail", f"Seeding {len(image_paths)} images into database...")
        self._seed_database(image_paths, multi_labels, sorted_cats)

        if progress_callback:
            progress_callback("progress", 100)
            progress_callback(
                "status",
                f"COCO ready: {len(image_paths)} images, {n_cats} categories",
            )
            progress_callback("detail", f"COCO data ready: {len(image_paths)} images, {n_cats} categories")

        print(
            f"COCO benchmark data: {len(image_paths)} images, "
            f"{n_cats} categories"
        )
        return image_paths, multi_labels

    # ------------------------------------------------------------------
    # Image caching
    # ------------------------------------------------------------------

    def _ensure_image(self, img_info: Dict[str, Any]) -> Optional[str]:
        local = self.images_dir / img_info["file_name"]
        if local.exists():
            return str(local)

        url = img_info.get("coco_url") or img_info.get("flickr_url")
        if not url:
            return None

        try:
            urllib.request.urlretrieve(url, local)
            return str(local)
        except Exception as exc:
            print(f"  Could not download {img_info['file_name']}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Database seeding (idempotent via INSERT OR IGNORE)
    # ------------------------------------------------------------------

    def _seed_database(
        self,
        image_paths:  List[str],
        multi_labels: List[List[float]],
        sorted_cats:  List[str],
    ) -> None:
        """Insert COCO images and ground-truth labels into the app DB."""
        cursor = metadata_store.connection.cursor()
        now    = datetime.now().isoformat()

        # Resolve tag IDs before opening the batch transaction so that
        # get_or_create_tag's internal commits don't conflict.
        cat_tag_ids: Dict[str, int] = {
            cat: metadata_store.get_or_create_tag(cat, category="coco", commit=True)
            for cat in sorted_cats
        }

        metadata_store.connection.execute("BEGIN")
        try:
            for img_path, label_vec in zip(image_paths, multi_labels):
                p        = Path(img_path)
                image_id = metadata_store.add_image(
                    file_path=img_path,
                    file_name=p.name,
                    file_size=p.stat().st_size if p.exists() else 0,
                    commit=False,
                )
                if image_id is None:
                    continue

                for cat, label in zip(sorted_cats, label_vec):
                    if label != 1.0:
                        continue
                    tag_id = cat_tag_ids[cat]

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO image_tags
                            (image_id, tag_id, confidence, source, created_at)
                        VALUES (?, ?, 1.0, 'coco_ground_truth', ?)
                        """,
                        (image_id, tag_id, now),
                    )
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO ground_truth
                            (image_id, tag_id, verified_by, verified_at)
                        VALUES (?, ?, 'coco', ?)
                        """,
                        (image_id, tag_id, now),
                    )

            metadata_store.connection.commit()
        except Exception:
            metadata_store.connection.rollback()
            raise
