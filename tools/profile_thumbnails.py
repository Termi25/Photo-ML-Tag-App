import os
import time
from pathlib import Path
from statistics import mean
import hashlib

try:
    import cv2
    cv2_available = True
except Exception:
    cv2_available = False

from PIL import Image

ROOT = Path("ExampleImageFolder")
SAMPLE_N = 100
THUMB_SIZE = 160
CACHE_DIR = Path("ExampleImageFolder") / ".thumbs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def sample_images(root, n):
    imgs = []
    for p in root.rglob("*.jpg"):
        imgs.append(p)
        if len(imgs) >= n:
            break
    for p in root.rglob("*.jpeg"):
        if len(imgs) >= n: break
        imgs.append(p)
    for p in root.rglob("*.png"):
        if len(imgs) >= n: break
        imgs.append(p)
    return imgs


def time_pil(paths):
    times = []
    for p in paths:
        t0 = time.perf_counter()
        with Image.open(p) as im:
            im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            im.load()
        t1 = time.perf_counter()
        times.append(t1-t0)
    return times


def time_cv2(paths):
    if not cv2_available:
        return []
    times = []
    import cv2
    for p in paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            h,w = img.shape[:2]
            max_dim = max(h,w)
            if max_dim > THUMB_SIZE:
                scale = THUMB_SIZE / max_dim
                new_w, new_h = int(w*scale), int(h*scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        t1 = time.perf_counter()
        times.append(t1-t0)
    return times


def time_stat(paths):
    times = []
    for p in paths:
        t0 = time.perf_counter()
        p.stat()
        t1 = time.perf_counter()
        times.append(t1-t0)
    return times


def run():
    paths = sample_images(ROOT, SAMPLE_N)
    if not paths:
        print('No images found under', ROOT)
        return
    print(f'Sampled {len(paths)} images for profiling')

    print('Measuring filesystem stat() overhead...')
    stat_times = time_stat(paths)
    print(f'stat: mean={mean(stat_times):.4f}s, median ~ {sorted(stat_times)[len(stat_times)//2]:.4f}s')

    print('\nMeasuring PIL thumbnail load times...')
    pil_times = time_pil(paths)
    print(f'PIL thumbnail: mean={mean(pil_times):.4f}s, median ~ {sorted(pil_times)[len(pil_times)//2]:.4f}s')

    if cv2_available:
        print('\nMeasuring OpenCV load times...')
        cv2_times = time_cv2(paths)
        print(f'cv2 read+resize: mean={mean(cv2_times):.4f}s, median ~ {sorted(cv2_times)[len(cv2_times)//2]:.4f}s')
    else:
        print('\nOpenCV not available')

    # --- Create disk cache for thumbnails using same hashing scheme ---
    print('\nWarming disk thumbnail cache...')
    created = 0
    for p in paths:
        try:
            key = hashlib.sha256(str(Path(p).resolve()).encode('utf-8')).hexdigest()
            cache_path = CACHE_DIR / f"{key}_{THUMB_SIZE}.jpg"
            if cache_path.exists():
                continue
            with Image.open(p) as im:
                im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                im.convert('RGB').save(cache_path, format='JPEG', quality=85)
            created += 1
        except Exception:
            pass
    print(f'Created {created} cached thumbnails in {CACHE_DIR}')

    # Measure loading cached thumbnails (open + load)
    print('\nMeasuring cached thumbnail open times...')
    cached_paths = [CACHE_DIR / f"{hashlib.sha256(str(Path(p).resolve()).encode('utf-8')).hexdigest()}_{THUMB_SIZE}.jpg" for p in paths]
    cached_times = []
    for cp in cached_paths:
        if not cp.exists():
            continue
        t0 = time.perf_counter()
        with Image.open(cp) as im:
            im.load()
        t1 = time.perf_counter()
        cached_times.append(t1-t0)
    if cached_times:
        print(f'Cached open: mean={mean(cached_times):.4f}s, median ~ {sorted(cached_times)[len(cached_times)//2]:.4f}s')
    else:
        print('No cached thumbnails to measure')

if __name__ == '__main__':
    run()
