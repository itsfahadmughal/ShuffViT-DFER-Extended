# preprocess_kmu_stream.py
import os, glob, h5py
import numpy as np
import skimage.io
from skimage.transform import resize
from tqdm import tqdm

# ---------- Config ----------
ck_path   = 'KMU'                 # parent folder with class dirs
img_size  = 224                   # resize target (HxW)
batch_sz  = 128                   # lower if RAM is tight (64–256)
compress  = True                  # False = faster writes, larger file
out_path  = os.path.join('KMUtada', 'mtcnnkmunew.h5')
# ----------------------------

# Your folder names (you changed to 'sad' — keeping that)
CLASSES = [
    (0, 'anger'),
    (1, 'disgust'),
    (2, 'fear'),
    (3, 'happy'),
    (4, 'sad'),        # <- 'sad' folder, not 'sadness'
    (5, 'surprise'),
]

def files_in(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
    fs = []
    for e in exts:
        fs.extend(glob.glob(os.path.join(folder, e)))
    return sorted(fs)

# 1) Collect file lists (tiny memory)
class_files = []
total = 0
print("Counting images per class:")
for label, name in CLASSES:
    path = os.path.join(ck_path, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing folder: {path}")
    fs = files_in(path)
    class_files.append((label, name, fs))
    total += len(fs)
    print(f"  {name:<8} -> {len(fs)}")
if total == 0:
    raise SystemExit("No images found under KMU/*")

# 2) Prepare output HDF5 (growable datasets)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
H, W, C = img_size, img_size, 3
comp = dict(chunks=True, compression='gzip', compression_opts=3) if compress else dict()

with h5py.File(out_path, 'w') as f:
    d_img = f.create_dataset('data_pixel', shape=(0, H, W, C),
                             maxshape=(None, H, W, C), dtype='uint8', **comp)
    d_lab = f.create_dataset('data_label', shape=(0,),
                             maxshape=(None,), dtype='int64')

    n_written = 0

    # 3) Stream per class in batches
    for label, name, fs in class_files:
        print(f"Processing {name} ...")
        for i in tqdm(range(0, len(fs), batch_sz), desc=name, unit='batch'):
            batch_paths = fs[i:i+batch_sz]
            b = len(batch_paths)

            buf_x = np.empty((b, H, W, C), dtype=np.uint8)
            buf_y = np.full((b,), label, dtype=np.int64)

            # Load & resize each image into the batch buffer
            for j, p in enumerate(batch_paths):
                I = skimage.io.imread(p)
                # normalize channels
                if I.ndim == 2:                         # grayscale -> RGB
                    I = np.stack([I, I, I], axis=-1)
                elif I.ndim == 3 and I.shape[2] == 4:    # RGBA -> RGB
                    I = I[:, :, :3]

                # resize to (H, W); skimage returns float64 0..1
                Ir = resize(I, (H, W), anti_aliasing=True, preserve_range=True)
                buf_x[j] = np.clip(Ir, 0, 255).astype(np.uint8)

            # Append this batch to the datasets
            d_img.resize((n_written + b, H, W, C))
            d_lab.resize((n_written + b,))
            d_img[n_written:n_written + b] = buf_x
            d_lab[n_written:n_written + b] = buf_y
            n_written += b

print(f"Saved HDF5: {out_path}")