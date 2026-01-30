import os, pathlib, shutil

# ---- EDIT THESE TWO PATHS ----
SRC  = r"KMU-dataset"         # where your images are now
REPO = r"ShuffViT-DFER"   # repo root
# --------------------------------

DST = os.path.join(REPO, "KMU")

# Map filename codes -> target folder names expected by preprocess_kmu.py
CODE2FOLDER = {
    "AN": "anger",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise",
    "AF": "fear",   # some sets use AF (afraid)
    "FE": "fear",   # some sets use FE (fear)
}

# make target dirs
for d in set(CODE2FOLDER.values()):
    os.makedirs(os.path.join(DST, d), exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
moved = 0
skipped = 0

for p in pathlib.Path(SRC).rglob("*"):
    if not p.is_file() or p.suffix not in exts:
        continue
    # Expect pattern like: 01_AN_mr_001.jpg -> tokens split by "_"
    parts = p.stem.split("_")
    code = parts[1].upper() if len(parts) > 1 else ""
    folder = CODE2FOLDER.get(code)
    if folder:
        dst = os.path.join(DST, folder, p.name)
        shutil.copy2(str(p), dst)   # change to shutil.move(...) if you want to MOVE instead
        moved += 1
    else:
        skipped += 1

print(f"Done. Copied {moved} files. Skipped {skipped} (unknown pattern).")
print(f"Now run:  python preprocess_kmu.py")
