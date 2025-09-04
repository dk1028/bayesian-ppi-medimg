import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load prediction results (test split of 44)
df = pd.read_csv(Path(r"C:\Users\kimdo\Downloads\제목 없는 스프레드시트 - autorater_predictions_all.csv"))

# 2. Select the most confident AD case and the most confident CN case
ad_row = df[df['label'] == "AD"].sort_values("autorater_prediction", ascending=False).iloc[0]
cn_row = df[df['label'] == "CN"].sort_values("autorater_prediction", ascending=True).iloc[0]

def load_mid_axial_slice(nifti_path):
    """
    Load the middle axial slice from a NIfTI volume, normalize it to [0,1], and return it.
    """
    img = nib.load(nifti_path).get_fdata()
    slice_idx = img.shape[2] // 2
    sl = img[:, :, slice_idx]
    sl = (sl - sl.min()) / (sl.max() - sl.min())
    return sl

# 3. Extract slices, predictions, and ground-truth labels
ad_slice = load_mid_axial_slice(ad_row["nifti_path"])
cn_slice = load_mid_axial_slice(cn_row["nifti_path"])

ad_pred, ad_h = ad_row["autorater_prediction"], ad_row["H"]
cn_pred, cn_h = cn_row["autorater_prediction"], cn_row["H"]

# 4. Create figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

# ── First row: AD case ─────────────────────────────────────────
axes[0, 0].imshow(ad_slice, cmap="gray")
axes[0, 0].set_title("(A) AD MRI Slice")
axes[0, 0].axis("off")

axes[0, 1].bar(["P(AD)"], [ad_pred], color="red")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title("(B) Autorater Prediction")

axes[0, 2].text(
    0.5, 0.5,
    f"Ground-Truth H = {'AD' if ad_h == 1 else 'CN'}",
    fontsize=14, ha="center", va="center"
)
axes[0, 2].axis("off")
axes[0, 2].set_title("(C) Ground-Truth Label")

# ── Second row: CN case ─────────────────────────────────────────
axes[1, 0].imshow(cn_slice, cmap="gray")
axes[1, 0].set_title("(A) CN MRI Slice")
axes[1, 0].axis("off")

axes[1, 1].bar(["P(AD)"], [cn_pred], color="blue")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title("(B) Autorater Prediction")

axes[1, 2].text(
    0.5, 0.5,
    f"Ground-Truth H = {'AD' if cn_h == 1 else 'CN'}",
    fontsize=14, ha="center", va="center"
)
axes[1, 2].axis("off")
axes[1, 2].set_title("(C) Ground-Truth Label")

plt.suptitle("Model prediction quality comparison: AD vs CN (example)", fontsize=16)
plt.show()
