import os
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# 1. Point to your ADNI_NIfTI root folder
root_dir = r"C:\Users\kimdo\Documents\ADNI_NIfTI"

# 2. Find one example subject (e.g. the first folder)
subj_id = sorted(os.listdir(root_dir))[0]
subj_folder = os.path.join(root_dir, subj_id)

# 3. Find the .nii.gz file inside that folder
nii_files = [f for f in os.listdir(subj_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
assert len(nii_files) > 0, "No NIfTI files found!"
nii_path = os.path.join(subj_folder, nii_files[0])

# 4. Load the NIfTI volume
img = nb.load(nii_path)
vol = img.get_fdata()

# 5. Extract the middle axial slice
z_mid = vol.shape[2] // 2
raw_slice = vol[:, :, z_mid]

# 6. Normalize to [0,1] and resize to 64×64
norm = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min())
proc_slice = resize(norm, (64, 64), order=1, mode='constant', cval=0, anti_aliasing=True)

# 7. Plot raw vs. preprocessed
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(raw_slice, cmap='gray')
axes[0].set_title('(A) Raw NIfTI slice')
axes[0].axis('off')
axes[1].imshow(proc_slice, cmap='gray')
axes[1].set_title('(B) Resized & normalized (64×64)')
axes[1].axis('off')
plt.tight_layout()

# 8. Save your pipeline figure
out_path = os.path.join("figures", "fig1_pipeline.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
plt.close(fig)
