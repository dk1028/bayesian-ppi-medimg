import pandas as pd
from pathlib import Path

# 1) Read file (comma-separated CSV)
META_CSV = Path(r"C:\Users\AV75950\Downloads\6570year_7_18_2025.csv")
meta = pd.read_csv(META_CSV)  # sep="," is the default

# 2) Clean column names: trim spaces and replace internal spaces with underscores
meta.columns = (
    meta.columns
        .str.strip()             # remove leading/trailing spaces
        .str.replace(' ', '_')   # "Acq Date" → "Acq_Date"
)

# Check
print("Cleaned columns:", meta.columns.tolist())

# 3) Parse dates
meta['Acq_Date'] = pd.to_datetime(
    meta['Acq_Date'],
    format='%m/%d/%Y',
    errors='coerce'
).dt.date

# Now keep only subject_id, label, Acq_Date
meta = meta.rename(columns={'Subject':'subject_id', 'Group':'label'})
meta = meta[['subject_id','label','Acq_Date']].drop_duplicates()

# 4) NIfTI file matching (same as before)
from pathlib import Path
records = []
NIFTI_ROOT = Path(r"C:\Users\AV75950\Documents\ADNI_NIfTI")

for subdir in NIFTI_ROOT.iterdir():
    if not subdir.is_dir(): continue
    for nii in subdir.glob("*.nii.gz"):
        stem = nii.stem
        parts = stem.split('_')
        subject_id = "_".join(parts[:3])
        date_str   = parts[3]
        try:
            scan_date = pd.to_datetime(date_str, format='%Y-%m-%d').date()
        except:
            continue
        records.append({
            'subject_id': subject_id,
            'scan_date' : scan_date,
            'nifti_path': str(nii)
        })

nifti_df = pd.DataFrame(records)

# 5) Merge
merged = pd.merge(
    nifti_df,
    meta,
    left_on = ['subject_id','scan_date'],
    right_on= ['subject_id','Acq_Date'],
    how='inner'
)

print("Number of matched samples:", len(merged))
print(merged.head())

# 6) Save
merged.to_csv("matched_cn_ad_labels.csv", index=False)
print("✅ Saved to matched_cn_ad_labels.csv")
