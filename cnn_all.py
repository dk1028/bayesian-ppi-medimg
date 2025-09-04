# autorater_cnn_train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1) Settings
CSV_PATH    = Path(r"C:\Users\AV75950\Documents\matched_cn_ad_labels3.csv")
NIFTI_ROOT  = Path(r"C:\Users\AV75950\Documents\ADNI_NIfTI")
BATCH_SIZE  = 2
EPOCHS      = 5
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Define Dataset
class MRIDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        # Load NIfTI and convert: numpy → tensor
        img = nib.load(row['nifti_path']).get_fdata().astype('float32')
        img = torch.from_numpy(img).unsqueeze(0)  # shape [1, D, H, W]
        # resize to 64³
        img = F.interpolate(img.unsqueeze(0),
                            size=(64,64,64),
                            mode='trilinear',
                            align_corners=False).squeeze(0)
        # normalize
        img = (img - img.mean())/(img.std()+1e-5)
        label = torch.tensor(1 if row['label']=='AD' else 0, dtype=torch.float32)
        return img, label

# 3) Simple 3D CNN
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8,16,3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.fc1   = nn.Linear(16*16*16*16, 64)
        self.fc2   = nn.Linear(64, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → [B,8,32,32,32]
        x = self.pool(F.relu(self.conv2(x)))  # → [B,16,16,16,16]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(1)

# 4) Load data and split
df = pd.read_csv(CSV_PATH)
train_df, test_df = train_test_split(df,
                                     test_size=0.2,
                                     stratify=df['label']=='AD',
                                     random_state=42)
train_ds = MRIDataset(train_df)
test_ds  = MRIDataset(test_df)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 5) Prepare model/loss/optimizer
model     = Simple3DCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 6) Training loop
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for X,y in train_dl:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}  Loss: {total_loss/len(train_dl):.4f}")

# 7) Predict on test set and save
model.eval()
preds, labs = [], []
with torch.no_grad():
    for X,y in test_dl:
        X = X.to(DEVICE)
        out = model(X).cpu().numpy()
        preds.extend(out.tolist()); labs.extend(y.tolist())
out_df = test_df.copy()
out_df['autorater_prediction'] = preds
out_df['H'] = labs
out_df.to_csv("autorater_predictions4.csv", index=False)
print("✅ autorater_predictions.csv generated")

all_dl = DataLoader(MRIDataset(df), batch_size=BATCH_SIZE)
preds = []
with torch.no_grad():
    for X, _ in all_dl:
        preds.extend(model(X.to(DEVICE)).cpu().numpy())
df['autorater_prediction'] = preds
df.to_csv("autorater_predictions_all4.csv", index=False)
