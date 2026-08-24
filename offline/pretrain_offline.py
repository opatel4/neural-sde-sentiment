#!/usr/bin/env python
"""
pretrain_offline.py  (auto-generated from SeniorDesignBaselineCNN.ipynb)
Headless offline pretraining for the Stevens vGPU box.

    SDP_DATA_DIR     where the regenerated *_IV_Surface_Data_Final.csv live
    SDP_WEIGHTS_DIR  where CNN_{model}_In{1,2}_OutBase.pth get written
    SDP_MODEL        Heston | Bates | Bergomi | rBergomi | all
    SDP_SENTIMENT    0 (In1) | 1 (In2) | both
"""
import os
_DATA_DIR    = os.environ.get("SDP_DATA_DIR",    os.path.expanduser("~/offline_out"))
_WEIGHTS_DIR = os.environ.get("SDP_WEIGHTS_DIR", os.path.expanduser("~/offline_models"))
_MODEL_ENV   = os.environ.get("SDP_MODEL", "all")
_SENT_ENV    = os.environ.get("SDP_SENTIMENT", "both")
os.makedirs(_WEIGHTS_DIR, exist_ok=True)
print(f"[cfg] data={_DATA_DIR}  weights={_WEIGHTS_DIR}  "
      f"model={_MODEL_ENV}  sentiment={_SENT_ENV}")
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

def load_and_prepare_data(csv_path, use_sentiment_input=True, predict_sentiment=True, grid_shape=(8, 11), batch_size=64):
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    starting_rows = len(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_rows = starting_rows - len(df)
    if dropped_rows > 0:
        print(f"WARNING: Dropped {dropped_rows} corrupted rows (NaN/Inf) from the dataset!")

    # 1. Isolate Columns
    iv_cols = [col for col in df.columns if col.startswith('IV_T')]
    ss_cols = [col for col in df.columns if col.startswith('SS_T')]

    # Base params vs Sentiment params (Assuming sentiment params end in '_Y' or '_JY' or '_S'/'_V' based on your dictionary)
    sentiment_param_keys = ['kappa_Y', 'theta_Y', 'sigma_Y', 'lambd_Y', 'mu_JY', 'sigma_JY', 'beta_S', 'beta_V', 'rho_SY']

    all_param_cols = [col for col in df.columns if not col.startswith('IV_T') and not col.startswith('SS_T')]

    if predict_sentiment:
        target_cols = all_param_cols
    else:
        target_cols = [col for col in all_param_cols if col not in sentiment_param_keys]

    print(f"Targeting {len(target_cols)} Parameters: {target_cols}")

    # 2. Build Inputs (X)
    X_iv = df[iv_cols].values.reshape(-1, 1, grid_shape[0], grid_shape[1])

    if use_sentiment_input:
        X_ss = df[ss_cols].values.reshape(-1, 1, grid_shape[0], grid_shape[1])
        X_surfaces = np.concatenate([X_iv, X_ss], axis=1) # Shape: (N, 2, 8, 11)
        print("Input: 2 Channels (IV + Synthetic Sentiment)")
    else:
        X_surfaces = X_iv # Shape: (N, 1, 8, 11)
        print("Input: 1 Channel (IV Only)")

    # 3. Build Targets (Y)
    Y_raw = df[target_cols].values

    # 4. Tensors & Datasets
    X_tensor = torch.tensor(X_surfaces, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Standardize inputs per channel
    train_tensors = train_dataset.dataset.tensors[0][train_dataset.indices]
    X_train_mean = train_tensors.mean(dim=(0, 2, 3), keepdim=True)
    X_train_std = train_tensors.std(dim=(0, 2, 3), keepdim=True)
    X_train_std[X_train_std == 0] = 1e-7

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, target_cols, X_train_mean, X_train_std

import torch
import torch.nn as nn

class VolatilitySurfaceCNN(nn.Module):
    def __init__(self, param_bounds, use_sentiment_input=True):
        super(VolatilitySurfaceCNN, self).__init__()

        # Toggle 1: Output size based on whether we pass just base bounds or base + sentiment bounds
        num_parameters = len(param_bounds)

        # Toggle 2: Input channels (2 if using sentiment scores, 1 if IV only)
        in_channels = 2 if use_sentiment_input else 1

        self.register_buffer('param_mins', torch.tensor([b[0] for b in param_bounds], dtype=torch.float32))
        self.register_buffer('param_maxs', torch.tensor([b[1] for b in param_bounds], dtype=torch.float32))

        ranges = self.param_maxs - self.param_mins
        ranges[ranges == 0] = 1e-7
        self.register_buffer('param_ranges', ranges)

        self.conv_block = nn.Sequential(
            # Dynamically set input channels based on the toggle
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        self.flattened_size = 32 * 8 * 11

        self.dense_block = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),

            # Dynamically set output size based on target parameters
            nn.Linear(128, num_parameters),
            nn.Sigmoid()
        )

    def forward(self, iv_surface):
        features = self.conv_block(iv_surface)
        features_flat = features.view(features.size(0), -1)

        out_0_to_1 = self.dense_block(features_flat)
        real_world_params = (out_0_to_1 * self.param_ranges) + self.param_mins

        return out_0_to_1, real_world_params

from pandas.core import base
import os
import time
import gc
import torch
import torch.nn as nn

# ==========================================
# MASTER CONFIGURATION
# ==========================================
base_models = (["Heston","Bates","Bergomi","rBergomi"]
               if _MODEL_ENV.lower() == "all" else [_MODEL_ENV])
for BASE_MODEL in base_models:
  SENTIMENT_MODEL = "OUJumps"
  csv_filename = os.path.join(_DATA_DIR, f'{BASE_MODEL}_IV_Surface_Data_Final.csv')

  SAVE_DIR = _WEIGHTS_DIR
  os.makedirs(SAVE_DIR, exist_ok=True)

  # The 4 experiments
  # PATCHED: default to the two OutBase variants, which are the ones the
  # fine-tuning stage actually loads (CNN_{model}_In{1,2}_OutBase.pth).
  # Set SDP_EXPERIMENTS=all to also train the OutSent variants.
  _all_experiments = [
      {"use_sent_in": False, "pred_sent_out": False},   # In1 OutBase
      {"use_sent_in": False, "pred_sent_out": True},    # In1 OutSent
      {"use_sent_in": True,  "pred_sent_out": False},   # In2 OutBase
      {"use_sent_in": True,  "pred_sent_out": True},    # In2 OutSent
  ]
  experiments = (_all_experiments
                 if os.environ.get("SDP_EXPERIMENTS","base").lower() == "all"
                 else [e for e in _all_experiments if not e["pred_sent_out"]])

  base_ranges = {
      "Bergomi":  {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "beta":[10.0,0.0]},
      "rBergomi": {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "H":[0.475,0.025]},
      "Heston":   {"kappa":[2.9,0.1], "theta":[0.14,0.01], "sigma":[0.7,0.1], "v0":[0.14,0.01], "rho_SV":[0.8,-0.9]},
      "Bates":    {"kappa":[3.0,0.1], "theta":[0.15,0.01], "sigma":[0.8,0.1], "v0":[0.15,0.01], "rho_SV":[-0.1,-0.9],
                  "lambdaJ":[2.0,0.0], "muJ":[0.05,-0.2], "sigmaJ":[0.20,0.01]}
  }

  sentiment_ranges = {
      "OUJumps": {
          "kappa_Y": [15.0, 5.0], "theta_Y": [0.0, 0.0], "sigma_Y": [0.6, 0.2],
          "lambd_Y": [38.0, 12.0], "mu_JY": [1.0, -0.5], "sigma_JY":[1.5, 0.5],
          "beta_S":  [0.15, 0.05], "beta_V": [0.09, -0.10], "rho_SY": [0.4, 0.3]
      }
  }


  def run_experiment(config):
      USE_IN = config["use_sent_in"]
      PRED_OUT = config["pred_sent_out"]

      channels = 2 if USE_IN else 1
      out_type = 'Full' if PRED_OUT else 'Base'

      base_filename = f"CNN_{BASE_MODEL}_In{channels}_Out{out_type}"
      full_model_path = os.path.join(SAVE_DIR, f"{base_filename}.pt")
      state_dict_path = os.path.join(SAVE_DIR, f"{base_filename}.pth")

      print("\n" + "="*60)
      print(f"STARTING EXPERIMENT: {base_filename}")
      print(f"Saving to: {SAVE_DIR}")
      print("="*60)

      train_loader, val_loader, test_loader, param_names, X_mean, X_std = load_and_prepare_data(
          csv_filename,
          use_sentiment_input=USE_IN,
          predict_sentiment=PRED_OUT
      )
      X_mean, X_std = X_mean.to(device), X_std.to(device)

      combined_lookup = {}
      combined_lookup.update(base_ranges.get(BASE_MODEL, {}))
      if PRED_OUT:
          combined_lookup.update(sentiment_ranges.get(SENTIMENT_MODEL, {}))

      combined_bounds = []
      for param in param_names:
          val1, val2 = combined_lookup[param]
          combined_bounds.append((min(val1, val2), max(val1, val2)))

      model = VolatilitySurfaceCNN(
          param_bounds=combined_bounds,
          use_sentiment_input=USE_IN
      ).to(device)

      param_mins = model.param_mins.to(device)
      param_maxs = model.param_maxs.to(device)

      EPOCHS = 1000
      PATIENCE = 30
      best_val_loss = float('inf')
      patience_counter = 0

      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

      for epoch in range(EPOCHS):
          model.train()
          train_loss = 0.0

          for X_batch, Y_batch_real in train_loader:
              X_batch, Y_batch_real = X_batch.to(device), Y_batch_real.to(device)

              X_batch_scaled = (X_batch - X_mean) / X_std
              Y_batch_0_to_1 = (Y_batch_real - param_mins) / model.param_ranges

              optimizer.zero_grad()
              preds_0_to_1, _ = model(X_batch_scaled)

              loss = criterion(preds_0_to_1, Y_batch_0_to_1)
              loss.backward()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
              optimizer.step()
              train_loss += loss.item()

          model.eval()
          val_loss = 0.0
          with torch.no_grad():
              for X_val, Y_val_real in val_loader:
                  X_val, Y_val_real = X_val.to(device), Y_val_real.to(device)

                  X_val_scaled = (X_val - X_mean) / X_std
                  Y_val_0_to_1 = (Y_val_real - param_mins) / model.param_ranges

                  preds_0_to_1_val, _ = model(X_val_scaled)
                  v_loss = criterion(preds_0_to_1_val, Y_val_0_to_1)
                  val_loss += v_loss.item()

          avg_train_loss = train_loss / len(train_loader)
          avg_val_loss = val_loss / len(val_loader)

          scheduler.step(avg_val_loss)

          if avg_val_loss < best_val_loss:
              best_val_loss = avg_val_loss
              patience_counter = 0

              torch.save(model, full_model_path)
              torch.save(model.state_dict(), state_dict_path)
          else:
              patience_counter += 1

          if (epoch + 1) % 10 == 0 or epoch == 0:
              print(f"  Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Patience: {patience_counter}/{PATIENCE}")

          if patience_counter >= PATIENCE:
              print(f"\n  [!] Early stopping at Epoch {epoch+1}. Best Val Loss: {best_val_loss:.5f}")
              break

      print(f"Saved Full PyTorch Model to: {full_model_path}")
      print(f"Saved State Dictionary to: {state_dict_path}")

  for exp_config in experiments:
      run_experiment(exp_config)

      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()

      time.sleep(2)

from pandas.core import base
import os
import time
import gc
import torch
import torch.nn as nn

# ===================================================================
# [PATCHED OUT] Duplicate of the training loop above - identical CSV,
# identical SAVE_DIR. Running both would train every model twice.
# ===================================================================

# # ==========================================
# # MASTER CONFIGURATION
# # ==========================================
# base_models = ["Bergomi","rBergomi"]
# for BASE_MODEL in base_models:
#   SENTIMENT_MODEL = "OUJumps"
#   csv_filename = os.path.join(_DATA_DIR, f'{BASE_MODEL}_IV_Surface_Data_Final.csv')

#   SAVE_DIR = _WEIGHTS_DIR
#   os.makedirs(SAVE_DIR, exist_ok=True)

#   # The 4 experiments
#   experiments = [
#       {"use_sent_in": False, "pred_sent_out": False},
#       {"use_sent_in": False, "pred_sent_out": True},
#       {"use_sent_in": True,  "pred_sent_out": False},
#       {"use_sent_in": True,  "pred_sent_out": True}
#   ]

#   base_ranges = {
#       "Bergomi":  {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "beta":[10.0,0.0]},
#       "rBergomi": {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "H":[0.475,0.025]},
#       "Heston":   {"kappa":[2.9,0.1], "theta":[0.14,0.01], "sigma":[0.7,0.1], "v0":[0.14,0.01], "rho_SV":[0.8,-0.9]},
#       "Bates":    {"kappa":[3.0,0.1], "theta":[0.15,0.01], "sigma":[0.8,0.1], "v0":[0.15,0.01], "rho_SV":[-0.1,-0.9],
#                   "lambdaJ":[2.0,0.0], "muJ":[0.05,-0.2], "sigmaJ":[0.20,0.01]}
#   }

#   sentiment_ranges = {
#       "OUJumps": {
#           "kappa_Y": [15.0, 5.0], "theta_Y": [0.0, 0.0], "sigma_Y": [0.6, 0.2],
#           "lambd_Y": [38.0, 12.0], "mu_JY": [1.0, -0.5], "sigma_JY":[1.5, 0.5],
#           "beta_S":  [0.15, 0.05], "beta_V": [0.09, -0.10], "rho_SY": [0.4, 0.3]
#       }
#   }


#   def run_experiment(config):
#       USE_IN = config["use_sent_in"]
#       PRED_OUT = config["pred_sent_out"]

#       channels = 2 if USE_IN else 1
#       out_type = 'Full' if PRED_OUT else 'Base'

#       base_filename = f"CNN_{BASE_MODEL}_In{channels}_Out{out_type}"
#       full_model_path = os.path.join(SAVE_DIR, f"{base_filename}.pt")
#       state_dict_path = os.path.join(SAVE_DIR, f"{base_filename}.pth")

#       print("\n" + "="*60)
#       print(f"STARTING EXPERIMENT: {base_filename}")
#       print(f"Saving to: {SAVE_DIR}")
#       print("="*60)

#       train_loader, val_loader, test_loader, param_names, X_mean, X_std = load_and_prepare_data(
#           csv_filename,
#           use_sentiment_input=USE_IN,
#           predict_sentiment=PRED_OUT
#       )
#       X_mean, X_std = X_mean.to(device), X_std.to(device)

#       combined_lookup = {}
#       combined_lookup.update(base_ranges.get(BASE_MODEL, {}))
#       if PRED_OUT:
#           combined_lookup.update(sentiment_ranges.get(SENTIMENT_MODEL, {}))

#       combined_bounds = []
#       for param in param_names:
#           val1, val2 = combined_lookup[param]
#           combined_bounds.append((min(val1, val2), max(val1, val2)))

#       model = VolatilitySurfaceCNN(
#           param_bounds=combined_bounds,
#           use_sentiment_input=USE_IN
#       ).to(device)

#       param_mins = model.param_mins.to(device)
#       param_maxs = model.param_maxs.to(device)

#       EPOCHS = 1000
#       PATIENCE = 30
#       best_val_loss = float('inf')
#       patience_counter = 0

#       criterion = nn.MSELoss()
#       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

#       for epoch in range(EPOCHS):
#           model.train()
#           train_loss = 0.0

#           for X_batch, Y_batch_real in train_loader:
#               X_batch, Y_batch_real = X_batch.to(device), Y_batch_real.to(device)

#               X_batch_scaled = (X_batch - X_mean) / X_std
#               Y_batch_0_to_1 = (Y_batch_real - param_mins) / model.param_ranges

#               optimizer.zero_grad()
#               preds_0_to_1, _ = model(X_batch_scaled)

#               loss = criterion(preds_0_to_1, Y_batch_0_to_1)
#               loss.backward()

#               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#               optimizer.step()
#               train_loss += loss.item()

#           model.eval()
#           val_loss = 0.0
#           with torch.no_grad():
#               for X_val, Y_val_real in val_loader:
#                   X_val, Y_val_real = X_val.to(device), Y_val_real.to(device)

#                   X_val_scaled = (X_val - X_mean) / X_std
#                   Y_val_0_to_1 = (Y_val_real - param_mins) / model.param_ranges

#                   preds_0_to_1_val, _ = model(X_val_scaled)
#                   v_loss = criterion(preds_0_to_1_val, Y_val_0_to_1)
#                   val_loss += v_loss.item()

#           avg_train_loss = train_loss / len(train_loader)
#           avg_val_loss = val_loss / len(val_loader)

#           scheduler.step(avg_val_loss)

#           if avg_val_loss < best_val_loss:
#               best_val_loss = avg_val_loss
#               patience_counter = 0

#               torch.save(model, full_model_path)
#               torch.save(model.state_dict(), state_dict_path)
#           else:
#               patience_counter += 1

#           if (epoch + 1) % 10 == 0 or epoch == 0:
#               print(f"  Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Patience: {patience_counter}/{PATIENCE}")

#           if patience_counter >= PATIENCE:
#               print(f"\n  [!] Early stopping at Epoch {epoch+1}. Best Val Loss: {best_val_loss:.5f}")
#               break

#       print(f"Saved Full PyTorch Model to: {full_model_path}")
#       print(f"Saved State Dictionary to: {state_dict_path}")

#   for exp_config in experiments:
#       run_experiment(exp_config)

#       gc.collect()
#       if torch.cuda.is_available():
#           torch.cuda.empty_cache()

#       time.sleep(2)

# best_model_path="/content/BatesNNWeights.pth"
# # 2. Save the model's state dictionary (the learned weights)
# torch.save(model.state_dict(), best_model_path)

# import pandas as pd

# def clean_spx_dates_and_days(filename):
#     """
#     Reads a CSV file, converts date columns to datetime,
#     and extracts the day of the month into new columns.
#     """
#     # Load the data
#     df = pd.read_csv(filename)

#     # Define date columns to process
#     date_cols = ['Trade Date', 'Expiry']

#     for col in date_cols:
#         if col in df.columns:
#             # 1. Convert to pandas datetime
#             df[col] = pd.to_datetime(df[col], errors='coerce')

#             # 2. Extract the day component (1-31)
#             # This creates new columns: 'Trade Date_Day' and 'Expiry_Day'
#             df[f'{col}_Day'] = df[col].dt.day

#     # Save the cleaned file
#     output_filename = filename.replace('.csv', '_with_days.csv')
#     df.to_csv(output_filename, index=False)

#     print(f"Successfully processed: {output_filename}")
#     return df

# # Usage: Plug in your filename here
# for year in range(2010, 2025):
#   input_file = f'<DATA_DIR>/cleanerSPXData_{year}.csv'
#   cleaned_df = clean_spx_dates_and_days(input_file)

#   # Display a sample of the new columns
#   print(cleaned_df[['Trade Date', 'Trade Date_Day', 'Expiry', 'Expiry_Day']].head())
#   print(cleaned_df.head())

# import pandas as pd
# import numpy as np
# from scipy.interpolate import Rbf
# import os

# def update_master_cnn_data(source_csv, master_csv='CNNData.csv'):
#     # 1. Check if source file exists
#     if not os.path.exists(source_csv):
#         print(f"--- Skipping: {source_csv} (File not found) ---")
#         return

#     print(f"Processing: {source_csv}...")

#     # 2. Load the Master Data to align the grid
#     try:
#         master_df = pd.read_csv(master_csv)
#         master_df['Trade Date'] = pd.to_datetime(master_df['Trade Date'])

#         lm_cols = [col for col in master_df.columns if col.startswith('LM_')]
#         lm_grid = sorted([float(col.split('_')[1]) for col in lm_cols])
#         tau_grid = sorted(master_df['Tau'].unique())
#     except Exception as e:
#         print(f"Error loading {master_csv}: {e}")
#         return

#     # 3. Load new source data
#     source_df = pd.read_csv(source_csv, low_memory=False)
#     source_df['Trade Date'] = pd.to_datetime(source_df['Trade Date'])
#     clean_source = source_df.dropna(subset=['LogMoneyness', 'Tau', 'Implied Volatility'])

#     if clean_source.empty:
#         print(f"No valid data in {source_csv}")
#         return

#     # 4. Interpolate new data onto Master Grid
#     new_rows = []
#     LM_mesh, T_mesh = np.meshgrid(lm_grid, tau_grid)
#     unique_dates = sorted(clean_source['Trade Date'].unique())

#     for date in unique_dates:
#         day_data = clean_source[clean_source['Trade Date'] == date]
#         if len(day_data) < 15: continue

#         try:
#             rbf = Rbf(day_data['LogMoneyness'], day_data['Tau'],
#                       day_data['Implied Volatility'], function='thin_plate')
#             iv_interp = rbf(LM_mesh, T_mesh)

#             for i, t_val in enumerate(tau_grid):
#                 row = {'Trade Date': date, 'Tau': t_val}
#                 for j, lm_val in enumerate(lm_grid):
#                     match_col = [c for c in lm_cols if abs(float(c.split('_')[1]) - lm_val) < 1e-5][0]
#                     row[match_col] = iv_interp[i, j]
#                 new_rows.append(row)
#         except:
#             continue

#     if not new_rows:
#         print(f"No new surfaces generated for {source_csv}")
#         return

#     # 5. Merge, Sort, and Deduplicate
#     new_data_df = pd.DataFrame(new_rows)
#     combined_df = pd.concat([master_df, new_data_df], ignore_index=True)
#     combined_df['Trade Date'] = pd.to_datetime(combined_df['Trade Date'])
#     combined_df = combined_df.sort_values(by=['Trade Date', 'Tau'])
#     combined_df = combined_df.drop_duplicates(subset=['Trade Date', 'Tau'], keep='first')

#     # 6. Save
#     combined_df.to_csv(master_csv, index=False)
#     print(f"Success! {master_csv} updated. Current total rows: {len(combined_df)}")

# # --- RUNNING THE LOOP ---
# # Make sure these paths match exactly where your files are stored in Google Drive
# base_path = '<DATA_DIR>/'
# master_path = '<DATA_DIR>/CNNData.csv'

# for year in range(2010, 2025):
#     file_to_process = f'{base_path}cleanerSPXData_{year}.csv' # Or your specific naming convention
#     update_master_cnn_data(file_to_process, master_path)

