# HW1

只保留三個程式：

- `src/data_process.py`：影像前處理與 `CachedImageDataset`。
- `src/model.py`：由你自行實作 VAE。
- `src/train.py`：訓練、checkpoint、loss 曲線、生成圖與 IS/FID。

從 `HW1/` 執行前處理（目前資料已處理好，不必重跑）：

```bash
python src/data_process.py
```

train/test 分開存成 `data/processed/train.pt`、`test.pt`。影像轉灰階、bilinear 縮放成 64 × 64 正方形，不做資料增強；Dataset 回傳 `[0, 1]` 的 float32 影像，不回傳標籤。重建快取可加 `--overwrite`，解析度可用 `--image-size` 調整。

從 `HW1/` 訓練。預設是 64×64、latent 128、Adam `1e-3`、batch 128、100 個 epoch、seed 42。Loss 是每張圖加總後再對 batch 平均的 BCE + KL，KL 權重為 1。

```bash
python src/train.py
```

每次未指定輸出路徑時，會新建 `outputs/vae_YYYYMMDD_HHMMSS/`，避免和舊的 CSV 或 checkpoint 混在一起。指定路徑時：

```bash
python src/train.py --output-dir outputs/vae_latent128_e100
```

該資料夾會有：

- `history.csv`、`loss.png`：每個 epoch 的 train/test total、重建、KL。
- `checkpoints/epoch_XXX.pt`：模型、optimizer、epoch 與 config。
- `samples/epoch_XXX.png`：每 10 個 epoch 及最後一個 epoch，由固定的 64 個標準常態 z 解碼成 8×8 圖。
- `reconstructions/epoch_XXX.png`：同一批 test 圖，上半是原圖、下半是重建。
- `metrics.json`：最後一個 checkpoint 的 IS/FID、真圖對照、抽樣索引、seed 與套件版本。

IS/FID 使用 pytorch-ignite 預設的 Inception-v3（ImageNet 1000 類機率）。真圖來自 train split 的 2000 張 seeded 隨機子集；假圖用同數量的標準常態 z 經 `decode` 生成。評分前把灰階 bilinear 放大到 299×299 並複製成 3 通道，值域維持 `[0, 1]`。同一設定下另有一組互斥的 2000 張真圖，用來算 real-vs-real FID 與真圖 IS。報告只比較這個設定下的分數。

短跑可加 `--epochs 1 --skip-eval`。第一次評分會下載約 104 MB 的 Inception 權重，請用 MPS 或 CUDA。

缺少套件時使用 `python -m pip install -r requirements.txt`。兩頁 NeurIPS 報告在 `report/hw1_report.pdf`。
