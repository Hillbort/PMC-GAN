# PCM-GAN (Real Data)

This folder contains a standalone PCM-GAN implementation aligned with the paper description.

## What It Includes
- `data_synth.py`: real data loader and dataset wrapper
- `models.py`: Generator, Discriminator, mHC-Transformer blocks, Physics Mask
- `losses.py`: relativistic adversarial loss, gradient penalty, tail NLL, physical penalty
- `train.py`: training pipeline (KNN threshold estimation + adversarial training)
- `generate.py`: generate samples using real-data conditions
- `test.py`: load a trained checkpoint and generate samples

## Quick Start

Train (real data):
```bash
python -m pcm_gan.train --data_csv pcm_gan_data/ororiginal_data/CAISO_zone_1_.csv --resolution hourly
```

Generate:
```bash
python -m pcm_gan.generate --ckpt pcm_gan_runs/pcm_gan.pt --data_csv pcm_gan_data/ororiginal_data/CAISO_zone_1_.csv --num 16
```

Test (inference):
```bash
python -m pcm_gan.test --ckpt pcm_gan_runs/pcm_gan.pt --data_csv pcm_gan_data/ororiginal_data/CAISO_zone_1_.csv --num 16
```

## Web UI (visual testing)
Dependencies: `flask`, `matplotlib`

```bash
python pcm_gan_web/app.py
```
Then open `http://127.0.0.1:5000` in your browser.
Outputs are saved under `pcm_gan_out/` as `.npy` and daily `.csv` files.

## Notes
This implementation follows the *structure* described in the paper:
  - conditional generator with mHC-Transformer blocks
  - physics-informed hard masking
  - relativistic adversarial loss + dual gradient penalty
  - tail likelihood regularization (GPD NLL)
  - physical soft penalties
- `u(w)` is estimated with a KNN strategy over the historical library in each batch
  (95% quantile of neighbor risk metric values), avoiding any extra pretraining stage.
