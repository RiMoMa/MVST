set -euo pipefail

CSV=YOUR_REPO_ROOT/data/my_list.csv
OUT=YOUR_REPO_ROOT/save/my_features
CKPT16=YOUR_REPO_ROOT/save/16/.../best.pth
CKPT32=YOUR_REPO_ROOT/save/32/.../best.pth
CKPT64=YOUR_REPO_ROOT/save/64/.../best.pth
CKPT128=YOUR_REPO_ROOT/save/128/.../best.pth
CKPT256=YOUR_REPO_ROOT/save/256/.../best.pth

mkdir -p "$OUT"

python MVST/16/save_features.py --csv $CSV --data_split test \
  --batch_size 8 --resample_to 16000 --pad_seconds 10 \
  --n_mels 128 --n_fft 1024 --hop_length 320 \
  --patch_size 16 --pretrained_ckpt $CKPT16 --out_dir $OUT

python MVST/32/save_features.py --csv $CSV --data_split test \
  --batch_size 8 --resample_to 16000 --pad_seconds 10 \
  --n_mels 128 --n_fft 1024 --hop_length 320 \
  --patch_size 32 --pretrained_ckpt $CKPT32 --out_dir $OUT

python MVST/64/save_features.py --csv $CSV --data_split test \
  --batch_size 8 --resample_to 16000 --pad_seconds 10 \
  --n_mels 128 --n_fft 1024 --hop_length 320 \
  --patch_size 64 --pretrained_ckpt $CKPT64 --out_dir $OUT

python MVST/128/save_features.py --csv $CSV --data_split test \
  --batch_size 8 --resample_to 16000 --pad_seconds 10 \
  --n_mels 128 --n_fft 1024 --hop_length 320 \
  --patch_size 128 --pretrained_ckpt $CKPT128 --out_dir $OUT

python MVST/256/save_features.py --csv $CSV --data_split test \
  --batch_size 8 --resample_to 16000 --pad_seconds 10 \
  --n_mels 128 --n_fft 1024 --hop_length 320 \
  --patch_size 256 --pretrained_ckpt $CKPT256 --out_dir $OUT

python tools/fuse_mvst_views.py \
  --views $OUT/features_mvst_16.csv $OUT/features_mvst_32.csv $OUT/features_mvst_64.csv $OUT/features_mvst_128.csv $OUT/features_mvst_256.csv \
  --out $OUT/features_mvst_fused.csv

echo "[DONE] Fused at $OUT/features_mvst_fused.csv"
