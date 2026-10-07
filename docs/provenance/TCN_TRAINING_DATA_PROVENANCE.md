# TCN 16F / stride4 training-data provenance

The original per-environment `processed_3d_win16_stride4` NPZ intermediates are not bundled. The bundled training loader, repeat01 config, split and unified source data nevertheless provide a reproducible reconstruction path.

Evidence:
- `reference/repeat01_config/01_tcn/config.json` records window=16, stride=4 and clip counts 23,962 / 4,924 / 5,897.
- `reference/training_code/tcn/data_st.py` requires stride4 file naming and validates `start_frame` offsets modulo 4.
- `tools/rebuild_tcn_stride4_from_unified.py` reconstructs 16F/stride4 clips from `unified_3d_all_envs.npz` using the last-frame label rule and reproduces the same split clip counts.

Limitation: this establishes the clip-boundary/label/split provenance but does not assert byte-for-byte identity with unavailable historical intermediate NPZ files.
