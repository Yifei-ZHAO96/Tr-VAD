### Dataset Generation for TIMIT Dataset

Functions:
- `add_silence.py`: Padding silence to the beginning and the end of each audio. Input required: `clean_file_path`, `sil_length`, `sr`.
- `augmentation.py`: Augment audio by adding noise with difference snrs. Input required: `clean_file_path`, `noise_file_path`, `sr`, `snr_`.
- `audio.py`: Other useful tools for audio data processing.

Use example: 
```bash
python data_gen.py '.../TIMIT/TRAIN' '.../AURORA' -sr 16000 -silence_pad 1 -snr "[-5, 0, 5, 10]"
```