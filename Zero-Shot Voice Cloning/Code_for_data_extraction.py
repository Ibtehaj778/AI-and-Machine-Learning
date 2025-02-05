from datasets import load_dataset
import torchaudio
from io import BytesIO
import os

# Define dataset parameters
DATASET_NAME = "speechbrain/LargeScaleASR"
DATASET_CONFIG = "small"  
SPLIT = "train"
BATCH_SIZE = 50000  # Number of new samples to download

# Define directories
SAVE_DIR = "/kaggle/working/audio_dataset"
LOG_FILE = "/kaggle/working/last_sample.txt"

# Create directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
# Load last index if exists
start_index = 100000

print(f"Resuming from sample {start_index}...")

# Load dataset in streaming mode
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT, streaming=True)

# Convert dataset to iterator
dataset_iter = iter(dataset)

# Skip previously downloaded samples
for _ in range(start_index):
    next(dataset_iter)  # Now this works!
# Start downloading new samples
for i, sample in enumerate(dataset_iter, start=start_index):
    if i >= start_index + BATCH_SIZE:
        break

    # Load and save audio
    audio_tensor, sample_rate = torchaudio.load(BytesIO(sample["wav"]["bytes"]))
    audio_path = os.path.join(SAVE_DIR, f"sample_{i}.wav")
    torchaudio.save(audio_path, audio_tensor, sample_rate)

    # Save transcript
    transcript_path = os.path.join(SAVE_DIR, f"sample_{i}.txt")
    with open(transcript_path, "w") as f:
        f.write(sample["text"])
          # Update log every 1000 samples
    if i % 1000 == 0:
        with open(LOG_FILE, "w") as f:
            f.write(str(i))
        print(f"Saved {i} samples...")

# Final log update
with open(LOG_FILE, "w") as f:
    f.write(str(i))

print(f"Finished downloading {BATCH_SIZE} new samples. Last saved index: {i}")
