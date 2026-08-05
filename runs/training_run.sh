#--------------------------------------------------------------------------------------------------
# DATASET DOWNLOAD AND TOKENIZER TRAINING

# The dataset is from HuggingFace Hub (karpathy/fineweb-edu-100b-shuffle)
# Download the first 8 shards of pretraining dataset (each shard contains ~250M characters)
uv run -m angstromchat.dataset -n 8

# Immediately kick off downloading more shards in the background while tokenizer trains
# The maximum total number of shards available in the entire dataset is 1822.
uv run -m angstromchat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!

# train tokenizer with 8 shards(8 * ~250M = ~2B characters)
uv run -m scripts.tokenizer_train

# evaluate tokenizer
uv run -m scripts.tokenizer_eval

#--------------------------------------------------------------------------------------------------
# BASE MODEL TRAINING

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# train base model


