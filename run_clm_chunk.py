#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
MODIFIED VERSION: Processes each example individually without cross-example concatenation/chunking.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
torch.autograd.set_detect_anomaly(True)
from datasets import load_dataset, load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.54.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "Each example will be truncated/padded to this length. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory containing datasets saved with save_to_disk (e.g., SCROLLS datasets). Should contain train/ and validation/ subdirectories."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.dataset_path is None:
            raise ValueError("Need either a dataset name, a training/validation file, or a dataset_path.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # TODO: Check if works
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # Or provide a dataset_path to load datasets saved with save_to_disk.
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_path is not None:
        # Load datasets from disk (e.g., SCROLLS datasets)
        from datasets import load_from_disk, DatasetDict
        logger.info(f"Loading datasets from disk: {data_args.dataset_path}")
        raw_datasets = DatasetDict()
        train_path = os.path.join(data_args.dataset_path, "train")
        val_path = os.path.join(data_args.dataset_path, "validation")
        if os.path.exists(train_path):
            raw_datasets["train"] = load_from_disk(train_path)
            logger.info(f"Loaded train dataset from {train_path}")
        if os.path.exists(val_path):
            raw_datasets["validation"] = load_from_disk(val_path)
            logger.info(f"Loaded validation dataset from {val_path}")
        if len(raw_datasets) == 0:
            raise ValueError(f"No train or validation datasets found at {data_args.dataset_path}")
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    if getattr(config, 'attention_type', None) == "newlucid":
        print("Copying K weights to W weights post-initialization...")
        with torch.no_grad():
            for layer in model.model.layers:
                attn_block = layer.self_attn
                attn_block.w_proj.weight.data.copy_(attn_block.k_proj.weight.data)
                if attn_block.w_proj.bias is not None:
                    attn_block.w_proj.bias.data.copy_(attn_block.k_proj.bias.data)

    # Set pad_token if not set (common for LLaMA tokenizers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad_token to eos_token: {tokenizer.eos_token}")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # Check if dataset has input/output format (SCROLLS) or just text
    has_input_output = "input" in column_names and "output" in column_names
    if has_input_output:
        logger.info("Detected input/output format dataset (e.g., SCROLLS). Will compute loss only on output tokens.")
        input_column_name = "input"
        output_column_name = "output"
    else:
        logger.info("Detected text-only format dataset. Will compute loss on all tokens.")
        text_column_name = "text" if "text" in column_names else column_names[0]

    # Determine block_size
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    logger.info(f"Using block_size={block_size} for tokenization with truncation and padding")

    # MODIFIED: Tokenize with padding and truncation to block_size
    # This processes each example individually without cross-example concatenation
    def tokenize_and_pad_function(examples):
        labels = []
        input_ids_list = []
        attention_mask_list = []
        total_tokens = 0
        total_pad_tokens = 0
        total_loss_tokens = 0
        total_input_tokens = 0

        if has_input_output:
            # SCROLLS format: separate input and output
            # Compute loss ONLY on output tokens, not on input tokens
            for inp, out in zip(examples[input_column_name], examples[output_column_name]):
                # Tokenize input and output separately (without adding special tokens)
                input_tokens = tokenizer(inp, add_special_tokens=False)["input_ids"]
                output_tokens = tokenizer(out, add_special_tokens=False)["input_ids"]

                # Add EOS token at the end so model learns to stop after answer
                output_tokens_with_eos = output_tokens + [tokenizer.eos_token_id]

                # IMPORTANT: Ensure we always have room for output tokens
                # If combined length exceeds block_size, truncate INPUT to make room for output
                if len(input_tokens) + len(output_tokens_with_eos) > block_size:
                    # Reserve space for output, truncate input
                    max_input_len = block_size - len(output_tokens_with_eos)
                    if max_input_len < 0:
                        # Output alone exceeds block_size, truncate output too
                        output_tokens_with_eos = output_tokens_with_eos[:block_size]
                        input_tokens = []
                    else:
                        input_tokens = input_tokens[:max_input_len]

                # Concatenate: input + output
                combined_tokens = input_tokens + output_tokens_with_eos

                # Ensure combined doesn't exceed block_size (safety check)
                if len(combined_tokens) > block_size:
                    combined_tokens = combined_tokens[:block_size]

                # Calculate how many tokens belong to input vs output in the combined sequence
                input_len = len(input_tokens)
                output_len = len(combined_tokens) - input_len

                # Pad to block_size
                num_padding = block_size - len(combined_tokens)
                padded_input_ids = combined_tokens + [tokenizer.pad_token_id] * num_padding
                attention_mask = [1] * len(combined_tokens) + [0] * num_padding

                # Create labels:
                # - Input portion: masked with -100 (no loss)
                # - Output portion: keep token ids (compute loss)
                # - Padding: masked with -100 (no loss)
                label = [-100] * input_len + combined_tokens[input_len:] + [-100] * num_padding

                input_ids_list.append(padded_input_ids)
                attention_mask_list.append(attention_mask)
                labels.append(label)

                # Track statistics
                total_tokens += block_size
                total_pad_tokens += num_padding
                total_input_tokens += input_len
                total_loss_tokens += output_len
        else:
            # Text-only format: compute loss on all tokens
            # Add EOS token to each example so model learns to generate EOS after completing the text
            texts_with_eos = [text + tokenizer.eos_token for text in examples[text_column_name]]

            # Tokenize with padding and truncation
            tokenizer_output = tokenizer(
                texts_with_eos,
                padding="max_length",
                truncation=True,
                max_length=block_size,
                return_tensors=None,  # Return lists, not tensors
            )

            for i, input_ids in enumerate(tokenizer_output["input_ids"]):
                label = input_ids.copy()
                attention_mask = tokenizer_output["attention_mask"][i]
                # Use attention mask to distinguish real tokens from padding
                # Real tokens (attention_mask=1): keep label
                # Padding tokens (attention_mask=0): mask with -100
                label = [token_id if attention_mask[j] == 1 else -100
                         for j, token_id in enumerate(label)]

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels.append(label)

                # Track statistics
                num_pad = input_ids.count(tokenizer.pad_token_id)
                num_loss = sum(1 for x in label if x != -100)
                total_tokens += len(input_ids)
                total_pad_tokens += num_pad
                total_loss_tokens += num_loss

        output = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels,
        }

        # Print diagnostics for first batch only
        if not hasattr(tokenize_and_pad_function, '_printed'):
            tokenize_and_pad_function._printed = True
            num_examples = len(output["input_ids"])
            logger.info("="*80)
            logger.info("PADDING VERIFICATION (First Batch)")
            logger.info("="*80)
            logger.info(f"Number of examples in batch: {num_examples}")
            logger.info(f"Block size (max_length): {block_size}")
            logger.info(f"Pad token ID: {tokenizer.pad_token_id}")
            logger.info(f"")
            logger.info(f"Batch statistics:")
            logger.info(f"  Total tokens: {total_tokens:,}")
            logger.info(f"  Total pad tokens: {total_pad_tokens:,}")
            logger.info(f"  Total real tokens: {total_tokens - total_pad_tokens:,}")
            if has_input_output:
                logger.info(f"  Total input tokens (masked): {total_input_tokens:,}")
                logger.info(f"  Total output tokens (loss computed): {total_loss_tokens:,}")
            else:
                logger.info(f"  Total tokens with loss computed: {total_loss_tokens:,}")
            logger.info(f"  Padding percentage: {100 * total_pad_tokens / total_tokens:.2f}%")

            if has_input_output:
                # For input/output: loss should only be on output tokens
                logger.info(f"  ✓ Loss computed on {total_loss_tokens:,} output tokens (input tokens masked)")
            else:
                # For text-only: loss should be on all real tokens
                if total_loss_tokens == (total_tokens - total_pad_tokens):
                    logger.info(f"  ✓ VERIFICATION PASSED: Loss computed on {total_loss_tokens:,} real tokens only")
                else:
                    logger.warning(f"  ✗ VERIFICATION FAILED: Mismatch in loss token count!")

            # Show details for first 3 examples
            logger.info(f"")
            logger.info(f"First {min(3, num_examples)} examples:")
            for idx in range(min(3, num_examples)):
                input_ids = output["input_ids"][idx]
                label_ids = output["labels"][idx]
                attn_mask = output["attention_mask"][idx]
                num_pad = input_ids.count(tokenizer.pad_token_id)
                num_real = len(input_ids) - num_pad
                num_loss = sum(1 for x in label_ids if x != -100)

                # Find where real content ends (last position with attention_mask=1)
                real_end_idx = max([i for i, m in enumerate(attn_mask) if m == 1], default=-1)

                logger.info(f"  Example {idx+1}:")
                logger.info(f"    Total tokens: {len(input_ids)}")
                logger.info(f"    Real tokens: {num_real}")
                logger.info(f"    Pad tokens: {num_pad}")
                logger.info(f"    Loss computed on: {num_loss}")
                logger.info(f"    Real content ends at position: {real_end_idx}")

                if has_input_output:
                    # Find input/output boundary (first position with non -100 label)
                    input_output_boundary = -1
                    for i, label in enumerate(label_ids):
                        if label != -100:
                            input_output_boundary = i
                            break
                    if input_output_boundary >= 0:
                        logger.info(f"    Input/output boundary at position: {input_output_boundary}")
                        logger.info(f"    Input tokens (masked): 0 to {input_output_boundary-1}")
                        logger.info(f"    Output tokens (loss): {input_output_boundary} to {real_end_idx}")

                        # Show around input/output boundary
                        if input_output_boundary >= 3:
                            io_start = max(0, input_output_boundary - 3)
                            io_end = min(len(input_ids), input_output_boundary + 4)
                            logger.info(f"    Input→Output boundary at pos {input_output_boundary}:")
                            logger.info(f"      input_ids[{io_start}:{io_end}]: {input_ids[io_start:io_end]}")
                            logger.info(f"      labels[{io_start}:{io_end}]:    {label_ids[io_start:io_end]}")

                logger.info(f"    First 10 input_ids: {input_ids[:10]}")
                logger.info(f"    First 10 labels:    {label_ids[:10]}")

                # Show around the boundary between real content and padding
                if real_end_idx >= 5:
                    boundary_start = max(0, real_end_idx - 5)
                    boundary_end = min(len(input_ids), real_end_idx + 6)
                    logger.info(f"    Boundary (real→pad) at pos {real_end_idx}:")
                    logger.info(f"      input_ids[{boundary_start}:{boundary_end}]: {input_ids[boundary_start:boundary_end]}")
                    logger.info(f"      labels[{boundary_start}:{boundary_end}]:    {label_ids[boundary_start:boundary_end]}")
                    logger.info(f"      attn_mask[{boundary_start}:{boundary_end}]:  {attn_mask[boundary_start:boundary_end]}")

                    # Check if the last real token is EOS
                    last_real_token = input_ids[real_end_idx]
                    last_real_label = label_ids[real_end_idx]
                    if last_real_token == tokenizer.eos_token_id:
                        if last_real_label == tokenizer.eos_token_id:
                            logger.info(f"      ✓ Real EOS at position {real_end_idx} is KEPT in labels (will learn to generate EOS)")
                        else:
                            logger.info(f"      ✗ Real EOS at position {real_end_idx} is MASKED (won't learn to stop!)")

                logger.info(f"    Last 10 input_ids:  {input_ids[-10:]}")
                logger.info(f"    Last 10 labels:     {label_ids[-10:]}")

            logger.info("="*80)

        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                tokenize_and_pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenizing and padding to {block_size}",
            )
        else:
            lm_datasets = raw_datasets.map(
                tokenize_and_pad_function,
                batched=True,
                remove_columns=column_names,
            )

    # NO group_texts step - each example stays as-is with padding/truncation

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
