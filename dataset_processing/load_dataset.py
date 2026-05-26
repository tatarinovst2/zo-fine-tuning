"""Load a dataset from HuggingFace and save as JSONL files for train/val/test splits."""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from datasets import load_dataset
from transformers import AutoTokenizer

from constants import ROOT_DIR


class DatasetInfoDict(TypedDict, total=False):
    """TypedDict for dataset information used in DATASET_OPTIONS."""

    dataset_link: str
    dataset_path: str
    balance: bool
    available_splits: list[str]
    subset: str


DATASET_OPTIONS: dict[str, DatasetInfoDict] = {
    "boolq": {
        "dataset_link": "google/boolq",
        "dataset_path": "data/boolq",
        "balance": True,
        "available_splits": ["train", "validation"]
    },
    "wic": {
        "dataset_link": "Deehan1866/WiC",
        "dataset_path": "data/wic",
        "balance": True,
        "available_splits": ["train", "validation", "test"]
    },
    "emotion": {
        "dataset_link": "dair-ai/emotion",
        "dataset_path": "data/emotion",
        "balance": True,
        "available_splits": ["train", "validation", "test"],
        "subset": "split"
    },
    "mnli": {
        "dataset_link": "nyu-mll/multi_nli",
        "dataset_path": "data/mnli",
        "balance": True,
        "available_splits": ["train", "validation_matched", "validation_mismatched"]
    },
    "samsum": {
        "dataset_link": "knkarthick/samsum",
        "dataset_path": "data/samsum",
        "balance": False,
        "available_splits": ["train", "validation", "test"]
    },
    "wmt14_cs_en": {
        "dataset_link": "wmt/wmt14",
        "dataset_path": "data/wmt14_cs_en",
        "balance": False,
        "available_splits": ["train", "validation", "test"],
        "subset": "cs-en"
    },
    "tabmwp": {
        "dataset_link": "TableSenseAI/TabMWP",
        "dataset_path": "data/tabmwp",
        "balance": False,
        "available_splits": ["train", "test"]
    },
    "math": {
        "dataset_link": "qwedsacf/competition_math",
        "dataset_path": "data/math",
        "balance": False,
        "available_splits": ["train"]
    }
}

TMP_DIR = Path(ROOT_DIR) / "tmp"


def get_hf_token() -> str | None:
    """
    Read HuggingFace token from the file.

    :return: The token string if the file exists, otherwise None.
    """
    token_path = Path(ROOT_DIR) / "HF_TOKEN.txt"
    if not token_path.exists():
        return None
    return token_path.read_text(encoding="utf-8").strip()


def ensure_git_dataset_repo(repo_id: str) -> Path:
    """
    Ensure that the given HuggingFace dataset is cloned locally and return the path to it.

    :param repo_id: The HuggingFace repository ID, e.g. "TableSenseAI/TabMWP".
    :return: Path to the local clone of the repository.
    """
    base_dir = TMP_DIR / "hf_datasets_git"
    base_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = base_dir / repo_id.replace("/", "__")
    if repo_dir.exists():
        return repo_dir
    if not (repo_dir / ".git").exists():
        token = get_hf_token()
        if token:
            url = f"https://{token}@huggingface.co/datasets/{repo_id}"
        else:
            url = f"https://huggingface.co/datasets/{repo_id}"
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(repo_dir)])
    return repo_dir


def read_tabmwp_table_tsv(context: dict, repo_id: str) -> str:
    """
    Read the TSV table content for a TabMWP entry.

    :param context: The "context" field from a TabMWP entry.
    :param repo_id: The HuggingFace repository ID where the TSV files are located.
    :raises RuntimeError: If the "tsv" path is not found in the context.
    :return: The content of the TSV file as a string.
    """
    rel = context.get("tsv")
    if not rel:
        raise RuntimeError("No TSV path in context")
    repo_root = ensure_git_dataset_repo(repo_id)
    return (repo_root / rel).read_text(encoding="utf-8")


def get_prompt_and_answer(entry: dict, name: str) -> dict[str, str]:  # pylint: disable=too-many-return-statements
    """
    Format an entry from the dataset into a prompt and answer pair according to the dataset name.

    :param entry: A single entry from the dataset.
    :param name: The name of the dataset (e.g., "boolq", "wic", "emotion", "samsum", "tabmwp").
    :raises NotImplementedError: If the dataset name is not recognized.
    :return: A dictionary with "prompt" and "answer" keys.
    """
    def emotion_label_to_answer(label: int) -> str:
        mapping = {
            0: "Sadness",
            1: "Joy",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }
        return mapping.get(label, "unknown")

    if name == "boolq":
        return {
            "prompt": f"Answer the question based on the information from the context.\n"
                      f"Context: {entry['passage']}\n"
                      f"Question: {entry['question'].capitalize()}?\n"
                      f"Answer with Yes or No.\nAnswer:",
            "answer": "Yes" if entry["answer"] else "No"
        }

    if name == "wic":
        return {
            "prompt": f"Does the word '{entry['phrase1']}' "
                      f"have the same meaning in these two sentences?\n"
                      f"Sentence 1: {entry['sentence1']}\nSentence 2: {entry['sentence2']}\n"
                      f"Answer with Yes or No.\nAnswer:",
            "answer": "Yes" if entry["label"] == 1 else "No"
        }

    if name == "emotion":
        return {
            "prompt": f"What emotion is expressed in this text?\n"
                      f"Text: {entry['text'].capitalize()}\n"
                      f"Answer with Sadness, Joy, Love, Anger, Fear or Surprise.\nAnswer:",
            "answer": emotion_label_to_answer(entry["label"])
        }

    if name == "samsum":
        return {
            "prompt": f"Summarize the following conversation:\n{entry['dialogue']}\nSummary:",
            "answer": entry["summary"].strip()
        }

    if name == "wmt14_cs_en":
        source = entry["translation"]["cs"].strip()
        target = entry["translation"]["en"].strip()

        return {
            "prompt": (
                f"Translate the following sentence from Czech to English.\n"
                f"Czech: {source}\n"
                f"English:"
            ),
            "answer": target
        }

    if name == "tabmwp":
        table_text = read_tabmwp_table_tsv(entry["context"], "TableSenseAI/TabMWP")
        question = entry["utterance"].strip()
        answer = str(entry["target_value"]).strip()
        prompt = (f"Using the following TSV table, answer the question.\nTSV table:\n{table_text}\n"
                  f"Question: {question}\nAnswer:")
        return {
            "prompt": prompt,
            "answer": answer
        }

    if name == "mnli":
        label_map = {
            0: "Entailment",
            1: "Neutral",
            2: "Contradiction"
        }

        return {
            "prompt": (
                f"Determine the relationship between the premise and the hypothesis.\n"
                f"Premise: {entry['premise']}\n"
                f"Hypothesis: {entry['hypothesis']}\n"
                f"Answer with Entailment, Neutral, or Contradiction.\n"
                f"Answer:"
            ),
            "answer": label_map[entry["label"]]
        }

    if name == "math":
        solution = entry["solution"].strip()

        return {
            "prompt": (
                "Solve the following math problem step by step.\n"
                "Put the final answer inside \\boxed{}.\n\n"
                f"Problem: {entry['problem'].strip()}\n\n"
                "Solution:"
            ),
            "answer": solution
        }

    raise NotImplementedError(f"Please implement get_prompt_and_answer for {name}")


def sample_data(records: list, limit: int | None, balance: bool) -> list:
    """
    Sample a subset of records with an optional balance across answer categories.

    :param records: A list of records, where record is a dictionary containing an "answer" key.
    :param limit: The maximum number of records to return. If None, return all records.
    :param balance: If True, attempt to balance the sample across different answer categories.
    :return: A list of sampled records, with length at most 'limit' and possibly balanced.
    """
    if limit is None or limit >= len(records):
        return records.copy()

    if not balance:
        return random.sample(records, limit)

    buckets: dict[str, list[dict]] = {}
    for record in records:
        if record["answer"] not in buckets:
            buckets[record["answer"]] = []
        buckets[record["answer"]].append(record)

    per = limit // len(buckets)
    picked = []
    for bucket in buckets.values():
        k = per if per <= len(bucket) else len(bucket)
        picked.extend(random.sample(bucket, k))
    need = limit - len(picked)
    if need > 0:
        leftovers = [r for r in records if r not in picked]
        picked.extend(random.sample(leftovers, min(need, len(leftovers))))
    return picked


def write_jsonl(records: list[dict], path: Path) -> None:
    """
    Write a list of records to a JSONL file.

    :param records: A list of dictionaries, each representing a record to write.
    :param path: The file path where the JSONL output should be written.
    """
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset preparation.

    :return: An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test jsonl for a HF dataset."
    )
    parser.add_argument("--dataset", required=True,
                        help="One of: " + ", ".join(DATASET_OPTIONS.keys()))
    parser.add_argument("--train_limit", type=int, default=1000)
    parser.add_argument("--val_limit", type=int, default=500)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=96)

    return parser.parse_args()


def load_raw_dataset(info: DatasetInfoDict, tmp_dir: Path) -> dict[str, list[dict]]:
    """
    Load the raw dataset from HuggingFace and return a dictionary of splits to lists of examples.

    :param info: A dictionary containing info like "dataset_link", "available_splits", "subset".
    :param tmp_dir: A temporary directory to use for caching the dataset.
    :return: A dictionary mapping split names (e.g., "train", "test" etc.) to lists of examples.
    """
    load_kwargs = {}

    if "subset" in info:
        load_kwargs["name"] = info["subset"]

    ds = load_dataset(info["dataset_link"], cache_dir=str(tmp_dir), **load_kwargs)

    return {split: list(ds[split]) for split in info["available_splits"]}


def validate_dataset_out_dir(out_dir: Path) -> None:
    """
    If the directory exists from the beginning, prompt for deletion. Ensure a directory exists.

    :param out_dir: The path to the output directory where the prepared dataset will be saved.
    """
    if out_dir.exists():
        response = input(f"Remove {out_dir} and continue? [y/N]: ")
        if response.lower() == "y":
            shutil.rmtree(out_dir)
        else:
            print("Cancelled.")

            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
            sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)


def ensure_test_val_exist(raw: dict, name: str) -> dict:
    """
    Make sure both "validation" and "test" splits are present, by splitting if necessary.

    :param raw: A dictionary containing the raw dataset splits as lists of examples.
    :param name: The name of the dataset, which may require special handling (e.g., "mnli").
    :return: The modified raw dataset dictionary with guaranteed "validation" and "test" splits.
    """
    if name == "mnli":
        raw["validation"] = raw["validation_matched"]
        raw["test"] = raw["validation_matched"]
        raw["mismatched_test"] = raw["validation_mismatched"]

    if "train" in raw and "validation" not in raw and "test" not in raw:
        data = raw["train"]
        random.shuffle(data)

        n = len(data)

        train_end = int(n * 0.4)
        val_end = int(n * 0.6)

        raw["train"] = data[:train_end]
        raw["validation"] = data[train_end:val_end]
        raw["test"] = data[val_end:]

    if "validation" not in raw and "test" in raw:
        data = raw["test"]
        random.shuffle(data)
        half = len(data) // 2
        raw["validation"], raw["test"] = data[:half], data[half:]

    if "test" not in raw and "validation" in raw:
        data = raw["validation"]
        random.shuffle(data)
        half = len(data) // 2
        raw["test"], raw["validation"] = data[:half], data[half:]

    return raw


def main() -> None:
    """Run the dataset preparation process."""
    args = parse_args()

    random.seed(42)

    name = args.dataset.lower()
    if name not in DATASET_OPTIONS:
        print("Unknown dataset:", name)
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASET_OPTIONS[name]

    out_dir = Path(dataset_info["dataset_path"])
    validate_dataset_out_dir(out_dir)

    raw = load_raw_dataset(dataset_info, TMP_DIR)
    raw = ensure_test_val_exist(raw, name)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B",
                                              use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = ["train", "validation", "test"]

    if name == "mnli":
        splits.append("mismatched_test")

    for split in splits:
        examples = raw.get(split, [])
        processed = [get_prompt_and_answer(ex, name) for ex in examples]
        processed = [  # Filter by length
            r for r in processed
            if len(tokenizer.encode(r["prompt"])) <= args.max_input_length
            and len(tokenizer.encode(r["answer"])) <= args.max_output_length
        ]

        limit = {
            "train": args.train_limit,
            "validation": args.val_limit,
            "test": args.test_limit,
            "mismatched_test": args.test_limit
        }[split]

        sampled = sample_data(processed, limit, dataset_info.get("balance", False))
        fname = "val.jsonl" if split == "validation" else f"{split}.jsonl"
        write_jsonl(sampled, out_dir / fname)
        print(f"Wrote {len(sampled)} to {out_dir/fname}")

    print("Done.")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
