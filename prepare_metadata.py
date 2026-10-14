import hashlib
import json
import os
from pathlib import Path

import pandas as pd

# Use a Hugging Face mirror that can be reached from AutoDL
# This value must be set before Transformers loads the model files
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Store all input and output files in the RPC dataset directory
DATASET_ROOT = Path("/root/autodl-tmp/retail_product_checkout")

# Use the training annotation file because it contains the complete
# product category list and the original Chinese product metadata
ANNOTATION_PATH = DATASET_ROOT / "instances_train2019.json"

# Save the generated price table in the dataset directory
PRICE_OUTPUT_PATH = DATASET_ROOT / "product_prices.csv"

# Save the Chinese and English product name mapping in the same directory
NAME_OUTPUT_PATH = DATASET_ROOT / "product_name_mapping.csv"

# Use a local MarianMT model for Chinese to English translation
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"

# Store downloaded model files on the persistent AutoDL data volume
TRANSLATION_MODEL_CACHE = Path("/root/autodl-tmp/translation_models")

# Translate several short product names in one model call
TRANSLATION_BATCH_SIZE = 32

# Record the model name in the output file for reproducibility
TRANSLATION_SOURCE = "Helsinki-NLP opus-mt-zh-en"


# These ranges represent simulated retail prices in Chinese yuan
# Each product class has a different range to make the prices more realistic
# The prices are created only for the automatic checkout demonstration
# They are not real prices supplied by the RPC dataset
CLASS_PRICE_RANGES = {
    "tissue":           (3.0, 25.0),
    "dessert":          (3.0, 20.0),
    "drink":            (2.0, 15.0),
    "canned_food":      (5.0, 30.0),
    "seasoner":         (2.0, 20.0),
    "instant_noodles":  (3.0, 15.0),
    "puffed_food":      (3.0, 15.0),
    "chocolate":        (4.0, 30.0),
    "alcohol":          (8.0, 80.0),
    "instant_drink":    (5.0, 40.0),
    "milk":             (3.0, 25.0),
    "personal_hygiene": (5.0, 45.0),
    "candy":            (2.0, 20.0),
    "dried_food":       (5.0, 35.0),
    "dried_fruit":      (5.0, 40.0),
    "gum":              (2.0, 12.0),
    "stationery":       (1.0, 20.0),
}


def load_product_metadata():
    # Stop early when the required annotation file is missing
    # This usually means that the dataset path is incorrect
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file was not found: {ANNOTATION_PATH}"
        )

    # Load the COCO annotation data from the training JSON file
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    # This RPC specific field stores Chinese product names
    # It also stores barcodes and the English product class labels
    metadata_key = "__raw_Chinese_name_df"

    if metadata_key not in annotation_data:
        raise KeyError(
            f"The field '{metadata_key}' was not found in the JSON file."
        )

    # Convert the product metadata into a table for easier processing
    products = pd.DataFrame(annotation_data[metadata_key])

    # These fields are required by the checkout and detection system
    required_columns = {
        "category_id",
        "sku_name",
        "sku_class",
        "clas",
        "name",
        "code",
    }

    # Report missing fields before creating incorrect output files
    missing_columns = required_columns - set(products.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the fields needed by the later stages of the project
    products = products[
        [
            "category_id",
            "sku_name",
            "sku_class",
            "clas",
            "name",
            "code",
        ]
    ].copy()

    # Rename the original fields to make their meanings clearer
    products = products.rename(
        columns={
            "clas": "class_name_cn",
            "name": "product_name_cn",
            "code": "barcode",
        }
    )

    # Store category identifiers as integers for YOLO class lookup
    products["category_id"] = products["category_id"].astype(int)

    # Store barcodes as text so leading zeros are never removed
    products["barcode"] = products["barcode"].astype(str)

    # Keep the rows in the same order as the category identifiers
    products = (
        products.sort_values("category_id")
        .reset_index(drop=True)
    )

    # Every product must have a unique category identifier
    # Duplicate identifiers would cause incorrect model label lookup
    if products["category_id"].duplicated().any():
        raise ValueError("Duplicate category IDs were found.")

    # Every product in this dataset should also have a unique barcode
    if products["barcode"].duplicated().any():
        raise ValueError("Duplicate product barcodes were found.")

    return products


def generate_simulated_price(category_id, sku_class):
    # Make sure every product class has a configured price range
    if sku_class not in CLASS_PRICE_RANGES:
        raise ValueError(
            f"No price range is configured for class: {sku_class}"
        )

    min_price, max_price = CLASS_PRICE_RANGES[sku_class]

    # Combine the class name and category identifier into a stable key
    # The same product will always produce the same hash value
    hash_text = f"{sku_class}:{category_id}"
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()

    # Convert part of the hash into a value between zero and one
    # This value determines the position inside the class price range
    ratio = int(digest[:8], 16) / 0xFFFFFFFF

    # Place the product price inside its product class range
    raw_price = min_price + ratio * (max_price - min_price)

    # Round prices to the nearest half yuan
    # This produces simple values suitable for a checkout demonstration
    price = round(raw_price * 2) / 2

    # Keep the final value inside the configured range
    return max(min_price, min(price, max_price))


def create_price_table(products):
    # Copy the product table so the original metadata remains unchanged
    price_table = products.copy()

    # Generate one stable price for every specific product SKU
    # Products in the same large class can still have different prices
    price_table["unit_price"] = price_table.apply(
        lambda row: generate_simulated_price(
            category_id=row["category_id"],
            sku_class=row["sku_class"],
        ),
        axis=1,
    )

    # Record the currency used by the checkout system
    price_table["currency"] = "CNY"

    # Clearly mark the values as simulated rather than real retail prices
    price_table["price_source"] = "simulated_by_class_range"

    # Save with a byte order mark for better Chinese text support in Excel
    price_table.to_csv(
        PRICE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return price_table


def load_existing_translations():
    # Return an empty mapping when the output file does not exist
    # The script will translate every product in this case
    if not NAME_OUTPUT_PATH.exists():
        return {}

    # Load barcodes as text to preserve their original format
    existing = pd.read_csv(
        NAME_OUTPUT_PATH,
        dtype={"barcode": str},
    )

    # An existing file can only be reused when it contains these fields
    required_columns = {
        "category_id",
        "product_name_en",
        "translation_source",
    }

    if not required_columns.issubset(existing.columns):
        return {}

    # Ignore translations created by a different translation service
    # This ensures that an old Google translation file is regenerated
    existing = existing[
        existing["translation_source"] == TRANSLATION_SOURCE
    ]

    # Remove rows that do not contain a completed English translation
    existing = existing.dropna(subset=["product_name_en"])

    existing = existing[
        existing["product_name_en"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    # Use the category identifier as the translation lookup key
    return dict(
        zip(
            existing["category_id"].astype(int),
            existing["product_name_en"].astype(str),
        )
    )


def load_translation_model():
    # Keep the downloaded model on persistent storage
    # Later runs can load the same files without another download
    TRANSLATION_MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    # Use the first CUDA device when a compatible GPU is available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Translation device: {device}")

    if torch.cuda.is_available():
        print(f"Translation GPU: {torch.cuda.get_device_name(0)}")

    print("Loading the MarianMT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL_NAME,
        cache_dir=str(TRANSLATION_MODEL_CACHE),
    )

    print("Loading the MarianMT translation model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL_NAME,
        cache_dir=str(TRANSLATION_MODEL_CACHE),
    )

    model = model.to(device)
    model.eval()

    print("MarianMT translation model loaded.")

    return tokenizer, model, device


def translate_product_batch(
    tokenizer,
    model,
    device,
    chinese_names,
):
    # Convert a batch of Chinese product names into model tokens
    encoded = tokenizer(
        chinese_names,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # Move every token tensor to the selected inference device
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # Disable gradient tracking because translation is inference only
    with torch.inference_mode():
        generated_tokens = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    # Decode generated tokens into readable English product names
    english_names = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return [name.strip() for name in english_names]


def save_name_mapping(name_mapping):
    # Save Chinese text using an encoding that works well with Excel
    name_mapping.to_csv(
        NAME_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def create_name_mapping(products):
    # Reuse completed translations from an earlier run
    # Only translations created by the same MarianMT model are reused
    existing_translations = load_existing_translations()

    # Keep the original product metadata in the mapping file
    name_mapping = products.copy()
    # Force an object column before assigning translated text
    # An all empty mapped column would otherwise become a float column
    name_mapping["product_name_en"] = (
        name_mapping["category_id"]
        .map(existing_translations)
        .astype(object)
    )

    # Record the local model used to create every English name
    name_mapping["translation_source"] = TRANSLATION_SOURCE

    # Machine translated names should be reviewed before formal use
    name_mapping["translation_reviewed"] = False

    # Find product rows that still require translation
    missing_mask = (
        name_mapping["product_name_en"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    missing_indices = name_mapping.index[missing_mask].tolist()

    if not missing_indices:
        print("All MarianMT translations already exist.")
        save_name_mapping(name_mapping)
        return name_mapping

    print(
        f"Products requiring translation: {len(missing_indices)}"
    )

    tokenizer, model, device = load_translation_model()
    translated_count = len(name_mapping) - len(missing_indices)

    # Translate missing products in batches for efficient GPU use
    for start in range(0, len(missing_indices), TRANSLATION_BATCH_SIZE):
        batch_indices = missing_indices[
            start:start + TRANSLATION_BATCH_SIZE
        ]

        chinese_names = (
            name_mapping.loc[batch_indices, "product_name_cn"]
            .astype(str)
            .tolist()
        )

        english_names = translate_product_batch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            chinese_names=chinese_names,
        )

        # Write each translated name back to its original product row
        name_mapping.loc[
            batch_indices,
            "product_name_en",
        ] = english_names

        translated_count += len(batch_indices)

        # Save every completed batch to support interrupted runs
        save_name_mapping(name_mapping)
        print(
            f"Translation progress: "
            f"{translated_count}/{len(name_mapping)}"
        )

    # Save the completed mapping table
    save_name_mapping(name_mapping)

    return name_mapping


def verify_outputs(products, price_table, name_mapping):
    # The two output files must contain one row for every product SKU
    expected_count = len(products)

    if len(price_table) != expected_count:
        raise ValueError(
            "The price table has an unexpected row count."
        )

    if len(name_mapping) != expected_count:
        raise ValueError(
            "The name mapping has an unexpected row count."
        )

    # Count missing values to make output problems easy to identify
    missing_prices = price_table["unit_price"].isna().sum()

    missing_translations = (
        name_mapping["product_name_en"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    # Print a compact summary that also works in a regular Python script
    print("\nPreparation summary")
    print("-" * 60)
    print(f"Product classes:       {products['sku_class'].nunique()}")
    print(f"Product SKUs:          {expected_count}")
    print(f"Unique barcodes:       {products['barcode'].nunique()}")
    print(f"Missing prices:        {missing_prices}")
    print(f"Missing translations:  {missing_translations}")
    print(f"Minimum price:         ¥{price_table['unit_price'].min():.2f}")
    print(f"Maximum price:         ¥{price_table['unit_price'].max():.2f}")
    print(f"Price file:            {PRICE_OUTPUT_PATH}")
    print(f"Name mapping file:     {NAME_OUTPUT_PATH}")

    # Print only English fields to keep notebook results consistent
    print("\nPrice table preview")
    print(
        price_table[
            [
                "category_id",
                "sku_name",
                "sku_class",
                "barcode",
                "unit_price",
                "currency",
                "price_source",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nName mapping preview")
    print(
        name_mapping[
            [
                "category_id",
                "sku_name",
                "sku_class",
                "product_name_en",
                "barcode",
                "translation_source",
                "translation_reviewed",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # Tell the user when another run is needed to retry failed translations
    if missing_translations > 0:
        print(
            "\nSome translations are missing. "
            "Run the script again to retry them."
        )


def main():
    # Create the output directory when it does not already exist
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    # Load the shared product metadata once
    print("Loading product metadata...")
    products = load_product_metadata()

    # Generate the fixed simulated price table
    print("Creating product price table...")
    price_table = create_price_table(products)

    # Generate or resume the Chinese and English name mapping
    print("Creating Chinese and English product name mapping...")
    name_mapping = create_name_mapping(products)

    # Check the final files and print a short preview
    verify_outputs(
        products=products,
        price_table=price_table,
        name_mapping=name_mapping,
    )


# Run the preparation process when this file is executed directly
if __name__ == "__main__":
    main()
