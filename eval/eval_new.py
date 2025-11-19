import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import json
import re
import logging
from scipy.special import softmax
from collections import defaultdict
import time
import argparse
import os

# ----------------------------
# 1. 解析参数
# ----------------------------
parser = argparse.ArgumentParser(description="Run evaluation script.")
parser.add_argument('--model_name', type=str, required=True, help="Name of the model to load.")
parser.add_argument('--OMICS', type=str, required=True, help="Omics data to process.")
parser.add_argument('--input_file_path', type=str, required=True, help="Input data to process.")
args = parser.parse_args()
model_name = args.model_name
OMICS = args.OMICS
input_file_path = args.input_file_path

# ----------------------------
# 2. 日志
# ----------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
os.makedirs("new_save/logging", exist_ok=True)
logging.basicConfig(
    filename=f'new_save/logging/metrics_{model_name}_{OMICS}_{timestamp}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

# ----------------------------
# 3. 工具函数
# ----------------------------
def extract_numeric_values(text):
    matches = re.findall(r'(-?\d+\.?\d*)', str(text))
    numeric_values = []
    for num in matches:
        value = np.float64(num)
        if value.is_integer():
            value = f'{int(value):.6g}'
        else:
            value = f'{value:.6g}'
        numeric_values.append(float(value))
    return numeric_values


def classify_by_keywords(text):
    positive_keywords = ['yes']
    negative_keywords = [
        'no', 'absence', 'not found', 'not detected', 'not associated',
        'not inferred', 'not linked', 'does not indicate', 'no evidence',
        'not predicted', 'absent'
    ]
    dont_know_keywords = ["don't know", 'unknown', 'unsure', 'uncertain', 'not applicable']

    text_lower = text.lower()

    if any(kw in text_lower for kw in positive_keywords):
        return 1
    elif any(kw in text_lower for kw in negative_keywords):
        return 0
    elif any(kw in text_lower for kw in dont_know_keywords):
        return "dont_know"
    else:
        return None

# ----------------------------
# 4. 情感模型（作为兜底）
# ----------------------------
MODEL = "/mnt/shared-storage-user/ai4agr-share/lijinzhe/PreModel/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to('cuda')

def classify_by_sentiment_model(text):
    encoded_inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to('cuda')

    with torch.no_grad():
        output = model(**encoded_inputs)

    scores = output.logits.cpu().numpy()
    scores = softmax(scores, axis=1)

    result_dict = {config.id2label[i]: score for i, score in enumerate(scores[0])}
    positive_score = result_dict['positive']
    negative_score = result_dict['negative']

    if positive_score > negative_score:
        return (1, positive_score)
    else:
        return (0, negative_score)

# ----------------------------
# 5. 通用保存函数
# ----------------------------
def save_processed_data(model_name, task_name, task_processed_data):
    dir_path = f"new_save/processed_data/{model_name}"
    file_path = f"{dir_path}/{task_name}_processed_data.json"
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as outfile:
        json.dump(task_processed_data, outfile, indent=4)
    logger.info(f"Task {task_name} procssed data saved in {file_path}")
    print(f"Task {task_name} procssed data saved in {file_path}")

# ----------------------------
# 6. 回归任务
# ----------------------------
def process_regression_task(task_name, task_entries):
    result_values = []
    label_values = []
    task_processed_data = []

    for entry in task_entries:
        label = float(entry["label"])
        extracted_result = extract_numeric_values(entry["model_output"])

        if len(extracted_result) == 0:
            logger.warning(f"No valid result extracted for task: {task_name}. Skipping entry.")
            logger.info(f"Model output: {entry['model_output']}. Label: {entry['label']}")
            result_values.append(np.inf)
        else:
            result_values.append(extracted_result[0])

        label_values.append(label)

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_ouput": extracted_result[0] if len(extracted_result) > 0 else np.inf,
            "original_model_output": entry["model_output"],
        })

    save_processed_data(model_name, task_name, task_processed_data)
    return task_processed_data, label_values, result_values


def compute_spearman(label_values, result_values):
    if len(result_values) == 0:
        return {"spearman": "Error: Empty data"}
    elif len(result_values) != len(label_values):
        return {"spearman": "Error: Mismatch in the number of extracted numeric values"}

    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    near_infinity_mask = np.isinf(result_values)
    if near_infinity_mask.any():
        logger.warning(
            f"Found {sum(near_infinity_mask)} result values near infinity. "
            f"These will be assigned a Spearman score of 0."
        )

    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
    else:
        spearman = 0
        logger.warning("No valid result values. Assign the spearman to 0.")

    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_spearman_score = (spearman * total_valid_points) / total_data_points
    else:
        final_spearman_score = spearman

    return {"spearman": final_spearman_score}


def compute_R2(label_values, result_values):
    if len(result_values) == 0:
        return {"R2": "Error: Empty data."}
    elif len(result_values) != len(label_values):
        return {"R2": "Error: Mismatch in the number of extracted numeric values."}

    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    near_infinity_mask = np.isinf(result_values)
    if near_infinity_mask.any():
        logger.warning(
            f"Found {sum(near_infinity_mask)} result values near infinity. "
            f"These will be assigned an R2 score of 0."
        )

    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc ** 2
        except Exception as e:
            logger.error(f"Error in computing R2: {e}. Assign the R2 to inf.")
            R2 = np.inf
    else:
        R2 = 0
        logger.error("No valid result values. Assign the R2 to 0.")

    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_R2_score = (R2 * total_valid_points) / total_data_points
    else:
        final_R2_score = R2

    return {"R2": final_R2_score}


def compute_mixed_score(label_values, result_values, threshold=30, max_value=1e3):
    if len(result_values) == 0:
        return {"mixed_score": "Error: Empty data."}
    elif len(result_values) != len(label_values):
        return {"mixed_score": "Error: Mismatch in the number of extracted numeric values"}

    result_values = pd.to_numeric(result_values, errors='coerce').flatten()
    label_values = pd.to_numeric(label_values, errors='coerce').flatten()

    near_infinity_mask = np.abs(result_values) > max_value
    if near_infinity_mask.any():
        logger.warning(
            f"Warning: Found {sum(near_infinity_mask)} result values too large will be "
            f"assigned a mixed score of 0."
        )
        logger.info(f"Large result values: {result_values[near_infinity_mask]} ")
        print(
            f"Warning: Found {sum(near_infinity_mask)} result values too large will be "
            f"assigned a mixed score of 0. Large result values: {result_values[near_infinity_mask]} "
        )

    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    num_infinity_values = near_infinity_mask.sum()
    if num_infinity_values > 0:
        mixed_score_infinity = 0

    label_binary = (valid_label_values < threshold).astype(int)
    result_binary = (valid_result_values < threshold).astype(int)

    precision = precision_score(label_binary, result_binary, average='binary')
    recall = recall_score(label_binary, result_binary, average="binary")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    logger.info(f"F1: {f1}")

    try:
        mae = mean_absolute_error(valid_label_values, valid_result_values)
        logger.info(f"MAE: {mae}")
    except ValueError as e:
        logger.error(f"Error in computing MAE: {e}")
        mae = np.inf

    mask = (valid_result_values >= 0) & (valid_result_values <= threshold)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask], valid_result_values[mask])
    else:
        range_mae = 100
    logger.info(f"Range MAE: {range_mae}")

    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    logger.info(
        f"(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n"
        f"(1 - range_mae / 100)={(1 - range_mae / 100)}\n"
        f"(1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}"
    )
    print(
        f"(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n"
        f"(1 - range_mae / 100)={(1 - range_mae / 100)}\n"
        f"(1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}"
    )

    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()

    if num_infinity_values > 0:
        final_mixed_score = (
            mixed_score_valid * total_valid_points +
            mixed_score_infinity * num_infinity_values
        ) / total_data_points
    else:
        final_mixed_score = mixed_score_valid

    return {"mixed_score": final_mixed_score}

# ----------------------------
# 7. 特定任务：ProgrammableRNASwitches / enhancer_activity
# ----------------------------
def compute_R2_for_ProgrammableRNASwitches_task(task_name, task_entries):
    on_result_values = []
    off_result_values = []
    on_off_result_values = []

    on_label_values = []
    off_label_values = []
    on_off_label_values = []

    task_processed_data = []

    for entry in task_entries:
        label = entry["label"]
        if isinstance(label, str):
            label = json.loads(label)

        on_label = float(label["ON"])
        off_label = float(label["OFF"])
        on_off_label = float(label["ON_OFF"])

        extracted_result = extract_numeric_values(entry["model_output"])

        if len(extracted_result) != 3:
            logger.warning(f"Length mismatch in task: {task_name}. Assigning result values to NaN.")
            on_result_values.append(np.nan)
            off_result_values.append(np.nan)
            on_off_result_values.append(np.nan)
            on_result = off_result = on_off_result = np.nan
        else:
            on_result = extracted_result[0]
            off_result = extracted_result[1]
            on_off_result = extracted_result[2]
            on_result_values.append(on_result)
            off_result_values.append(off_result)
            on_off_result_values.append(on_off_result)

        on_label_values.append(on_label)
        off_label_values.append(off_label)
        on_off_label_values.append(on_off_label)

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": {
                "ON": on_result,
                "OFF": off_result,
                "ON_Off": on_off_result
            },
            "original_model_output": entry["model_output"]
        })

    save_processed_data(model_name, task_name, task_processed_data)

    on_result_values = np.array(on_result_values)
    off_result_values = np.array(off_result_values)
    on_off_result_values = np.array(on_off_result_values)

    on_label_values = np.array(on_label_values)
    off_label_values = np.array(off_label_values)
    on_off_label_values = np.array(on_off_label_values)

    on_valid_mask = np.isfinite(on_result_values) & np.isfinite(on_label_values)
    off_valid_mask = np.isfinite(off_result_values) & np.isfinite(off_label_values)
    on_off_valid_mask = np.isfinite(on_off_result_values) & np.isfinite(on_off_label_values)

    if not on_valid_mask.all():
        logger.warning(f"Found invalid ON result/label values at positions: {np.where(~on_valid_mask)[0]}")
    if not off_valid_mask.all():
        logger.warning(f"Found invalid OFF result/label values at positions: {np.where(~off_valid_mask)[0]}")
    if not on_off_valid_mask.all():
        logger.warning(f"Found invalid ON/OFF result/label values at positions: {np.where(~on_off_valid_mask)[0]}")

    on_result_values_valid = on_result_values[on_valid_mask]
    off_result_values_valid = off_result_values[off_valid_mask]
    on_off_result_values_valid = on_off_result_values[on_off_valid_mask]

    on_label_values_valid = on_label_values[on_valid_mask]
    off_label_values_valid = off_label_values[off_valid_mask]
    on_off_label_values_valid = on_off_label_values[on_off_valid_mask]

    try:
        on_R2 = compute_R2(on_label_values_valid, on_result_values_valid)['R2'] if len(on_result_values_valid) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for ON: {e}")
        on_R2 = 0

    try:
        off_R2 = compute_R2(off_label_values_valid, off_result_values_valid)['R2'] if len(off_result_values_valid) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for OFF: {e}")
        off_R2 = 0

    try:
        on_off_R2 = compute_R2(on_off_label_values_valid, on_off_result_values_valid)['R2'] if len(on_off_result_values_valid) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for ON/OFF: {e}")
        on_off_R2 = 0

    total_on_points = max(len(on_result_values_valid) + np.sum(~on_valid_mask), 1)
    total_off_points = max(len(off_result_values_valid) + np.sum(~off_valid_mask), 1)
    total_on_off_points = max(len(on_off_result_values_valid) + np.sum(~on_off_valid_mask), 1)

    final_on_R2 = (on_R2 * len(on_result_values_valid)) / total_on_points if len(on_result_values_valid) > 0 else 0
    final_off_R2 = (off_R2 * len(off_result_values_valid)) / total_off_points if len(off_result_values_valid) > 0 else 0
    final_on_off_R2 = (on_off_R2 * len(on_off_result_values_valid)) / total_on_off_points if len(on_off_result_values_valid) > 0 else 0

    avg_R2 = (final_on_R2 + final_off_R2 + final_on_off_R2) / 3

    return {"R2": avg_R2}


def compute_PCC_for_enhancer_activity_task(task_name, task_entries):
    hk_result_values = []
    dev_result_values = []
    hk_label_values = []
    dev_label_values = []
    task_processed_data = []

    for entry in task_entries:
        label = entry["label"]
        model_output = entry["model_output"]

        if isinstance(label, str):
            label = json.loads(label)

        hk_label = float(label["hk"])
        dev_label = float(label["dev"])

        extracted_result = extract_numeric_values(model_output)

        if len(extracted_result) != 2:
            logger.warning(f"Length mismatch in task: {task_name}. Assigning result values to infinity.")
            hk_result = dev_result = np.inf
            hk_result_values.append(np.inf)
            dev_result_values.append(np.inf)
        else:
            hk_result = extracted_result[0]
            dev_result = extracted_result[1]
            hk_result_values.append(hk_result)
            dev_result_values.append(dev_result)

        hk_label_values.append(hk_label)
        dev_label_values.append(dev_label)

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": {
                "hk": hk_result,
                "dev": dev_result
            },
            "original_model_output": entry["model_output"]
        })

    save_processed_data(model_name, task_name, task_processed_data)

    hk_result_values = np.array(hk_result_values)
    dev_result_values = np.array(dev_result_values)
    hk_label_values = np.array(hk_label_values)
    dev_label_values = np.array(dev_label_values)

    hk_valid_mask = np.isfinite(hk_result_values) & np.isfinite(hk_label_values)
    dev_valid_mask = np.isfinite(dev_result_values) & np.isfinite(dev_label_values)

    if not hk_valid_mask.all():
        logger.warning(f"Found invalid HK result/label values at positions: {np.where(~hk_valid_mask)[0]}")
        logger.info(f"Invalid HK result/label values: {hk_result_values[~hk_valid_mask]}, {hk_label_values[~hk_valid_mask]}")
    if not dev_valid_mask.all():
        logger.warning(f"Found invalid Dev result/label values at positions: {np.where(~dev_valid_mask)[0]}")
        logger.info(f"Invalid Dev result/label values: {dev_result_values[~dev_valid_mask]}, {dev_label_values[~dev_valid_mask]}")

    hk_result_values_valid = hk_result_values[hk_valid_mask]
    hk_label_values_valid = hk_label_values[hk_valid_mask]
    dev_result_values_valid = dev_result_values[dev_valid_mask]
    dev_label_values_valid = dev_label_values[dev_valid_mask]

    if len(hk_result_values_valid) > 0:
        try:
            hk_pcc, _ = pearsonr(hk_result_values_valid, hk_label_values_valid)
        except Exception as e:
            logger.error(f"Error computing Pearson correlation for HK: {e}")
            hk_pcc = np.inf
    else:
        return {"PCC": "Error: HK has insufficient valid data after removing NaNs and infs."}

    if len(dev_result_values_valid) > 0:
        try:
            dev_pcc, _ = pearsonr(dev_result_values_valid, dev_label_values_valid)
        except Exception as e:
            logger.error(f"Error computing Pearson correlation for Dev: {e}")
            dev_pcc = np.inf
    else:
        return {"PCC": "Error: Dev has insufficient valid data after removing NaNs and infs."}

    total_hk_points = len(hk_result_values_valid) + np.sum(~hk_valid_mask)
    total_dev_points = len(dev_result_values_valid) + np.sum(~dev_valid_mask)

    final_hk_pcc = (hk_pcc * len(hk_result_values_valid)) / total_hk_points if len(hk_result_values_valid) > 0 else 0
    final_dev_pcc = (dev_pcc * len(dev_result_values_valid)) / total_dev_points if len(dev_result_values_valid) > 0 else 0

    return {"PCC": {"hk_PCC": final_hk_pcc, "dev_PCC": final_dev_pcc}}

# ----------------------------
# 8. 二分类任务
# ----------------------------
def process_binary_classification_task(task_name, task_entries):
    label_classes = []
    result_classes = []
    task_processed_data = []

    for entry in task_entries:
        label_class = 1 if entry["label"] == 'positive' else 0

        if entry["model_output"] is None:
            result_class = 1 - label_class
            score = 0
        else:
            score = 0
            result_class = classify_by_keywords(entry["model_output"])

            if result_class == "dont_know" and label_class is not None:
                result_class = 1 - label_class
            elif result_class is None:
                result_class, score = classify_by_sentiment_model(entry["model_output"])

        result_classes.append(result_class)
        label_classes.append(label_class)

        task_processed_data.append({
            "input": entry["input"],
            "original_label": entry["label"],
            "processed_label": label_class,
            "original_model_output": entry["model_output"],
            "processed_model_output": result_class,
            "score": str(score) if score != 0 else "N/A"
        })

    save_processed_data(model_name, task_name, task_processed_data)
    return task_processed_data, label_classes, result_classes


def compute_MCC(label_classes, result_classes):
    if len(result_classes) == 0:
        return {"MCC": "Error: Empty data."}
    elif len(result_classes) != len(label_classes):
        return {"MCC": "Error: Mismatch in the number of extracted numeric values."}
    else:
        mcc = matthews_corrcoef(label_classes, result_classes)
        return {"MCC": mcc}


def compute_Acc(label_classes, result_classes):
    if len(result_classes) == 0:
        return {"Acc": "Error: Insufficient data for classification. Number of model outputs is 0."}
    elif len(result_classes) != len(label_classes):
        return {"Acc": "Error: Mismatched labels. The number of model outputs does not match the number of labels."}
    else:
        acc = accuracy_score(label_classes, result_classes)
        return {"Acc": acc}

# ----------------------------
# 9. NoncodingRNAFamily 多分类任务
# ----------------------------
RNA_CLASSES = sorted(
    ['5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA',
     'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES',
     'leader', 'scaRNA'],
    key=len,
    reverse=True
)


def extract_rna_family(text):
    for rna_class in RNA_CLASSES:
        if rna_class in text:
            return rna_class
    return None


def compute_Acc_for_NoncodingRNAFamily_task(task_name, task_entries):
    correct_count = 0
    total_count = 0
    task_processed_data = []

    for entry in task_entries:
        label_family = entry["label"]
        result_family = extract_rna_family(entry["model_output"])

        if result_family is None:
            logger.warning(f"No valid RNA family extracted from result: {entry['model_output']}")

        if result_family == label_family:
            correct_count += 1
        else:
            logger.warning("Not matching.")
            logger.info(f"Model output: {entry['model_output']}. Label: {entry['label']}")

        total_count += 1

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": result_family,
            "original_model_output": entry["model_output"]
        })

    save_processed_data(model_name, task_name, task_processed_data)

    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Task {task_name}: Accuracy = {accuracy:.4f}")

    return {"Acc": accuracy}

# ----------------------------
# 10. Modification 多标签分类任务
# ----------------------------
modification_classes = sorted(
    ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am',
     'm7G', 'Psi', 'AtoI', 'none'],
    key=len,
    reverse=True
)


def extract_modifications(text):
    extracted_modifications = []
    for mod_class in modification_classes:
        if re.search(rf'\b{mod_class}\b', text):
            extracted_modifications.append(mod_class)
    return extracted_modifications


def convert_to_binary_vector(modifications, classes=modification_classes):
    binary_vector = []

    if modifications is None:
        modifications = []

    for mod in classes:
        binary_vector.append(1 if mod in modifications else 0)
    return binary_vector


def compute_AUC_for_Modification_task(task_name, task_entries):
    y_true = []
    y_pred = []
    task_processed_data = []

    for entry in task_entries:
        predicted_modifications = extract_modifications(entry["model_output"])
        true_modifications = entry["label"].split(',')

        if predicted_modifications == [] and true_modifications == ['none']:
            predicted_modifications_tmp = classify_by_keywords(entry["model_output"])

            if predicted_modifications_tmp == 0:
                predicted_modifications = ['none']
            elif predicted_modifications_tmp == 1:
                predicted_modifications = []
            elif predicted_modifications_tmp is None:
                sentiment_result, sentiment_score = classify_by_sentiment_model(entry["model_output"])
                if sentiment_result == 0:
                    predicted_modifications = ['none']
                    logger.info(
                        f"Label: {entry['label']} Model output: {entry['model_output']} "
                        f"The result is assigned to a negative sentiment score: "
                        f"{(sentiment_result, sentiment_score)}"
                    )
                else:
                    predicted_modifications = []
                    logger.info(
                        f"Label: {entry['label']} Model output: {entry['model_output']} "
                        f"The result is assigned to a positive sentiment score: "
                        f"{(sentiment_result, sentiment_score)}"
                    )

        y_true.append(convert_to_binary_vector(true_modifications))
        y_pred.append(convert_to_binary_vector(predicted_modifications))

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_ouput": predicted_modifications,
            "original_model_output": entry["model_output"]
        })

    save_processed_data(model_name, task_name, task_processed_data)

    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError as e:
        logger.error(f"Error calculating AUC for task: {task_name}. Error: {str(e)}")
        auc = None

    if auc is not None:
        logger.info(f"Task {task_name}: AUC = {auc:.4f}")
    else:
        logger.info("AUC could not be computed")

    return {"AUC": auc}

# ----------------------------
# 11. FunctionEC Fmax 任务
# ----------------------------
def count_f1_max(pred, target):
    if pred.numel() == 0 or target.numel() == 0:
        logger.warning("Empty input provided. Returning F1 score of 0.0.")
        return 0.0

    order = pred.argsort(descending=True, dim=1, stable=True)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()

    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)

    if torch.isnan(all_f1).any():
        logger.warning(f"NaN encountered in F1 score computation. all_f1: {all_f1}")
        return 0.0

    return all_f1.max()


with open("ec_labels.json", "r") as f:
    ec_labels = json.load(f)


def ec_to_multihot(ec_list, ec_labels):
    multihot = torch.zeros(len(ec_labels))
    if not ec_list:
        return multihot
    multihot = torch.zeros(len(ec_labels))
    for ec in ec_list:
        if ec in ec_labels:
            idx = ec_labels.index(ec)
            multihot[idx] = 1
    return multihot


def compute_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels):
    all_preds = []
    all_labels = []
    task_processed_data = []

    for entry in task_entries:
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', entry['label'])
        result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(entry['model_output']))

        if result_ec == []:
            logger.warning(f"EC num not found in the result: {entry['model_output']}")
        if label_ec == []:
            logger.warning(f"EC num not found in the result: {entry['label']}")

        pred_multihot = ec_to_multihot(result_ec, ec_labels)
        label_multihot = ec_to_multihot(label_ec, ec_labels)

        all_preds.append(pred_multihot)
        all_labels.append(label_multihot)

        task_processed_data.append({
            'input': entry['input'],
            "label": entry["label"],
            "processed_label": label_ec,
            "original_model_output": entry["model_output"],
            'processed_model_output': result_ec,
        })

    save_processed_data(model_name, task_name, task_processed_data)

    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)

    try:
        fmax_score = count_f1_max(all_preds, all_labels)
    except ValueError as e:
        logger.error(f"Error calculating Fmax for task: {task_name}. Error: {str(e)}")
        fmax_score = None

    if fmax_score is not None:
        logger.info(f"Task {task_name}: Fmax = {fmax_score:.4f}")
        return {"Fmax": fmax_score.item()}
    else:
        logger.info("Fmax could not be computed")
        return {"Fmax": None}

# ----------------------------
# 12. 预处理输入数据：保留子任务，并在外层做“合并任务”
# ----------------------------
def preprocess_input_data(input_file_path):
    valid_lines = []
    with open(input_file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    valid_lines.append(data)
                else:
                    print(f"Skipping non-dictionary entry: {line.strip()}")
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")

    if len(valid_lines) == 0:
        print("No valid JSON entries found.")
        return None

    df = pd.DataFrame(valid_lines)
    print(f"Number of data samples: {len(df)}")
    logger.info(f"Number of data samples: {len(df)}")

    df.rename(columns={'result': 'model_output'}, inplace=True)
    df['task'] = df['task'].replace('rna_protein_interaction', 'ncRNAProteinInter')
    df['task'] = df['task'].replace('antibody_antigen', 'AntibodyAntigen')

    # 不再过滤 *_all 任务，保留子任务：pd-prom_300_all / cpd-prom_core_all 等
    # df = df[~df['task'].str.endswith('_all')]

    # 规范 tf 名字
    df['task'] = df['task'].str.replace('tf-h', 'tf_h')
    df['task'] = df['task'].str.replace('tf-m', 'tf_m')

    df = df[df['label'].notna()]
    df.reset_index(inplace=True, drop=True)

    data = df.to_dict(orient='records')

    # 先按“完整 task 名”分组（子任务层面）
    grouped_data = defaultdict(list)
    for entry in data:
        task_name = entry['task']
        grouped_data[task_name].append(entry)

    return grouped_data

# 13. 主流程：构建子任务 + 合并大任务
grouped_data_raw = preprocess_input_data(input_file_path)
logger.info(f"Raw grouped data for tasks: {list(grouped_data_raw.keys())}")
print(f"Raw grouped data for tasks: {list(grouped_data_raw.keys())}")

# 读取任务配置
with open("register_tasks.json", "r") as f:
    task_type_data = json.load(f)

# 定义 *已知* 的子任务 -> 大任务映射（手动写死的一些）
SUBTASK_GROUPS = {
    "pd": [
        "pd-prom_300_tata",
        "pd-prom_300_all",
        "pd-prom_300_notata",
    ],
    "cpd": [
        "cpd-prom_core_all",
        "cpd-prom_core_tata",
        "cpd-prom_core_notata",
    ],
    "tf_m": [
        "tf_m-0", "tf_m-1", "tf_m-2", "tf_m-3", "tf_m-4",
    ],
    "tf_h": [
        "tf_h-0", "tf_h-1", "tf_h-2", "tf_h-3", "tf_h-4",
    ],
}

# 🔹 自动收集 emp 的子任务（例如 emp-H3K9ac、emp-H3K4me3 等）
emp_subtasks = [
    name for name in grouped_data_raw.keys()
    if name != "emp" and name.startswith("emp-")
]
if emp_subtasks:
    SUBTASK_GROUPS["emp"] = emp_subtasks
    logger.info(f"Detected emp subtasks for merging: {emp_subtasks}")
    print(f"Detected emp subtasks for merging: {emp_subtasks}")

# 自动收集 promoter_enhancer_interaction 的子任务
pe_subtasks = [
    name for name in grouped_data_raw.keys()
    if name != "promoter_enhancer_interaction"
       and name.startswith("promoter_enhancer_interaction-")
]
if pe_subtasks:
    SUBTASK_GROUPS["promoter_enhancer_interaction"] = pe_subtasks
    logger.info(f"Detected promoter_enhancer_interaction subtasks for merging: {pe_subtasks}")
    print(f"Detected promoter_enhancer_interaction subtasks for merging: {pe_subtasks}")

# 构造最终 grouped_data：
# 1) 保留所有子任务本身
# 2) 追加合并后的大任务（pd / cpd / tf_m / tf_h / emp / promoter_enhancer_interaction）
grouped_data = defaultdict(list)

# 1) 子任务原样保存
for task_name, entries in grouped_data_raw.items():
    grouped_data[task_name] = entries

# 2) 大任务合并
for group_name, sub_tasks in SUBTASK_GROUPS.items():
    merged_entries = []
    for sub in sub_tasks:
        if sub in grouped_data_raw:
            merged_entries.extend(grouped_data_raw[sub])
    if len(merged_entries) > 0:
        grouped_data[group_name] = merged_entries

logger.info(f"Final grouped data for tasks (including merged groups): {list(grouped_data.keys())}")
print(f"Final grouped data for tasks (including merged groups): {list(grouped_data.keys())}")

# ----------------------------
# 14. 根据 task_name 获取配置用的“基任务名”
# ----------------------------
def get_base_task_name(task_name, task_type_data):
    """
    - 如果 task_name 本身在 register_tasks.json 里，就直接用它。
    - 否则根据前缀判断其所属大任务（pd / cpd / tf_m / tf_h / tf 等），
      再用大任务去 task_type_data 里查类型 & 指标。
    """
    if task_name in task_type_data:
        return task_name

    # 特殊子任务：先按我们约定的命名规则归类
    if task_name.startswith("pd-prom_300_"):
        return "pd"
    if task_name.startswith("cpd-prom_core_"):
        return "cpd"
    if task_name.startswith("tf_m-"):
        return "tf_m"
    if task_name.startswith("tf_h-"):
        return "tf_h"
    if task_name.startswith("tf-"):
        return "tf"

    # 兜底：按 '-' 拆分第一个字段
    base = task_name.split('-')[0]
    if base in task_type_data:
        return base

    raise KeyError(f"Task name '{task_name}' not found in register_tasks.json and cannot infer base task name.")

# ----------------------------
# 15. 逐任务计算指标
# ----------------------------
metrics = {}

for task_name, task_entries in grouped_data.items():
    try:
        base_task_name = get_base_task_name(task_name, task_type_data)
    except KeyError as e:
        logger.error(str(e))
        print(str(e))
        continue

    task_type = task_type_data[base_task_name]["type"]
    task_metrics = task_type_data[base_task_name]["metrics"]
    print(f"Prosessing {task_name} task (base: {base_task_name})...")

    if task_type == "regression":
        task_processed_data, label_values, result_values = process_regression_task(task_name, task_entries)

        if task_metrics == "spearman":
            metrics[task_name] = compute_spearman(label_values, result_values)
        elif task_metrics == "R2":
            metrics[task_name] = compute_R2(label_values, result_values)
        elif task_metrics == "mixed_score":
            metrics[task_name] = compute_mixed_score(label_values, result_values, threshold=30)

    elif task_type == "binary classification":
        task_processed_data, label_classes, result_classes = process_binary_classification_task(task_name, task_entries)

        if task_metrics == "MCC":
            metrics[task_name] = compute_MCC(label_classes, result_classes)
        elif task_metrics == "Acc":
            metrics[task_name] = compute_Acc(label_classes, result_classes)

    elif task_type == "multilabel regression":
        if base_task_name == "ProgrammableRNASwitches":
            metrics[task_name] = compute_R2_for_ProgrammableRNASwitches_task(task_name, task_entries)
        elif base_task_name == "enhancer_activity":
            metrics[task_name] = compute_PCC_for_enhancer_activity_task(task_name, task_entries)

    elif task_type == "multiclass classification":
        if base_task_name == "NoncodingRNAFamily":
            metrics[task_name] = compute_Acc_for_NoncodingRNAFamily_task(task_name, task_entries)

    elif task_type == "multilabel classification":
        if base_task_name == "FunctionEC":
            metrics[task_name] = compute_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels)
        elif base_task_name == "Modification":
            metrics[task_name] = compute_AUC_for_Modification_task(task_name, task_entries)

    print(f"The metrics {task_metrics} for task {task_name} is {str(metrics[task_name][task_metrics])}")
    logger.info(f"{task_metrics} of {task_name} is {str(metrics[task_name][task_metrics])}")

# ----------------------------
# 16. 缩放 & 按 omics 聚合保存
# ----------------------------
def round_and_scale_results(data, decimal_places=2, scale_factor=100):
    for key, value in data.items():
        if isinstance(value, dict):
            round_and_scale_results(value, decimal_places, scale_factor)
        elif isinstance(value, (float, int)):
            data[key] = float(round(value * scale_factor, decimal_places))


metrics_grouped_by_omics = defaultdict(dict)

for task_name, task_metrics in metrics.items():
    base_task_name = get_base_task_name(task_name, task_type_data)
    omics = task_type_data[base_task_name]["omics"]

    scaled_metrics = task_metrics.copy()
    round_and_scale_results(scaled_metrics)

    metrics_grouped_by_omics[omics][task_name] = scaled_metrics

metrics_file_path = f"new_save/metrics_result/metrics_result_{model_name}_{OMICS}.json"
os.makedirs("new_save/metrics_result", exist_ok=True)
with open(metrics_file_path, "w") as outfile:
    json.dump(metrics_grouped_by_omics, outfile, indent=4)

logger.info(f"Metrics saved to {metrics_file_path}")
print(f"Metrics saved to {metrics_file_path}")