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

# Set up argument parser to accept model_name as an argument
parser = argparse.ArgumentParser(description="Run evaluation script.")
parser.add_argument('--model_name', type=str, required=True, help="Name of the model to load.")
parser.add_argument('--OMICS', type=str, required=True, help="Omics data to process.")
parser.add_argument('--input_file_path', type=str, required=True, help="Input data to process.")
args = parser.parse_args()
model_name = args.model_name
OMICS = args.OMICS
input_file_path = args.input_file_path

# Set up logging
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
logging.basicConfig(
    filename=f'logging/metrics_{model_name}_{OMICS}_{timestamp}.log',
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrites the log file each time the script runs
)

# Create a logger
logger = logging.getLogger(__name__)

# Extract numeric values from a string using regex
def extract_numeric_values(text):
    matches = re.findall(r'(-?\d+\.?\d*)', str(text))
    
    # Convert to floats and ensure values are limited to 6 significant digits
    numeric_values = []
    for num in matches:
        value = np.float64(num)  # Convert to NumPy float64 for consistency
        
        # Limit the value to 6 significant digits
        if value.is_integer():  # If it's an integer, format as an integer with 6 digits max
            value = f'{int(value):.6g}'
        else:  # For floats, format with 6 significant digits
            value = f'{value:.6g}'
        
        numeric_values.append(float(value))  # Convert back to float for numeric operations

    return numeric_values

# Classify text into binary categories by keywords
def classify_by_keywords(text):
    positive_keywords = ['yes']
    negative_keywords = ['no', 'absence', 'not found', 'not detected', 'not associated', 'not inferred', 'not linked', 'does not indicate', 'no evidence', 'not predicted', 'absent']
    dont_know_keywords = ['don\'t know', 'unknown', 'unsure', 'uncertain', 'not applicable']

    text_lower = text.lower()

    # Check for positive keywords
    if any(kw in text_lower for kw in positive_keywords):
        return 1
    # Check for negative keywords
    elif any(kw in text_lower for kw in negative_keywords):
        return 0
    # Check for "don't know how to answer"
    elif any(kw in text_lower for kw in dont_know_keywords):
        return "dont_know"
    else:
        return None

# Initialize the sentiment analysis model
MODEL = "/tos-bjml-ai4agr/lijinzhe/BioMLLM/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to('cuda')

# Use the sentiment analysis model as fallback if classification by keywords fails
def classify_by_sentiment_model(text):
    # Encode the text
    encoded_inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    # print("Encoded inputs:", encoded_inputs)
    # print("Original text input:", text)
    
    # # Ensure input_ids and attention_mask match in size
    # if encoded_inputs['input_ids'].shape != encoded_inputs['attention_mask'].shape:
    #         print("Shape mismatch: input_ids shape:", encoded_inputs['input_ids'].shape, 
    #               "attention_mask shape:", encoded_inputs['attention_mask'].shape)
            
    # Get model output (no gradients required)
    with torch.no_grad():
        output = model(**encoded_inputs)
    # print("Model output logits shape:", output.logits.shape) 
    
    
    # Get the logits and apply softmax
    scores = output.logits.cpu().numpy()
    scores = softmax(scores, axis=1)
    # print("Softmax scores:", scores) 
    
    # Get the label with the highest probability
    result_dict = {config.id2label[i]: score for i, score in enumerate(scores[0])}
    positive_score = result_dict['positive']
    negative_score = result_dict['negative']


    # Determine sentiment (positive=1/negative=0)
    if positive_score > negative_score:
        return (1, positive_score)
    elif negative_score > positive_score:
        return (0, negative_score)

# Save the processed data for each task in a separate file
def save_processed_data(model_name, task_name, task_processed_data):
    dir_path = f"processed_data/{model_name}"
    file_path = f"{dir_path}/{task_name}_processed_data.json"
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as outfile:
        json.dump(task_processed_data, outfile, indent=4)
    logger.info(f"Task {task_name} procssed data saved in {file_path}")
    print(f"Task {task_name} procssed data saved in {file_path}")

# Process regression task
def process_regression_task(task_name, task_entries):
    result_values = []
    label_values = []
    task_processed_data = [] 

    for entry in task_entries:
        label = float(entry["label"])
        extracted_result = extract_numeric_values(entry["model_output"]) 
        
        # Ensure the extracted result is a valid numeric array
        if len(extracted_result) == 0:
            logger.warning(f"No valid result extracted for task: {task_name}. Skipping entry.")
            logger.info(f"Model output: {entry['model_output']}. Label: {entry['label']}") 
            result_values.append(np.inf)    # Assign infinity if no valid result is extracted
        else:
            result_values.append(extracted_result[0])     # Take the first valid extracted result

        label_values.append(label)

        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_ouput": extracted_result[0] if len(extracted_result) > 0 else np.inf,            
            "original_model_output": entry["model_output"], 
        })
    
    save_processed_data(model_name, task_name, task_processed_data)  
    return task_processed_data, label_values, result_values

# Compute spearman correlation
def compute_spearman(label_values, result_values):
    if len(result_values) == 0:
        return {
            "spearman": "Error: Empty data"
        }
    elif len(result_values) != len(label_values):
        return {
            "spearman": "Error: Mismatch in the number of extracted numeric values"
        }
    
    # Convert the label and result values to numpy arrays   
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)
    
    if near_infinity_mask.any():
        logger.warning(f"Found {sum(near_infinity_mask)} result values near infinity. These will be assigned a Spearman score of 0.")
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Spearman correlation for valid values
    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
    else:
        spearman = 0  # Fallback if no valid pairs
        logger.warning(f"No valid result values. Assign the spearman to 0.")

    # Combine Spearman score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_spearman_score = (spearman * total_valid_points + 0 * num_infinity_values) / total_data_points
    else:
        final_spearman_score = spearman  # Edge case: no near-infinity values

    return {
        "spearman": final_spearman_score
    }

# Compute R2
def compute_R2(label_values, result_values):
    # Check for empty data
    if len(result_values) == 0:
        return {
            "R2": "Error: Empty data."
        }
    
    # Check for equal length of arrays
    elif len(result_values) != len(label_values):
        return {
            "R2": "Error: Mismatch in the number of extracted numeric values."
        }

   # Convert the label and result values to numpy arrays   
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)
    
    if near_infinity_mask.any():
        logger.warning(f"Found {sum(near_infinity_mask)} result values near infinity. These will be assigned an R2 score of 0.")
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Pearson correlation coefficient for valid values
    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc ** 2
        except Exception as e:
            logger.error(f"Error in computing R2: {e}. Assign the R2 to inf.")
            R2 = np.inf  # Fallback to inf if computation fails
    else:
        R2 = 0  # Fallback if no valid pairs
        logger.error(f"No valid result values. Assign the R2 to 0.")

    # Combine R2 score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_R2_score = (R2 * total_valid_points + 0 * num_infinity_values) / total_data_points
    else:
        final_R2_score = R2  # Edge case: no near-infinity values

    return {
        "R2": final_R2_score
    }

# Compute mixed score
def compute_mixed_score(label_values, result_values, threshold=30, max_value=1e3):
    if len(result_values) == 0:
        return {
            "mixed_score": "Error: Empty data."
        }
    elif len(result_values) != len(label_values):
        return {
            "mixed_score": "Error: Mismatch in the number of extracted numeric values"
        }

    # Convert the label and result values to numeric arrays using pandas to handle non-numeric entries
    result_values = pd.to_numeric(result_values, errors='coerce').flatten()
    label_values = pd.to_numeric(label_values, errors='coerce').flatten()

    # Identify near-infinity values
    near_infinity_mask = np.abs(result_values) > max_value
    if near_infinity_mask.any():
        logger.warning(f"Warning: Found {sum(near_infinity_mask)} result values too large will be assigned a mixed score of 0.")
        logger.info(f"Large result values: {result_values[near_infinity_mask]} ")
        print(f"Warning: Found {sum(near_infinity_mask)} result values too large will be assigned a mixed score of 0. Large result values: {result_values[near_infinity_mask]} ")
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Assign a mixed score of 0 to near-infinity pairs
    num_infinity_values = near_infinity_mask.sum()
    if num_infinity_values > 0:
        mixed_score_infinity = 0

    # Convert to binary based on the threshold for valid values
    label_binary = (valid_label_values < threshold).astype(int)
    result_binary = (valid_result_values < threshold).astype(int)
    
    # Compute precision, recall, F1 score for valid values
    precision = precision_score(label_binary, result_binary, average='binary')
    recall = recall_score(label_binary, result_binary, average="binary")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    logger.info("F1:", f1)

    try:
        # Compute mean absolute error (MAE) for valid values
        mae = mean_absolute_error(valid_label_values, valid_result_values)
        logger.info("MAE:", mae)
    except ValueError as e:
        logger.error(f"Error in computing MAE: {e}")
        mae = np.inf  # Fallback to infinity if error occurs
    
    # Mask to keep only values in the range [0, threshold] for valid values
    mask = (valid_result_values >= 0) & (valid_result_values <= threshold)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask], valid_result_values[mask])
    else:
        range_mae = 100  # Fallback if no values within the range
    logger.info("Range MAE:", range_mae)

    # Ensure MAE and range_mae are within reasonable bounds to avoid overflow
    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    # Compute mixed score for valid values
    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    logger.info(f"(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n (1 - range_mae / 100)={(1 - range_mae / 100)}\n (1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}")
    print(f"(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n (1 - range_mae / 100)={(1 - range_mae / 100)}\n (1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}")

    # Compute the final mixed score, averaging in the score for the near-infinity pairs
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    
    if num_infinity_values > 0:
        final_mixed_score = (mixed_score_valid * total_valid_points + mixed_score_infinity * num_infinity_values) / total_data_points
    else:
        final_mixed_score = mixed_score_valid  # Edge case: no near-infinity values

    return {
        "mixed_score": final_mixed_score
    }


# Programmable Switch task: multilabel regression output one average correlation
def compute_R2_for_ProgrammableRNASwitches_task(task_name, task_entries):
    on_result_values = []
    off_result_values = []
    on_off_result_values = []

    on_label_values = []
    off_label_values = []
    on_off_label_values = []

    task_processed_data = []

    # Loop through each entry in the task
    for entry in task_entries:
        label = entry["label"]
        on_label = float(label["ON"])
        off_label = float(label["OFF"])
        on_off_label = float(label["ON_OFF"])

        # Extract numeric values from the model output
        extracted_result = extract_numeric_values(entry["model_output"])

        # Handle missing or invalid data by assigning np.nan
        if len(extracted_result) != 3:
            logger.warning(f"Length mismatch in task: {task_name}. Assigning result values to NaN.")
            on_result_values.append(np.nan)
            off_result_values.append(np.nan)
            on_off_result_values.append(np.nan)
        else:
            on_result = extracted_result[0]
            off_result = extracted_result[1]
            on_off_result = extracted_result[2]
            on_result_values.append(on_result)
            off_result_values.append(off_result)
            on_off_result_values.append(on_off_result)

        # Append the label values
        on_label_values.append(on_label)
        off_label_values.append(off_label)
        on_off_label_values.append(on_off_label)

        # Save processed task data for this entry
        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": {
                "ON": on_result if len(extracted_result) == 3 else np.nan,
                "OFF": off_result if len(extracted_result) == 3 else np.nan,
                "ON_Off": on_off_result if len(extracted_result) == 3 else np.nan
            },
            "original_model_output": entry["model_output"]
        })

    # Save the processed task data
    save_processed_data(model_name, task_name, task_processed_data)

    # Convert to numpy arrays for easier manipulation
    on_result_values = np.array(on_result_values)
    off_result_values = np.array(off_result_values)
    on_off_result_values = np.array(on_off_result_values)

    on_label_values = np.array(on_label_values)
    off_label_values = np.array(off_label_values)
    on_off_label_values = np.array(on_off_label_values)

    # Filter out NaN values in ON, OFF, and ON/OFF result/label pairs
    on_valid_mask = np.isfinite(on_result_values) & np.isfinite(on_label_values)
    off_valid_mask = np.isfinite(off_result_values) & np.isfinite(off_label_values)
    on_off_valid_mask = np.isfinite(on_off_result_values) & np.isfinite(on_off_label_values)

    # Log the removed NaN/inf values
    if not on_valid_mask.all():
        logger.warning(f"Found invalid ON result/label values at positions: {np.where(~on_valid_mask)[0]}")
    if not off_valid_mask.all():
        logger.warning(f"Found invalid OFF result/label values at positions: {np.where(~off_valid_mask)[0]}")
    if not on_off_valid_mask.all():
        logger.warning(f"Found invalid ON/OFF result/label values at positions: {np.where(~on_off_valid_mask)[0]}")

    # Filter the valid ON, OFF, and ON/OFF values
    on_result_values = on_result_values[on_valid_mask]
    off_result_values = off_result_values[off_valid_mask]
    on_off_result_values = on_off_result_values[on_off_valid_mask]

    on_label_values = on_label_values[on_valid_mask]
    off_label_values = off_label_values[off_valid_mask]
    on_off_label_values = on_off_label_values[on_off_valid_mask]

    # Compute R2 for valid ON, OFF, and ON/OFF values
    try:
        on_R2 = compute_R2(on_result_values, on_label_values)['R2'] if len(on_result_values) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for ON: {e}")
        on_R2 = 0  # Assign 0 in case of error

    try:
        off_R2 = compute_R2(off_result_values, off_label_values)['R2'] if len(off_result_values) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for OFF: {e}")
        off_R2 = 0  # Assign 0 in case of error

    try:
        on_off_R2 = compute_R2(on_off_result_values, on_off_label_values)['R2'] if len(on_off_result_values) > 0 else 0
    except Exception as e:
        logger.error(f"Error computing R2 for ON/OFF: {e}")
        on_off_R2 = 0  # Assign 0 in case of error

    # Combine R2 scores for ON, OFF, and ON/OFF values
    total_on_points = max(len(on_result_values) + np.sum(~on_valid_mask), 1)
    total_off_points = max(len(off_result_values) + np.sum(~off_valid_mask), 1)
    total_on_off_points = max(len(on_off_result_values) + np.sum(~on_off_valid_mask), 1)

    # Assign average R2 with 0 for invalid entries
    final_on_R2 = (on_R2 * len(on_result_values)) / total_on_points if len(on_result_values) > 0 else 0
    final_off_R2 = (off_R2 * len(off_result_values)) / total_off_points if len(off_result_values) > 0 else 0
    final_on_off_R2 = (on_off_R2 * len(on_off_result_values)) / total_on_off_points if len(on_off_result_values) > 0 else 0

    avg_R2 = (final_on_R2 + final_off_R2 + final_on_off_R2) / 3

    return {
        "R2": avg_R2
    }


# Enhancer Activity Task: multilabel regression output two individual correlation
def compute_PCC_for_enhancer_activity_task(task_name, task_entries):
    hk_result_values = []
    dev_result_values = []
    
    hk_label_values = []
    dev_label_values = []

    task_processed_data = []

    # Loop through each entry in the task
    for entry in task_entries:
        label = entry["label"]
        model_output = entry["model_output"]

        hk_label = float(label["hk"])
        dev_label = float(label["dev"])

        # Extract model output values for HK and Dev enhancer activity
        extracted_result = extract_numeric_values(model_output)
        
        # Handle missing or invalid data by assigning np.inf
        if len(extracted_result) != 2:
            logger.warning(f"Length mismatch in task: {task_name}. Assigning result values to infinity.")
            hk_result_values.append(np.inf)
            dev_result_values.append(np.inf)
        else:
            hk_result = extracted_result[0]
            dev_result = extracted_result[1]
            hk_result_values.append(hk_result)
            dev_result_values.append(dev_result)

        # Append the label values
        hk_label_values.append(hk_label)
        dev_label_values.append(dev_label)

        # Save processed task data for this entry
        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": {
                "hk": hk_result if len(extracted_result) == 2 else np.inf,
                "dev": dev_result if len(extracted_result) == 2 else np.inf
            },
            "original_model_output": entry["model_output"]
        })

    # Save the processed task data
    save_processed_data(model_name, task_name, task_processed_data)

    # Convert to numpy arrays for easier manipulation
    hk_result_values = np.array(hk_result_values)
    dev_result_values = np.array(dev_result_values)
    hk_label_values = np.array(hk_label_values)
    dev_label_values = np.array(dev_label_values)

    # Filter out NaN or inf values in both HK and Dev result/label pairs
    hk_valid_mask = np.isfinite(hk_result_values) & np.isfinite(hk_label_values)
    dev_valid_mask = np.isfinite(dev_result_values) & np.isfinite(dev_label_values)

    # Log the removed NaN/inf values
    if not hk_valid_mask.all():
        logger.warning(f"Found invalid HK result/label values at positions: {np.where(~hk_valid_mask)[0]}")
        logger.info(f"Invalid HK result/label values: {hk_result_values[~hk_valid_mask]}, {hk_label_values[~hk_valid_mask]}")
    
    if not dev_valid_mask.all():
        logger.warning(f"Found invalid Dev result/label values at positions: {np.where(~dev_valid_mask)[0]}")
        logger.info(f"Invalid Dev result/label values: {dev_result_values[~dev_valid_mask]}, {dev_label_values[~dev_valid_mask]}")

    # Filter the valid HK and Dev values
    hk_result_values = hk_result_values[hk_valid_mask]
    hk_label_values = hk_label_values[hk_valid_mask]
    dev_result_values = dev_result_values[dev_valid_mask]
    dev_label_values = dev_label_values[dev_valid_mask]

    # Compute Pearson correlation for valid HK and Dev enhancer activities
    if len(hk_result_values) > 0:
        try:
            hk_pcc, _ = pearsonr(hk_result_values, hk_label_values)
        except Exception as e:
            logger.error(f"Error computing Pearson correlation for HK: {e}")
            hk_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            "PCC": "Error: HK has insufficient valid data after removing NaNs and infs."
        }
    if len(dev_result_values) > 0:
        try:
            dev_pcc, _ = pearsonr(dev_result_values, dev_label_values)
        except Exception as e:
            logger.error(f"Error computing Pearson correlation for Dev: {e}")
            dev_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            "PCC": "Error: Dev has insufficient valid data after removing NaNs and infs."
        }

    # Combine results with NaN/inf values consideration
    total_hk_points = len(hk_result_values) + np.sum(~hk_valid_mask)
    total_dev_points = len(dev_result_values) + np.sum(~dev_valid_mask)

    # Assign mixed score with 0 for invalid entries
    final_hk_pcc = (hk_pcc * len(hk_result_values) + 0 * np.sum(~hk_valid_mask)) / total_hk_points if len(hk_result_values) > 0 else 0
    final_dev_pcc = (dev_pcc * len(dev_result_values) + 0 * np.sum(~dev_valid_mask)) / total_dev_points if len(dev_result_values) > 0 else 0

    return {
        "PCC": {
            "hk_PCC": final_hk_pcc,
            "dev_PCC": final_dev_pcc                
        }
    }


# Process binary classification task
def process_binary_classification_task(task_name, task_entries):
    label_classes = []
    result_classes = []
    task_processed_data = []

    for entry in task_entries:
        label_class = 1 if entry["label"]=='positive' else 0
        # If model output is empty, classified as wrong cases
        if entry["model_output"] is None:
            result_class = 1 - label_class
        else:   
            # Initialize score
            score = 0
            # Classify by keyword
            result_class = classify_by_keywords(entry["model_output"])

            # If model don't know the answer, classified as wrong cases
            if result_class == "dont_know" and label_class is not None:
                result_class = 1 - label_class

            # If the result cannot be classified as yes/no/dont know, classify by the sentiment model
            elif result_class is None:
                result_class, score = classify_by_sentiment_model(entry["model_output"])  
                
                # If assigned a low prob score, classied as wrong cases
                # if  score < 0.5:   
                #     result_class = 1 - label_class

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
    
# Compute matthews correlation coefficient (MCC)
def compute_MCC(label_classes, result_classes):
    if len(result_classes) == 0:
        return {
            "MCC": "Error: Empty data."
        }
    elif len(result_classes) != len(label_classes):
        return {
            "MCC": "Error: Mismatch in the number of extracted numeric values."
        }
    else:
        mcc = matthews_corrcoef(label_classes, result_classes)
        return {
            "MCC": mcc
        }

# Compute accuracy score (Acc)
def compute_Acc(label_classes, result_classes):
    if len(result_classes) == 0:
        return {
            "Acc": "Error: Insufficient data for classification. Number of model outputs is 0."
        }
    elif len(result_classes) != len(label_classes):
        return {
            "Acc": "Error: Mismatched labels. The number of model outputs does not match the number of labels."
        }
    else:
        acc = accuracy_score(label_classes, result_classes)
        return {
            "Acc": acc
        }
    
## NoncodingRNAFamily task: multiclass classification
# Sort RNA_CLASSES by length in descending order to match longer family names first
RNA_CLASSES = sorted(['5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA', 'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES', 'leader', 'scaRNA'], key=len, reverse=True)

# Extract RNA family from the text
def extract_rna_family(text):
    for rna_class in RNA_CLASSES:
        if rna_class in text:  
            return rna_class
    return None

# Compute ACC metric for NoncodingRNAFamily multiclass classification task
def compute_Acc_for_NoncodingRNAFamily_task(task_name, task_entries):
    correct_count = 0
    total_count = 0
    task_processed_data = [] 
    
    for entry in task_entries:
        label_family = entry["label"]
        result_family = extract_rna_family(entry["model_output"]) 

        if result_family is None:
            logger.warning(f"No valid RNA family extracted from result: {entry['model_output']}")
        
        # Compare extracted family with the ground truth label
        if result_family == label_family:
            correct_count += 1
        else:
            logger.warning(f"Not matching.")
            logger.info(f"Model output: {entry['model_output']}. Label: {entry['label']}")
        
        total_count += 1
        
        # Store original and processed data
        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_output": result_family,
            "original_model_output": entry["model_output"]
        })
    
    save_processed_data(model_name, task_name, task_processed_data)

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Task {task_name}: Accuracy = {accuracy:.4f}")

    return {
        "Acc": accuracy
    }

## Modification Task: multilabel classification
# List of possible RNA modification classes
modification_classes = sorted(
    ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Psi', 'AtoI', 'none'],
    key=len, reverse=True  # Sort by length, longest first
)

# Extract RNA modification labels from the output text
def extract_modifications(text):
    extracted_modifications = []
    for mod_class in modification_classes:
        # Use word boundaries to ensure whole-word match
        if re.search(rf'\b{mod_class}\b', text):
            extracted_modifications.append(mod_class)
    return extracted_modifications

# Convert modification labels to a binary multihot vector
def convert_to_binary_vector(modifications, classes=modification_classes):
    binary_vector = []
    
    # Handle case where modifications is None
    if modifications is None:
        modifications = []  # Treat None as an empty list
    
    for mod in classes:
        if mod in modifications:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector


# Compute AUC metrics for Modification task
def compute_AUC_for_Modification_task(task_name, task_entries):
    y_true = []
    y_pred = []
    task_processed_data = [] 
    
    for entry in task_entries:
        predicted_modifications = extract_modifications(entry["model_output"])
        true_modifications = entry["label"].split(',')
           
        # Handle case where result is empty and label is "none"
        if predicted_modifications == [] and true_modifications == ['none']:
            # Classify by keyword
            predicted_modifications = classify_by_keywords(entry["model_output"])

            # If keyword negative, assigned to prediction to be the "none" class
            if predicted_modifications == 0:
                predicted_modifications = ['none']

            elif predicted_modifications == 1:
                predicted_modifications = []
            
            # If the result cannot be classified, use the sentiment model
            elif predicted_modifications is None:
                sentiment_result, sentiment_score = classify_by_sentiment_model(entry["model_output"])
            
                # If classified as negative, manually label as 'none'
                if sentiment_result == 0:
                    predicted_modifications = ['none']
                    logger.info(f"Label: {entry['label']} Model output: {entry['model_output']} The result is assigned to a negative sentiment score: {(sentiment_result, sentiment_score)}")
                    
                else:
                    predicted_modifications = []
                    logger.info(f"Label: {entry['label']} Model output: {entry['model_output']} The result is assigned to a positive sentiment score: {(sentiment_result, sentiment_score)}")
        
        # Convert the predicted and true modifications to binary vectors
        y_true.append(convert_to_binary_vector(true_modifications))
        y_pred.append(convert_to_binary_vector(predicted_modifications))
        
        # Store the processed data
        task_processed_data.append({
            "input": entry["input"],
            "label": entry["label"],
            "processed_model_ouput": predicted_modifications,
            "original_model_output": entry["model_output"] 
        })
    
    save_processed_data(model_name, task_name, task_processed_data)
    
    # Compute the AUC for each class, then average the AUC across all classes
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError as e:
        logger.error(f"Error calculating AUC for task: {task_name}. Error: {str(e)}")
        auc = None

    logger.info(f"Task {task_name}: AUC = {auc:.4f}" if auc is not None else "AUC could not be computed")

    return {
        "AUC": auc
    }

## FunctionEC Task
# Modified from SaProt https://github.com/westlake-repl/SaProt/blob/main/utils/metrics.py
def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold. 
    Handles cases where either predictions or targets are empty.
    
    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
        
    Returns:
        float: The maximum F1 score or 0.0 if inputs are empty.
    """
    # Check if either pred or target is empty
    if pred.numel() == 0 or target.numel() == 0:
        logger.warnign(f"Empty input provided. Returning F1 score of 0.0.")
        return 0.0

    # Proceed with the original logic if inputs are not empty
    order = pred.argsort(descending=True, dim=1, stable=True)
    # print(f"order: {order}")
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    # print("isstart {}".format(is_start))
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()
    
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    
    if torch.isnan(all_f1).any():
        logger.warning(f"NaN encountered in F1 score computation. all_f1: {all_f1}")
        return 0.0

    return all_f1.max()


# Read EC labels
with open("ec_labels.json", "r") as f:
    ec_labels = json.load(f)


# Convert EC number to binary multihot vectors
def ec_to_multihot(ec_list, ec_labels):
    multihot = torch.zeros(len(ec_labels))
    if not ec_list:  # Check if ec_list is empty
        return multihot
    multihot = torch.zeros(len(ec_labels))
    for ec in ec_list:
        if ec in ec_labels:
            idx = ec_labels.index(ec)
            multihot[idx] = 1
    return multihot

# Compute Fmax metric for FunctionEC task
def compute_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels):
    all_preds = []
    all_labels = []
    task_processed_data = []

    for entry in task_entries:
        # Parse the EC numbers from 'output' and 'label'
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', entry['label'])
        result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(entry['model_output']))


        if result_ec == []:
            logger.warning(f"EC num not found in the result: {entry['model_output']}")
        if label_ec == []:
            logger.warning(f"EC num not found in the result: {entry['label']}")

        # Convert EC numbers to multi-hot vectors
        pred_multihot = ec_to_multihot(result_ec, ec_labels) 
        label_multihot = ec_to_multihot(label_ec, ec_labels)

        # Store the results
        all_preds.append(pred_multihot)
        all_labels.append(label_multihot)
    
        # Save processed task data
        task_processed_data.append({
            'input': entry['input'],
            "label": entry["label"], 
            "processed_label": label_ec,
            "original_model_output": entry["model_output"],
            'processed_model_output': result_ec,
        })
    
    save_processed_data(model_name, task_name, task_processed_data)

    # # Stack the predictions and targets for batch processing
    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)
    
    # Compute the Fmax score
    try:
        fmax_score = count_f1_max(all_preds, all_labels)
    except ValueError as e:
        logger.error(f"Error calculating Fmax for task: {task_name}. Error: {str(e)}")
        fmax_score = None

    logger.info(f"Task {task_name}: Fmax = {fmax_score:.4f}" if fmax_score is not None else "Fmax could not be computed")

    return {
        "Fmax": fmax_score.item()
    }
    
# Preprocess input data and return grouped data by 'task'
def preprocess_input_data(input_file_path):
    valid_lines = []
    # Open the input file and process each line
    with open(input_file_path, 'r') as f:
        for line in f:
            try:
                # Try to load the line as a JSON object
                data = json.loads(line)
                
                # Ensure the parsed data is a dictionary
                if isinstance(data, dict):
                    valid_lines.append(data)
                else:
                    print(f"Skipping non-dictionary entry: {line.strip()}")

            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line.strip()}")

    # If no valid lines were found, return early
    if len(valid_lines) == 0:
        print("No valid JSON entries found.")
        return None
   
    df = pd.DataFrame(valid_lines)    # Convert to a DataFrame

    # df = pd.read_json(input_file_path, lines=True, encoding_errors="ignore")
    print(f"Number of data samples: {len(df)}")
    logger.info(f"Number of data samples: {len(df)}")
    df.rename(columns={'result':'model_output'},inplace=True)
    df['task'] = df['task'].replace('rna_protein_interaction', 'ncRNAProteinInter')
    df['task'] = df['task'].replace('antibody_antigen', 'AntibodyAntigen')
    # Process entries with null labels
    # null_label_df = df[df['label'].isna()]
    # # null_label_df.to_json(f"{model_name}_result_label_null.json", orient='records', lines=True)
    
    # Remove data for _all task
    df = df[~df['task'].str.endswith('_all')]
    
    # Replace 'tf-h' with 'tf_h' and 'tf-m' with 'tf_m' in the 'task' column
    df['task'] = df['task'].str.replace('tf-h', 'tf_h')
    df['task'] = df['task'].str.replace('tf-m', 'tf_m')
    
    # Keep data if label is not null
    df = df[df['label'].notna()]
    df.reset_index(inplace=True, drop=True)
    
    # Convert to dictionary format for grouping
    data = df.to_dict(orient='records')
    
    # Group the data by 'task'
    grouped_data = defaultdict(list)
    for entry in data:
        task_name = entry['task'].split('-')[0]
        grouped_data[task_name].append(entry)
    
    return grouped_data


# Preprocess input data file
grouped_data = preprocess_input_data(input_file_path)
logger.info(f"Grouped data for tasks: {list(grouped_data.keys())}")
print(f"Grouped data for tasks: {list(grouped_data.keys())}")

# Read task type mapping
with open("register_tasks.json", "r") as f:
    task_type_data = json.load(f)
    
metrics = {}

# Loop over tasks
for task_name, task_entries in grouped_data.items():
    task_type = task_type_data[task_name]["type"]
    task_metrics = task_type_data[task_name]["metrics"]
    print(f"Prosessing {task_name} task...")

    if task_type == "regression":
        task_processed_data, label_values, result_values = process_regression_task(task_name, task_entries)

        if task_metrics == "spearman":
            metrics[task_name] = compute_spearman(label_values, result_values)

        elif task_metrics == "R2":
            metrics[task_name] = compute_R2(label_values, result_values)

        elif task_metrics == "mixed_score":
            metrics[task_name] = compute_mixed_score(label_values, result_values, threshold = 30)

    elif task_type == "binary classification":
        task_processed_data, label_classes, result_classes = process_binary_classification_task(task_name, task_entries)

        if task_metrics == "MCC":
            metrics[task_name] = compute_MCC(label_classes, result_classes)

        elif task_metrics == "Acc": 
            metrics[task_name] = compute_Acc(label_classes, result_classes)
    
    elif task_type == "multilabel regression":

        if task_name == "ProgrammableRNASwitches":
            metrics[task_name] = compute_R2_for_ProgrammableRNASwitches_task(task_name, task_entries)

        elif task_name == "enhancer_activity":
            metrics[task_name] = compute_PCC_for_enhancer_activity_task(task_name, task_entries)
    
    elif task_type == "multiclass classification":

        if task_name == "NoncodingRNAFamily":
            metrics[task_name] = compute_Acc_for_NoncodingRNAFamily_task(task_name, task_entries)
    
    elif task_type == "multilabel classification":

        if task_name == "FunctionEC":
            metrics[task_name] = compute_Fmax_for_FunctionEC_task(task_name, task_entries, ec_labels)

        elif task_name == "Modification":
            metrics[task_name] = compute_AUC_for_Modification_task(task_name, task_entries)
    
    print(f"The metrics {task_metrics} for task {task_name} is {str(metrics[task_name][task_metrics])}")
    logger.info(f"{task_metrics} of {task_name} is {str(metrics[task_name][task_metrics])}")


def round_and_scale_results(data, decimal_places=2, scale_factor=100):
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursive call if the value is a dictionary
            round_and_scale_results(value, decimal_places, scale_factor)
        elif isinstance(value, (float, int)):
            # Round and scale numeric values
            data[key] = float(round(value * scale_factor, decimal_places))


metrics_grouped_by_omics = defaultdict(dict)

for task_name, task_metrics in metrics.items():
    # Get the omics type from task_type_data
    omics = task_type_data[task_name]["omics"]
    
    # Scale the metrics
    scaled_metrics = task_metrics.copy()  # Make a copy to avoid modifying the original
    round_and_scale_results(scaled_metrics)  # Apply scaling to the metrics
    
    # Add the scaled metrics to the grouped dictionary
    metrics_grouped_by_omics[omics][task_name] = scaled_metrics


# Save the metrics (results) to a new JSON file
metrics_file_path = f"metrics_result/metrics_result_{model_name}_{OMICS}.json"
with open(metrics_file_path, "w") as outfile:
    json.dump(metrics_grouped_by_omics, outfile, indent=4)

logger.info(f"Metrics saved to {metrics_file_path}")
print(f"Metrics saved to {metrics_file_path}")