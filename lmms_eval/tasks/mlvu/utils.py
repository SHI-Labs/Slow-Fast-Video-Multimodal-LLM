from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys
from typing import List, Dict, Optional, Union
import re
import cv2
import numpy as np
from loguru import logger as eval_logger

# TASK_TYPES = ["TR", "AR", "VS", "NQA", "ER", "PQA", "SSC", "AO", "AC"]
TASK_TYPES = ['topic_reasoning', 'ego', 'plotQA', 'needle', 'order', 'count', 'anomaly_reco']

hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "mlvu.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mlvu_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]

def extract_question_and_options(input_str):

    lines = input_str.split("\n")
    lines = [_ for _ in lines if _ != ""]
    question, options = lines[0], lines[1:5]
    for i in range(len(options)):
        if '(A)' in options[i]: 
            options[i] = options[i].replace("(A)", "A.")
        elif '(B)' in options[i]: 
            options[i] = options[i].replace("(B)", "B.")
        elif '(C)' in options[i]: 
            options[i] = options[i].replace("(C)", "C.")
        elif '(D)' in options[i]: 
            options[i] = options[i].replace("(D)", "D.")
        else:
            raise ValueError
        
    return question, options


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
    system_prompt = "Select the best answer to the following multiple-choice question based on the video."
    
    question, options = extract_question_and_options(doc['question'])
    if not question.endswith("."):
        if not question.endswith("?"):
            question = question + "?"
            
    option_str = ""
    for idx, opt in enumerate(options):
        option_str = option_str + opt
        if idx < (len(options) - 1):
            option_str = option_str + "\n"
    
    full_prompt = system_prompt + "\n" + question + "\n" + option_str + "\n" + "Answer with the option's letter from the given choices directly."
    
    return full_prompt    


def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred[0]
    else:
        return s[0]


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    # print("****************",pred)
    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0
