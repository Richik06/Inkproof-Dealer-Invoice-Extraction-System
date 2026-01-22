"""
Field validation and matching utilities
"""

import re
from typing import Optional, Tuple
from difflib import SequenceMatcher
import Levenshtein


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate fuzzy match score between two strings
    Returns score between 0 and 1
    """
    if not str1 or not str2:
        return 0.0
    
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    if str1 == str2:
        return 1.0
    
    lev_score = Levenshtein.ratio(str1, str2)
    seq_score = SequenceMatcher(None, str1, str2).ratio()
    
    return max(lev_score, seq_score)


def validate_dealer_name(name: str, master_list: list = None) -> Tuple[bool, float]:
    """
    Validate dealer name against master list
    Returns (is_valid, match_score)
    """
    if not name or len(name) < 3:
        return False, 0.0
    
    if not master_list:
        has_letters = bool(re.search(r'[a-zA-Z]', name))
        return has_letters, 0.8 if has_letters else 0.3
    
    best_match_score = 0.0
    for master_name in master_list:
        score = fuzzy_match_score(name, master_name)
        best_match_score = max(best_match_score, score)
    
    is_valid = best_match_score >= 0.90
    return is_valid, best_match_score


def validate_model_name(model: str, master_list: list = None) -> Tuple[bool, float]:
    """
    Validate model name
    Returns (is_valid, match_score)
    """
    if not model or len(model) < 2:
        return False, 0.0
    
    has_alphanum = bool(re.search(r'[a-zA-Z0-9]', model))
    if not has_alphanum:
        return False, 0.0
    
    if not master_list:
        return True, 0.85
    
    for master_model in master_list:
        if model.upper() == master_model.upper():
            return True, 1.0
        if model.upper() in master_model.upper() or master_model.upper() in model.upper():
            return True, 0.9
    
    return False, 0.5


def validate_horse_power(hp: Optional[int]) -> Tuple[bool, float]:
    """
    Validate horse power value
    Returns (is_valid, confidence)
    """
    if hp is None:
        return False, 0.0
    
    if 20 <= hp <= 200:
        return True, 0.95
    elif 10 <= hp <= 300:
        return True, 0.7
    else:
        return False, 0.3


def validate_asset_cost(cost: Optional[int]) -> Tuple[bool, float]:
    """
    Validate asset cost value
    Returns (is_valid, confidence)
    """
    if cost is None:
        return False, 0.0
    
    if 100000 <= cost <= 50000000:
        return True, 0.95
    elif 50000 <= cost <= 100000000:
        return True, 0.7
    else:
        return False, 0.3


def validate_bbox(bbox: list, image_size: Tuple[int, int]) -> Tuple[bool, float]:
    """
    Validate bounding box
    Returns (is_valid, confidence)
    """
    if not bbox or len(bbox) != 4:
        return False, 0.0
    
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, 0.3
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.2
    
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = width * height
    
    area_ratio = bbox_area / image_area
    if 0.001 < area_ratio < 0.3:
        return True, 0.9
    elif area_ratio <= 0.001:
        return False, 0.4
    else:
        return False, 0.5


def calculate_iou(bbox1: list, bbox2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r'[.,;:!?]', '', text)
    
    return text.strip()


def extract_numbers_from_text(text: str) -> list:
    """Extract all numbers from text"""
    if not text:
        return []
    
    text = text.replace(',', '')
    numbers = re.findall(r'\d+\.?\d*', text)
    
    return [float(n) for n in numbers]