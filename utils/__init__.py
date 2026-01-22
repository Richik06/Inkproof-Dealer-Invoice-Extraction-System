"""
Utils package for Document AI Extraction System
"""

from .validation import (
    fuzzy_match_score,
    validate_dealer_name,
    validate_model_name,
    validate_horse_power,
    validate_asset_cost,
    validate_bbox,
    calculate_iou,
    normalize_text,
    extract_numbers_from_text
)

__all__ = [
    'fuzzy_match_score',
    'validate_dealer_name',
    'validate_model_name',
    'validate_horse_power',
    'validate_asset_cost',
    'validate_bbox',
    'calculate_iou',
    'normalize_text',
    'extract_numbers_from_text'
]

__version__ = '1.0.0'