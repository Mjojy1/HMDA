import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv('HMDA_DATA_DIR', '/mnt/data/HMDA_Project'))
REPORT_DIR = Path(os.getenv('HMDA_REPORT_DIR', str(ROOT / 'outputs/reports')))
YEARS = list(range(2021, 2026))
NUMERIC = ['loan_amount', 'income', 'combined_loan_to_value_ratio']
CATEGORICAL = ['debt_to_income_ratio', 'loan_type', 'loan_term']
FEATURES = NUMERIC + CATEGORICAL
TARGET_NOTE = 'Saved model target: action 1 (originated) or 2 (approved but not accepted) = positive; action 3 = denied. Origination alone is action 1.'
