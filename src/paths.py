from pathlib import Path
import os

PATRENT_DIR =  Path(__file__).parent.resolve().parent
DATA_DIR = PATRENT_DIR / 'data_all'
RAW_DATA_DIR = PATRENT_DIR / 'data_all' / 'raw'
TRANSFORMED_DATA_DIR = PATRENT_DIR / 'data_all' / 'transformed'
MODEL_DIR = PATRENT_DIR / 'trained models'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():    
    os.mkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)

if not Path(MODEL_DIR).exists():
    os.mkdir(MODEL_DIR)