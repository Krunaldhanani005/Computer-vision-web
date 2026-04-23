import re

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())
