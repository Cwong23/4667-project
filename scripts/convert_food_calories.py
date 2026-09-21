"""
Parses the Food and Calories csv and creates a json based on that
"""

import csv
import json
import re


def fix_unicode_escapes(text):
    if not text:
        return text
    text = re.sub(r'\\\\(u[0-9a-fA-F]{4})', r'\\\1', text)
    text = re.sub(
        r'\\u([0-9a-fA-F]{4})',
        lambda m: chr(int(m.group(1), 16)),
        text
    )
    replacements = {
        '\u2019': "'",
        '\u2018': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2013': '-',
        '\u2014': '-',
    }
    for smart, plain in replacements.items():
        text = text.replace(smart, plain)
    return text


def convert_nutrition_csv_to_json(csv_file_path, json_file_path):
    data = []

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        fieldnames = ['food_item', 'serving_size', 'calories']
        csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)

        for row in csv_reader:
            final_food_name = fix_unicode_escapes(row['food_item'])

            gram_match = re.search(r'\((\d+)\s*g\)', row['serving_size'])
            grams = int(gram_match.group(1)) if gram_match else None

            cal_match = re.search(r'(\d+)', row['calories'])
            calories = int(cal_match.group(1)) if cal_match else None

            data.append({
                "food_item": final_food_name.strip(),
                "grams": grams,
                "calories": calories
            })

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


convert_nutrition_csv_to_json(
    r'C:\Users\chris\Downloads\archive\food_calories.csv', './food_calories.json')
