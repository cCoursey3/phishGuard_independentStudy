import json
import os

# Define the path to the JSON file
json_path = r'C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\files\Training_Files\Threat,Promise,PII\NER_For_Extraction\datasets\ner.json'

# Define the directory where the text files will be saved
output_dir = r'C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\files\Training_Files\Threat,Promise,PII\NER_For_Extraction\datasets\output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the JSON file
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the row_data and save each to a separate text file
for i, item in enumerate(data):
    row_data = item.get('row_data', '')
    file_name = f'row_data_{i+1}.txt'
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(row_data)

print("Row data has been extracted and saved to text files.")
