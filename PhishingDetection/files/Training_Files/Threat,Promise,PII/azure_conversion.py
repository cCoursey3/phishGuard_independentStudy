import json
import unicodedata

def sanitize_text(text):
    # Normalize the text to NFC (Normalization Form C) to ensure consistent encoding
    text = unicodedata.normalize('NFC', text)
    # Encode the text to bytes and then decode it back to a string to handle any encoding issues
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text

def convert_to_azure_format_with_encoding(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    azure_format = {"documents": []}
    for i, item in enumerate(data):
        sanitized_text = sanitize_text(item["sentence_text"])
        
        document = {
            "language": "en",
            "id": str(i + 1),
            "text": sanitized_text,
            "entities": []
        }
        
        if "annotations" in item:
            for annotation in item["annotations"]:
                if "start" in annotation and "end" in annotation:
                    start = annotation["start"]
                    end = annotation["end"]
                    entity_text = sanitize_text(annotation["text"])
                    
                    # Ensure offsets are correctly calculated after sanitization
                    corrected_start = sanitized_text.find(entity_text, start)
                    if corrected_start == -1:
                        print(f"Warning: Entity text not found in sanitized text for document id {document['id']}")
                        corrected_start = start  # fallback to original start if not found
                    
                    entity = {
                        "category": annotation["label"],
                        "offset": corrected_start,
                        "length": len(entity_text)
                    }
                    document["entities"].append(entity)
                else:
                    print(f"Warning: Missing 'start' or 'end' in annotation for document id {document['id']}")
        
        azure_format["documents"].append(document)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(azure_format, f, ensure_ascii=False, indent=4)

# Convert your JSON file with encoding handling
convert_to_azure_format_with_encoding(
    r'C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\files\Training_Files\Threat,Promise,PII\NER_For_Extraction\datasets\ner_pii_request.json',
    'azure_ner_formatted_sanitized.json'
)
