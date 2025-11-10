import json
import re
import openai
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


openai.api_key = 'sk-proj-V5OXKEGTtZTujtZQrH86T3BlbkFJmWi9NpKKXVvVAYIBztik'

def extract_contextual_info(text):
    messages = [
        {"role": "system", "content": "You are an assistant that helps to extract information from email signatures."},
        {"role": "user", "content": f"Given the email signature below, extract the general organization as well as any people mentioned. If an entity is not present, return 'None' for that entity:\n\nEmail signature:\n{text}\n\nOutput format:\nGeneral Organization: [Extracted general organization]\nPerson: [Extracted person or 'None']"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    
    response_content = response.choices[0]['message']['content'].strip()
    #print(response_content)
    
    # Extract organization and person from the response content
    org_match = re.search(r"General Organization: (.*)", response_content)
    person_match = re.search(r"Person: (.*)", response_content)
    
    general_organization = org_match.group(1) if org_match else "None"
    person = person_match.group(1) if person_match else "None"

    return general_organization, person


