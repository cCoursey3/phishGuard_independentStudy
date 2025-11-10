import re
import nltk
import spacy
from spacy.matcher import Matcher

# Download necessary NLTK data
#nltk.download('punkt')

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

def preprocess_email(email_text):
    clean_text = re.sub(r'<.*?>', '', email_text)
    return clean_text

def match_patterns(text, patterns):
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def is_likely_name(sentence):
    # Improved heuristic for identifying names
    name_pattern = r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+\.?$'
    return re.match(name_pattern, sentence.strip()) is not None

def extract_sender_name(text):
    # Patterns to identify the sender name
    name_patterns = [
        r'\bI am ([A-Z][a-z]+\s[A-Z][a-z]+)\b',
        r'\bI\'m ([A-Z][a-z]+\s[A-Z][a-z]+)\b',
        r'\bMy name is ([A-Z][a-z]+\s[A-Z][a-z]+)\b'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def is_signature_section(sentence, signature_patterns, body_context_patterns):
    if match_patterns(sentence, signature_patterns):
        return not match_patterns(sentence, body_context_patterns)
    return False

def segment_email(sentences):
    intro = []
    body = []
    signature = []
    body_started = False
    signature_start = False
    
    for i, sentence in enumerate(sentences):
        if match_patterns(sentence, intro_patterns) and not body_started:
            intro.append(sentence)
        elif is_signature_section(sentence, signature_patterns, body_context_patterns):
            signature_start = True
            signature.append(sentence)
        elif i == len(sentences) - 1 and is_likely_name(sentence):
            signature_start = True
            signature.append(sentence)
        else:
            if signature_start:
                signature.append(sentence)
            else:
                body.append(sentence)
                body_started = True
    
    return intro, body, signature

def add_custom_entities(nlp, text):
    # Define custom entities for cryptocurrency, social media, phone numbers, and addresses
    custom_patterns = [
        {"label": "CRYPTO", "pattern": [{"LOWER": "bitcoin"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "ethereum"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "crypto"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "cryptocurrency"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "btc"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "eth"}]},
        {"label": "CRYPTO", "pattern": [{"LOWER": "litecoin"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "twitter"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "facebook"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "linkedin"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "instagram"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "snapchat"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "tiktok"}]},
        {"label": "SOCIAL_MEDIA", "pattern": [{"LOWER": "social media"}]}
    ]

    matcher = Matcher(nlp.vocab)
    for pattern in custom_patterns:
        matcher.add(pattern["label"], [pattern["pattern"]])

    doc = nlp(text)
    matches = matcher(doc)
    
    # Convert matches to entities
    custom_entities = [(doc[start:end].text, nlp.vocab.strings[match_id]) for match_id, start, end in matches]
    
    # Extract phone numbers and addresses using regex
    phone_pattern = re.compile(r'\b(\+?\d{1,4}[\s-])?(?:\d{10}|\(?\d{3}\)?[\s-]\d{3}[\s-]\d{4})\b')
    address_pattern = re.compile(r'\b\d{1,5}\s(?:[A-Za-z0-9#]+\s){1,5}(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl|Square|Sq|Apartment|Apt|Suite|Ste|Building|Bldg|Unit|Fl|Floor|Room|Rm|Box|Mailbox|PO Box)\b,\s[A-Za-z]+\s[A-Za-z]{2}\s\d{5}\b')    
    
    
    phone_matches = [(match.group(), "PHONE") for match in phone_pattern.finditer(text)]
    address_matches = [(match.group(), "ADDRESS") for match in address_pattern.finditer(text)]
    
    # Combine all entities
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != "DATE"] + custom_entities + phone_matches + address_matches
    return entities

# Patterns
intro_patterns = [
    r'hi\b', r'hello\b', r'dear\b', r'greetings\b', r'to whom it may concern\b'
]
signature_patterns = [
    r'regards\b', r'sincerely\b', r'best\b', r'thank you\b', r'yours\b', r'Mrs\.\b', r'Dr\.\b', r'Professor\.\b'
    r'contact me at\b', r'phone\b', r'email\b', r'website\b', r'www\b', r'Mr\.\b', r'waiting to hear from you\b'
]
body_context_patterns = [
    r'difficulties\b', r'plan\b', r'take away\b', r'tried to kill\b', r'offer you\b'
]
