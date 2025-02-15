from bs4 import BeautifulSoup
import re
from config.config import Config

def clean_html(text):
    """Remove HTML tags from text."""
    return BeautifulSoup(text, 'html.parser').get_text()

def clean_url_string(text):
    """Clean text for use in URLs."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r'\s+', '-', cleaned)
    return cleaned

def clean_and_enhance_text(text, is_tags=False):
    """Clean and enhance text for better embedding."""
    text = clean_html(text).strip()
    if is_tags:
        # Convert comma-separated tags to space-separated and repeat important ones
        tags = [t.strip() for t in text.split(",") if t.strip()]
        # Repeat important category tags to increase their weight
        enhanced_tags = []
        for tag in tags:
            tag_lower = tag.lower()
            if any(key in tag_lower for key in ['garage', 'window', 'door', 'gate']):
                enhanced_tags.extend([tag] * 3)  # Repeat important tags
            else:
                enhanced_tags.append(tag)
        return " ".join(enhanced_tags)
    return text

def extract_product_type(name, tags):
    """Extract product type/category from name and tags."""
    combined_text = (name + " " + tags).lower()
    detected_types = []
    
    for category, keywords in Config.PRODUCT_TYPES.items():
        if any(keyword in combined_text for keyword in keywords):
            detected_types.append(category)
    
    return " ".join(detected_types) if detected_types else "other"

