import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
    SESSION_LIFETIME_HOURS = 1

    # MySQL Configuration
    MYSQL_CONFIG = {
        "host": os.getenv('MYSQL_HOST', 'localhost'),
        "user": os.getenv('MYSQL_USER'),
        "password": os.getenv('MYSQL_PASSWORD'),
        "database": os.getenv('MYSQL_DATABASE')
    }

    # LLM Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'deepseek-r1:7b')

    # Vector DB Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'products')

    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')

    # Vector Search Configuration
    VECTOR_WEIGHTS = {
        'NAME': 1.5,        # Increased name weight
        'DESCRIPTION': 0.8, # Decreased description weight
        'TAGS': 4.0,       # Significantly increased tags weight
        'CATEGORY': 2.0    # Added explicit category weight
    }

    # Search Configuration
    SEARCH_RESULTS_LIMIT = 10
    SEARCH_THRESHOLD_MULTIPLIER = {
        'FOLLOW_UP': 0.3,
        'NEW_QUERY': 0.2
    }

    # Product Types Configuration
    PRODUCT_TYPES = {
        'garage': ['garage', 'garage door', 'garage gates'],
        'window': ['window', 'windows', 'okno', 'okna'],
        'door': ['door', 'doors', 'drzwi'],
        'gate': ['gate', 'gates', 'brama', 'bramy']
    }

    # URL Configuration
    BASE_URL = "https://aikondistribution.com/products" 
