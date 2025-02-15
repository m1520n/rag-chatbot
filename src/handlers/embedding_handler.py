import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config import Config
from src.handlers.data_processor import clean_and_enhance_text, extract_product_type

class EmbeddingHandler:
    """Handler for text embedding operations."""
    
    def __init__(self):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.weights = Config.VECTOR_WEIGHTS

    def encode_query(self, query):
        """Encode a query into a vector."""
        try:
            return self.model.encode(query).tolist()
        except Exception as e:
            print(f"❌ Error encoding query: {str(e)}")
            return None

    def create_product_embedding(self, name, description, tags, product):
        """Create a weighted embedding for a product."""
        try:
            # Generate embeddings with enhanced text
            embedding_name = self.model.encode(name)
            embedding_descr = self.model.encode(description)
            embedding_tags = self.model.encode(tags)
            embedding_category = self.model.encode(product)

            # Combine vectors with adjusted weights
            final_embedding = (
                self.weights['NAME'] * embedding_name +
                self.weights['DESCRIPTION'] * embedding_descr +
                self.weights['TAGS'] * embedding_tags +
                self.weights['CATEGORY'] * embedding_category
            )

            # Normalize the final embedding
            final_embedding = final_embedding / np.linalg.norm(final_embedding)
            
            return final_embedding.tolist()
        except Exception as e:
            print(f"❌ Error creating product embedding: {str(e)}")
            return None

# Create a singleton instance
embeddings = EmbeddingHandler() 