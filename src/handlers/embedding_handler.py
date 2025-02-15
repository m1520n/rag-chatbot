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

    def create_product_embedding(self, name, description, tags):
        """Create a weighted embedding for a product."""
        try:
            # Clean and enhance text
            name_clean = clean_and_enhance_text(name)
            descr_clean = clean_and_enhance_text(description)
            tags_clean = clean_and_enhance_text(tags, is_tags=True)
            
            # Extract and add product type as additional context
            product_type = extract_product_type(name_clean, tags_clean)
            enhanced_name = f"{product_type} {name_clean}"
            
            # Generate embeddings with enhanced text
            embedding_name = self.model.encode(enhanced_name)
            embedding_descr = self.model.encode(descr_clean)
            embedding_tags = self.model.encode(tags_clean)
            embedding_category = self.model.encode(product_type)

            # Combine vectors with adjusted weights
            final_embedding = (
                self.weights['NAME'] * embedding_name +
                self.weights['DESCRIPTION'] * embedding_descr +
                self.weights['TAGS'] * embedding_tags +
                self.weights['CATEGORY'] * embedding_category
            )

            # Normalize the final embedding
            final_embedding = final_embedding / np.linalg.norm(final_embedding)
            
            return {
                'embedding': final_embedding.tolist(),
                'name_clean': name_clean,
                'tags_clean': tags_clean,
                'product_type': product_type
            }
        except Exception as e:
            print(f"❌ Error creating product embedding: {str(e)}")
            return None

# Create a singleton instance
embeddings = EmbeddingHandler() 