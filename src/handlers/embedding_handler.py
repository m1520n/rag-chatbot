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
            if not isinstance(query, str):
                print(f"❌ Query must be a string, got {type(query)}")
                return None
                
            # Clean and enhance the query text
            clean_query = clean_and_enhance_text(query)
            if not clean_query:
                print("❌ Query text is empty after cleaning")
                return None
                
            print(f"📝 Cleaned query: {clean_query}")
            
            # Generate embedding for the cleaned query
            embedding = self.model.encode(clean_query)
            if not isinstance(embedding, np.ndarray):
                print(f"❌ Expected numpy array from model, got {type(embedding)}")
                return None
                
            # Ensure we have a 1D array
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # Normalize the embedding
            normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            # Convert to list and ensure it's in the correct format
            return normalized_embedding[0].tolist()
        except Exception as e:
            print(f"❌ Error encoding query: {str(e)}")
            print(f"Query: {query}")
            print(f"Clean query: {clean_query if 'clean_query' in locals() else 'not cleaned yet'}")
            return None

    def create_product_embedding(self, name, description, tags, product_type):
        """Create a weighted embedding for a product."""
        try:
            # Clean and enhance the input texts
            clean_name = clean_and_enhance_text(name)
            clean_descr = clean_and_enhance_text(description)
            clean_tags = clean_and_enhance_text(tags)
            product_type = extract_product_type(clean_name, clean_descr)

            # Generate embeddings with enhanced text
            embedding_name = self.model.encode(clean_name)
            embedding_descr = self.model.encode(clean_descr)
            embedding_tags = self.model.encode(clean_tags)
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
                'name_clean': clean_name,
                'description_clean': clean_descr,
                'tags_clean': clean_tags,
                'product_type': product_type
            }
        except Exception as e:
            print(f"❌ Error creating product embedding: {str(e)}")
            return None

# Create a singleton instance
embeddings = EmbeddingHandler() 