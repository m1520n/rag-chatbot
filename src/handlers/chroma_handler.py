import chromadb
from src.config.config import Config
from src.handlers.data_processor import clean_url_string

class ChromaHandler:
    """Handler for vector database operations using ChromaDB."""
    
    def __init__(self):
        """Initialize ChromaDB handler."""
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(Config.CHROMA_COLLECTION_NAME)

    def add_product(self, product_id, embedding_result):
        """Add a product to the vector database."""
        try:
            print(f"📝 Embedding result: {embedding_result}")
            url = f"{Config.BASE_URL}/{embedding_result['name_clean'].replace(' ', '-').lower()}-{product_id}"
            
            self.collection.add(
                ids=[str(product_id)],
                embeddings=[embedding_result['embedding']],
                metadatas=[{
                    "name": embedding_result['name_clean'],
                    "url": url,
                    "tags": embedding_result['tags_clean'],
                    "product_type": embedding_result['product_type'],
                    "description": embedding_result['description_clean']
                }]
            )
            return url
        except Exception as e:
            print(f"❌ Error adding product to vector database: {str(e)}")
            return None

    def search_products(self, query_vector, conversation_history=[]):
        """Search for products using vector similarity."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=Config.SEARCH_RESULTS_LIMIT
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            products = []
            scores = results["distances"][0]
            min_score = min(scores)
            max_score = max(scores)
            
            threshold_multiplier = (Config.SEARCH_THRESHOLD_MULTIPLIER['FOLLOW_UP'] 
                                if len(conversation_history) > 0 
                                else Config.SEARCH_THRESHOLD_MULTIPLIER['NEW_QUERY'])
            threshold = min_score + (max_score - min_score) * threshold_multiplier

            print(f"📝 Threshold: {threshold}")

            for i in range(len(results["ids"][0])):
                score = scores[i]
                if score > threshold:
                    continue

                product_id = results['ids'][0][i]
                metadata = results["metadatas"][0][i]
                
                products.append({
                    'id': product_id,
                    'score': score,
                    'metadata': {
                        'name_clean': metadata['name'],
                        'description_clean': metadata['description'],
                        'tags_clean': metadata.get('tags', ''),
                        'product_type': metadata.get('product_type', '')
                    }
                })

            return products
        except Exception as e:
            print(f"❌ Error searching vector database: {str(e)}")
            return []

    def get_product(self, product_id):
        """Get a product from the vector database by ID."""
        try:
            result = self.collection.get(
                ids=[str(product_id)],
                include=['metadatas', 'embeddings']
            )
            if result['ids']:
                return {
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0]
                }
            return None
        except Exception as e:
            print(f"❌ Error fetching product from vector database: {str(e)}")
            return None

    def get_all_embeddings(self):
        """Get all embeddings with metadata."""
        try:
            results = self.collection.get(
                include=['embeddings', 'metadatas', 'ids']
            )
            return results
        except Exception as e:
            print(f"❌ Error getting all embeddings: {str(e)}")
            return None

    def count_indexed_products(self):
        """Count the number of products in the vector database."""
        try:
            return len(self.collection.get()['ids'])
        except Exception as e:
            print(f"❌ Error counting indexed products: {str(e)}")
            return 0

    def cleanup_index(self):
        """Remove all entries from the vector database."""
        try:
            self.client.delete_collection(Config.CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(Config.CHROMA_COLLECTION_NAME)
            print("✅ Vector database cleaned up successfully")
            return True
        except Exception as e:
            print(f"❌ Error cleaning up vector database: {str(e)}")
            return False

    def remove_product(self, product_id):
        """Remove a single product from the vector database."""
        try:
            self.collection.delete(ids=[str(product_id)])
            print(f"✅ Product {product_id} removed from vector database")
            return True
        except Exception as e:
            print(f"❌ Error removing product {product_id}: {str(e)}")
            return False

    def remove_products(self, product_ids):
        """Remove multiple products from the vector database."""
        try:
            self.collection.delete(ids=[str(pid) for pid in product_ids])
            print(f"✅ {len(product_ids)} products removed from vector database")
            return True
        except Exception as e:
            print(f"❌ Error removing products: {str(e)}")
            return False

# Create a singleton instance
vector_db = ChromaHandler() 