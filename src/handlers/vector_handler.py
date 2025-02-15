import chromadb
from src.config.config import Config
from src.handlers.data_processor import clean_url_string

class VectorHandler:
    """Handler for vector database operations using ChromaDB."""
    
    def __init__(self):
        """Initialize vector database handler."""
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(Config.CHROMA_COLLECTION_NAME)

    def add_product(self, product_id, embedding_result):
        """Add a product to the vector database."""
        try:
            url = f"{Config.BASE_URL}/{embedding_result['name_clean'].replace(' ', '-').lower()}-{product_id}"
            
            self.collection.add(
                ids=[str(product_id)],
                embeddings=[embedding_result['embedding']],
                metadatas=[{
                    "name": embedding_result['name_clean'],
                    "url": url,
                    "tags": embedding_result['tags_clean'],
                    "product_type": embedding_result['product_type'],
                    "description": embedding_result['description']
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

            for i in range(len(results["ids"][0])):
                score = scores[i]
                if score > threshold:
                    continue

                name = results["metadatas"][0][i]["name"]
                product_id = results['ids'][0][i]
                url_name = clean_url_string(name)
                url = f"{Config.BASE_URL}/{url_name}-{product_id}"
                
                products.append(f"- **{name}** [View Product]({url})")

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

    def debug_index(self):
        """Print debug information about the vector index."""
        try:
            results = self.collection.get()
            print(f"Indexed products: {len(results['ids'])}")
            for i in range(min(5, len(results["ids"]))):  # Show first 5 products
                print(f"ID: {results['ids'][i]}, Name: {results['metadatas'][i]['name']}")
        except Exception as e:
            print(f"❌ Error accessing vector database: {str(e)}")

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
            # Delete the entire collection
            self.client.delete_collection(Config.CHROMA_COLLECTION_NAME)
            
            # Recreate the collection
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