from src.handlers.mysql_handler import mysql_db
from src.handlers.chroma_handler import vector_db
from src.handlers.embedding_handler import embeddings

class DatabaseHandler:
    def __init__(self):
        """Initialize database handler that coordinates between MySQL and ChromaDB."""
        self.mysql = mysql_db
        self.vector_db = vector_db

    def index_products(self):
        """Index products from MySQL into the vector database."""
        products = self.mysql.fetch_active_products()
        if not products:
            print("❌ No products found in MySQL!")
            return

        for product in products:
            # Get fields
            name = product.get("name_en", "")
            descr = " ".join([
                product.get("descr_en", ""),
                product.get("descr2_en", "")
            ])
            tags = product.get("tags_en", "")

            # Create embedding and get metadata
            result = embeddings.create_product_embedding(name, descr, tags)
            
            product_id = str(product['id'])
            url = self.vector_db.add_product(product_id, result)
            print(f"📝 Indexing: {result['name_clean']} (Type: {result['product_type']})")

        print("✅ Products indexed successfully!")

    def debug_index(self):
        """Print the number of indexed products."""
        self.vector_db.debug_index()

    def search_products(self, query_vector, conversation_history=[]):
        """Search for products using vector similarity."""
        return self.vector_db.search_products(query_vector, conversation_history)

# Create a singleton instance
db = DatabaseHandler() 