from handlers.data_processor import clean_and_enhance_text, extract_product_type
from src.handlers.mysql_handler import MySQLHandler
from src.handlers.vector_handler import VectorHandler
from src.handlers.embedding_handler import EmbeddingHandler

class ProductService:
    """Service layer for coordinating product-related operations."""
    
    def __init__(self):
        """Initialize service with its dependencies."""
        self.mysql = MySQLHandler()
        self.vector_db = VectorHandler()
        self.embeddings = EmbeddingHandler()

    def index_all_products(self):
        """Index all active products from MySQL into the vector database."""
        products = self.mysql.fetch_active_products()
        if not products:
            print("❌ No products found in MySQL!")
            return

        for product in products:
            self._index_single_product(product)

        print("✅ Products indexed successfully!")

    def _index_single_product(self, product):
        """Index a single product into the vector database."""
        # Get fields
        name = product.get("name_en", "")
        descr = " ".join([
            product.get("descr_en", ""),
            product.get("descr2_en", "")
        ])
        tags = product.get("tags_en", "")

        # Create embedding and get metadata
        result = self.embeddings.create_product_embedding(name, descr, tags)
        
        product_id = str(product['id'])
        url = self.vector_db.add_product(product_id, result)
        print(f"📝 Indexing: {result['name_clean']} (Type: {result['product_type']})")

    def search_products(self, query, conversation_history=[]):
        """Search for products using semantic search."""
        query_vector = self.embeddings.encode_query(query)
        return self.vector_db.search_products(query_vector, conversation_history)

    def get_product(self, product_id):
        """Get product details from both MySQL and vector database."""
        mysql_data = self.mysql.get_product_by_id(product_id)
        vector_data = self.vector_db.get_product(product_id)
        
        if not mysql_data:
            return None
            
        return {
            **mysql_data,
            'vector_data': vector_data
        }

    def debug_index(self):
        """Print debug information about the vector index."""
        self.vector_db.debug_index()

    def preview_product_embedding(self, page=1, per_page=10):
        """Preview how products will be processed for embedding with pagination.
        
        Args:
            page (int): The page number (1-based)
            per_page (int): Number of items per page
        
        Returns:
            dict: Contains paginated preview data and pagination metadata
        """
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get total count first
        total_count = self.mysql.count_active_products()
        
        # Get paginated products
        products = self.mysql.fetch_active_products_paginated(offset, per_page)
        
        preview_data = []
        for product in products:
            name = product.get("name_en", "")
            tags = product.get("tags_en", "")
            description = " ".join([
                product.get("descr_en", ""),
                product.get("descr2_en", "")
            ])
             # Clean and enhance text
            name_clean = clean_and_enhance_text(name)
            description_clean = clean_and_enhance_text(description)
            tags_clean = clean_and_enhance_text(tags, is_tags=True)
            
            # Extract and add product type as additional context
            product_type = extract_product_type(name_clean, tags_clean)
            enhanced_name = f"{product_type} {name_clean}"

            # Get embedding data
            embedding_data = self.embeddings.create_product_embedding(enhanced_name, description_clean, tags_clean, product_type)
            if not embedding_data:
                continue

            # Get vector data if product is already indexed
            vector_data = self.vector_db.get_product(str(product['id']))
            
            preview_data.append({
                'id': product['id'],
                'original_data': {
                    'name': name,
                    'description': description,
                    'tags': tags
                },
                'processed_data': {
                    'name_clean': name_clean,
                    'product_type': product_type,
                    'description_clean': description_clean,
                    'tags_clean': tags_clean
                },
                'embedding_vector': embedding_data[:5] + ['...'],  # Show only first 5 dimensions
                'is_indexed': vector_data is not None
            })

        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1

        return {
            'items': preview_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev
            }
        }

# Create a singleton instance
product_service = ProductService() 