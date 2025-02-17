from src.handlers.data_processor import clean_and_enhance_text, extract_product_type
from src.handlers.mysql_handler import MySQLHandler
from src.handlers.chroma_handler import ChromaHandler
from src.handlers.embedding_handler import EmbeddingHandler

class ProductService:
    """Service layer for coordinating product-related operations."""
    
    def __init__(self):
        """Initialize service with its dependencies."""
        self.mysql = MySQLHandler()
        self.vector_db = ChromaHandler()
        self.embeddings = EmbeddingHandler()
        self._indexing_status = {
            'status': 'idle',
            'progress': 0,
            'current_product': None,
            'total_products': 0,
            'processed_products': 0,
            'last_indexed': None,
            'error': None,
            'errors': []  # List to store individual product errors
        }

    def get_indexing_status(self):
        """Get current indexing status and statistics."""
        total_products = self.mysql.count_active_products()
        indexed_products = self.vector_db.count_indexed_products()
        
        return {
            'total_products': total_products,
            'indexed_products': indexed_products,
            'last_indexed': self._indexing_status['last_indexed']
        }

    def get_indexing_progress(self):
        """Get current indexing progress."""
        return self._indexing_status

    def start_indexing(self):
        """Start the indexing process in a background thread."""
        if self._indexing_status['status'] == 'in_progress':
            raise Exception('Indexing is already in progress')

        import threading
        thread = threading.Thread(target=self._run_indexing)
        thread.daemon = True
        thread.start()

    def _run_indexing(self):
        """Run the indexing process."""
        try:
            # Reset status
            self._indexing_status.update({
                'status': 'in_progress',
                'progress': 0,
                'current_product': None,
                'error': None,
                'errors': []
            })

            # Clean up existing index
            if not self.vector_db.cleanup_index():
                raise Exception("Failed to clean up existing index")

            # Get all products
            products = self.mysql.fetch_active_products()
            total = len(products)
            self._indexing_status['total_products'] = total

            for i, product in enumerate(products, 1):
                try:
                    # Update status
                    self._indexing_status.update({
                        'current_product': f"Product {product['id']}",
                        'progress': int((i / total) * 100),
                        'processed_products': i
                    })

                    # Index the product
                    self._index_single_product(product)

                except Exception as e:
                    error_msg = f"Error indexing product {product['id']}: {str(e)}"
                    print(error_msg)
                    self._indexing_status['errors'].append(error_msg)
                    # Continue with next product

            # Update final status
            from datetime import datetime
            self._indexing_status.update({
                'status': 'completed',
                'progress': 100,
                'current_product': None,
                'last_indexed': datetime.now().isoformat(),
                'error': None if not self._indexing_status['errors'] else f"Completed with {len(self._indexing_status['errors'])} errors"
            })

        except Exception as e:
            self._indexing_status.update({
                'status': 'error',
                'error': str(e),
                'progress': 0
            })

    def index_all_products(self):
        """Index all active products from MySQL into the vector database."""
        products = self.mysql.fetch_active_products()
        if not products:
            print("❌ No products found in MySQL!")
            return

        for product in products:
            self._index_single_product(product)

        print("✅ Products indexed successfully!")

    def _prepare_product_data(self, product, create_embedding=True):
        """Prepare product data for embedding.
        
        Args:
            product (dict): Raw product data from MySQL
            create_embedding (bool): Whether to create embedding vector
            
        Returns:
            tuple: (original_data, processed_data, embedding_data)
            where:
                original_data (dict): Original product fields
                processed_data (dict): Cleaned and processed text
                embedding_data (dict): Data ready for vector database (if create_embedding=True)
        """
        # Extract original fields
        name = product.get("name_en", "")
        description = " ".join([
            product.get("descr_en", ""),
            product.get("descr2_en", "")
        ])
        tags = product.get("tags_en", "")
        
        # Clean and process text
        name_clean = clean_and_enhance_text(name)
        description_clean = clean_and_enhance_text(description)
        tags_clean = clean_and_enhance_text(tags, is_tags=True)
        
        # Extract product type
        product_type = extract_product_type(name_clean, tags_clean)
        
        original_data = {
            'name': name,
            'description': description,
            'tags': tags
        }
        
        processed_data = {
            'name_clean': name_clean,
            'product_type': product_type,
            'description_clean': description_clean,
            'tags_clean': tags_clean
        }
        
        if create_embedding:
            enhanced_name = f"{product_type} {name_clean}"
            # Create embedding
            embedding_result = self.embeddings.create_product_embedding(
                enhanced_name, 
                description_clean, 
                tags_clean, 
                product_type
            )
            
            embedding_data = embedding_result if embedding_result else None
        else:
            embedding_data = None
        
        return original_data, processed_data, embedding_data

    def _index_single_product(self, product):
        """Index a single product into the vector database."""
        original_data, processed_data, embedding_data = self._prepare_product_data(product, create_embedding=True)
        
        if not embedding_data:
            raise Exception("Failed to create embedding for product")
        
        product_id = str(product['id'])
        url = self.vector_db.add_product(product_id, embedding_data)
        print(f"📝 Indexing: {processed_data['name_clean']} (Type: {processed_data['product_type']})")

    def search_products(self, query, conversation_history=[]):
        """Search for products using semantic search."""
        try:
            # Create embedding for the search query
            query_vector = self.embeddings.encode_query(query)
            if query_vector is None:
                print("❌ Failed to create query embedding")
                return []
            
            # Search products using the embedding
            results = self.vector_db.search_products(query_vector, conversation_history)
            return results
        except Exception as e:
            print(f"❌ Error searching products: {str(e)}")
            return []

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

    def preview_product_embedding(self, page=1, per_page=10, filters=None):
        """Preview how products will be processed for embedding with pagination."""
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get total count first
        total_count = self.mysql.count_active_products(filters)
        
        # Get paginated products
        products = self.mysql.fetch_active_products_paginated(offset, per_page, filters)
        
        preview_data = []
        for product in products:
            # Get clean data without creating embeddings
            original_data, processed_data, _ = self._prepare_product_data(product, create_embedding=False)
            
            # Get vector data if product is already indexed
            vector_data = self.vector_db.get_product(str(product['id']))
            
            preview_data.append({
                'id': product['id'],
                'original_data': original_data,
                'processed_data': processed_data,
                'embedding_vector': vector_data['embedding'][:5] + ['...'] if vector_data else [],
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

    def preview_single_product_embedding(self, product_id):
        """Preview embedding for a single product."""
        product = self.mysql.get_product_by_id(product_id)
        if not product:
            return None
            
        # Get clean data without creating embeddings
        original_data, processed_data, _ = self._prepare_product_data(product, create_embedding=False)

        # Get vector data if product is already indexed
        vector_data = self.vector_db.get_product(str(product_id))
        
        return {
            'id': product_id,
            'original_data': original_data,
            'processed_data': processed_data,
            'embedding_vector': vector_data['embedding'][:5] + ['...'] if vector_data else [],
            'is_indexed': vector_data is not None
        }

    def cleanup_index(self):
        """Clean up the vector database."""
        return self.vector_db.cleanup_index()

    def remove_product(self, product_id):
        """Remove a product from the vector database."""
        return self.vector_db.remove_product(product_id)

    def remove_products(self, product_ids):
        """Remove multiple products from the vector database."""
        return self.vector_db.remove_products(product_ids)

    def get_embeddings_for_visualization(self):
        """Get all embeddings with metadata for visualization."""
        try:
            results = self.vector_db.get_all_embeddings()
            if not results or not results['ids']:
                return [], [], []
                
            return (
                results['embeddings'],
                results['metadatas'],
                results['ids']
            )
        except Exception as e:
            print(f"❌ Error getting embeddings for visualization: {str(e)}")
            return [], [], []

# Create a singleton instance
product_service = ProductService() 