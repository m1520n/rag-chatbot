import mysql.connector
from src.config.config import Config

class MySQLHandler:
    """Handler for MySQL database operations."""
    
    def __init__(self):
        """Initialize MySQL handler with configuration."""
        self.config = Config.MYSQL_CONFIG

    def _get_connection(self):
        """Create and return a new database connection."""
        return mysql.connector.connect(**self.config)

    def _execute_query(self, query, params=None, fetch=True):
        """Execute a query with proper connection handling."""
        connection = self._get_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            if not fetch:
                connection.rollback()
            return [] if fetch else False
        finally:
            cursor.close()
            connection.close()

    def fetch_active_products(self):
        """Fetch all active products from the database."""
        query = """
            SELECT id, name_en, descr_en, descr2_en, tags_en
            FROM products WHERE active = '1'
        """
        products = self._execute_query(query)
        print(f"🔍 Found {len(products)} active products")
        return products

    def get_product_by_id(self, product_id):
        """Fetch a single product by ID."""
        query = """
            SELECT id, name_en, descr_en, descr2_en, tags_en
            FROM products WHERE id = %s AND active = '1'
        """
        results = self._execute_query(query, (product_id,))
        return results[0] if results else None

    def update_product_metadata(self, product_id, metadata):
        """Update product metadata."""
        query = """
            UPDATE products 
            SET metadata = %s
            WHERE id = %s
        """
        return self._execute_query(query, (metadata, product_id), fetch=False)

    def count_active_products(self, filters=None):
        """Count total number of active products with optional filtering."""
        base_query = "SELECT COUNT(*) as count FROM products WHERE active = '1'"
        
        if filters and 'empty_fields' in filters:
            conditions = []
            for field in filters['empty_fields']:
                if field == 'description':
                    conditions.append("(descr_en IS NULL OR descr_en = '' OR descr2_en IS NULL OR descr2_en = '')")
                elif field == 'name':
                    conditions.append("(name_en IS NULL OR name_en = '')")
                elif field == 'tags':
                    conditions.append("(tags_en IS NULL OR tags_en = '')")
            
            if conditions:
                base_query += " AND (" + " OR ".join(conditions) + ")"

        result = self._execute_query(base_query)
        return result[0]['count'] if result else 0

    def fetch_active_products_paginated(self, offset, limit, filters=None):
        """Fetch active products with pagination and filtering.
        
        Args:
            offset (int): Number of records to skip
            limit (int): Number of records to return
            filters (dict): Dictionary of filters, e.g. {'empty_fields': ['description', 'tags']}
        """
        base_query = """
            SELECT id, name_en, descr_en, descr2_en, tags_en
            FROM products 
            WHERE active = '1'
        """
        
        if filters and 'empty_fields' in filters:
            conditions = []
            for field in filters['empty_fields']:
                if field == 'description':
                    conditions.append("(descr_en IS NULL OR descr_en = '' OR descr2_en IS NULL OR descr2_en = '')")
                elif field == 'name':
                    conditions.append("(name_en IS NULL OR name_en = '')")
                elif field == 'tags':
                    conditions.append("(tags_en IS NULL OR tags_en = '')")
            
            if conditions:
                base_query += " AND (" + " OR ".join(conditions) + ")"

        base_query += " ORDER BY id DESC LIMIT %s OFFSET %s"
        return self._execute_query(base_query, (limit, offset))

# Create a singleton instance
mysql_db = MySQLHandler() 