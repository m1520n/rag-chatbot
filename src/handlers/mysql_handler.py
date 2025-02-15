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
            FROM products WHERE active = 1
        """
        products = self._execute_query(query)
        print(f"🔍 Found {len(products)} active products")
        return products

    def get_product_by_id(self, product_id):
        """Fetch a single product by ID."""
        query = """
            SELECT id, name_en, descr_en, descr2_en, tags_en
            FROM products WHERE id = %s AND active = 1
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

# Create a singleton instance
mysql_db = MySQLHandler() 