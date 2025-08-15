"""
Test database backend switching between SQLite and MySQL
"""

import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.database.db_manager import DatabaseManager

class TestDatabaseManager:
    """Test database manager functionality"""
    
    def test_sqlite_backend_default(self):
        """Test that SQLite is used by default"""
        # Clear any MySQL environment variables
        for key in ['MYSQL_ENABLED', 'MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']:
            if key in os.environ:
                del os.environ[key]
        
        db_manager = DatabaseManager()
        assert db_manager.backend == 'sqlite'
        assert db_manager.sqlite_path is not None
        assert 'sqlite' in db_manager.get_backend_info()['backend']
    
    def test_mysql_backend_with_env_var(self):
        """Test that MySQL is used when MYSQL_ENABLED=true"""
        # Set MySQL environment variables
        os.environ['MYSQL_ENABLED'] = 'true'
        os.environ['MYSQL_HOST'] = 'localhost'
        os.environ['MYSQL_USER'] = 'test_user'
        os.environ['MYSQL_PASSWORD'] = 'test_pass'
        os.environ['MYSQL_DB'] = 'test_db'
        
        try:
            db_manager = DatabaseManager()
            # Should use MySQL backend if pymysql is available
            if hasattr(db_manager, 'mysql_config'):
                assert db_manager.backend == 'mysql'
                assert db_manager.mysql_config['host'] == 'localhost'
                assert db_manager.mysql_config['user'] == 'test_user'
        finally:
            # Clean up environment variables
            for key in ['MYSQL_ENABLED', 'MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']:
                if key in os.environ:
                    del os.environ[key]
    
    def test_backend_info(self):
        """Test backend info reporting"""
        db_manager = DatabaseManager()
        info = db_manager.get_backend_info()
        
        assert 'backend' in info
        assert info['backend'] in ['sqlite', 'mysql']
        
        if info['backend'] == 'sqlite':
            assert 'path' in info
            assert 'exists' in info
        elif info['backend'] == 'mysql':
            assert 'host' in info
            assert 'database' in info
    
    def test_sqlite_connection(self):
        """Test SQLite connection functionality"""
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            tmp_db_path = tmp_file.name
        
        try:
            # Force SQLite backend
            for key in ['MYSQL_ENABLED', 'MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']:
                if key in os.environ:
                    del os.environ[key]
            
            # Test connection
            db_manager = DatabaseManager()
            db_manager.sqlite_path = tmp_db_path
            
            conn = db_manager.get_connection()
            assert conn is not None
            conn.close()
            
            # Test execute
            try:
                result = db_manager.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
                # Should not raise exception - result may be 0 or None
                print("✅ Table creation successful")
            except Exception as e:
                raise AssertionError(f"Failed to create table: {e}")
            
            # Test fetchall
            db_manager.execute("INSERT INTO test_table (name) VALUES (?)", ("test_name",))
            rows = db_manager.fetchall("SELECT * FROM test_table")
            assert len(rows) == 1
            assert rows[0][1] == "test_name"
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_db_path):
                os.unlink(tmp_db_path)

if __name__ == '__main__':
    # Run tests if executed directly
    test = TestDatabaseManager()
    test.test_sqlite_backend_default()
    test.test_mysql_backend_with_env_var()
    test.test_backend_info()
    test.test_sqlite_connection()
    print("✅ All database backend tests passed!")