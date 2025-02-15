from flask import Flask
from datetime import timedelta
from src.config.config import Config
from src.handlers.routes import main, chat, admin

def create_app(config_object=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='../templates',  # Adjust template path since we moved the file
                static_folder='../static')       # Adjust static path if you have static files
    
    # Configure app
    app.config["CACHE_TYPE"] = "null"
    app.secret_key = config_object.SECRET_KEY
    app.permanent_session_lifetime = timedelta(hours=config_object.SESSION_LIFETIME_HOURS)
    
    # Register blueprints
    app.register_blueprint(main)
    app.register_blueprint(chat)
    app.register_blueprint(admin)
    
    return app 