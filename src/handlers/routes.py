from flask import Blueprint, render_template, request, jsonify, session
from src.handlers.chat_bot import chat_with_bot
from src.services.product_service import product_service

# Create blueprints for different parts of the application
main = Blueprint('main', __name__)
chat = Blueprint('chat', __name__)
admin = Blueprint('admin', __name__)

@main.route('/')
def home():
    """Home page route."""
    # Initialize empty conversation history for new sessions
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return render_template('index.html')

@chat.route('/chat', methods=['POST'])
def handle_chat():
    """Chat endpoint for handling messages."""
    message = request.json.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get conversation history from session
    conversation_history = session.get('conversation_history', [])
    
    # Call chat_with_bot with conversation history
    result = chat_with_bot(message, conversation_history)
    response = result["response"]
    debug_info = result["debug_info"]
    
    # Update conversation history
    conversation_history.append({
        'role': 'user',
        'content': message
    })
    conversation_history.append({
        'role': 'assistant',
        'content': response
    })
    
    # Keep only last 10 messages to prevent session from growing too large
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    # Save updated history back to session
    session['conversation_history'] = conversation_history
    
    return jsonify({
        'response': response,
        'debug_info': debug_info
    })

@admin.route('/')
@admin.route('/dashboard')
def admin_dashboard():
    """Display admin dashboard."""
    return render_template('admin/dashboard.html')

@admin.route('/admin/indexing')
def indexing_page():
    """Display indexing management page."""
    return render_template('admin/indexing.html')

@admin.route('/admin/indexing/status')
def get_indexing_status():
    """Get current indexing status."""
    try:
        status = product_service.get_indexing_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin.route('/admin/indexing/start', methods=['POST'])
def start_indexing():
    """Start the indexing process."""
    try:
        product_service.start_indexing()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin.route('/admin/indexing/progress')
def get_indexing_progress():
    """Get current indexing progress."""
    try:
        progress = product_service.get_indexing_progress()
        return jsonify(progress)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin.route('/admin/indexing/cleanup', methods=['POST'])
def cleanup_index():
    """Clean up the vector database."""
    try:
        if product_service.cleanup_index():
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to clean up index'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin.route('/admin/embeddings')
def list_embeddings():
    """Display embedding preview page."""
    return render_template('admin/embeddings.html')

@admin.route('/admin/embeddings/data')
def get_embeddings_data():
    """Return embedding preview data as JSON with pagination."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Parse filters
        filters = {}
        empty_fields = request.args.getlist('empty_fields[]')
        if empty_fields:
            filters['empty_fields'] = empty_fields
            
    except ValueError:
        return jsonify({'error': 'Invalid pagination parameters'}), 400

    preview_data = product_service.preview_product_embedding(
        page=page, 
        per_page=per_page,
        filters=filters
    )
    return jsonify(preview_data)

@admin.route('/admin/embeddings/<int:product_id>')
def show_embedding(product_id):
    """Display embedding preview for a specific product."""
    preview_data = product_service.preview_single_product_embedding(product_id)
    if not preview_data:
        return jsonify({'error': 'Product not found'}), 404
    return jsonify(preview_data)

@admin.route('/admin/embeddings/visualize')
def visualize_embeddings():
    """Display vector embeddings visualization."""
    return render_template('admin/visualize.html')

@admin.route('/admin/embeddings/data/vectors')
def get_vectors_data():
    """Return vector data for visualization."""
    embeddings, metadatas, ids = product_service.get_embeddings_for_visualization()
    return jsonify({
        'embeddings': embeddings,
        'metadatas': metadatas,
        'ids': ids
    }) 