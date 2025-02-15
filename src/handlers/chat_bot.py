import re
import ollama
from src.config.config import Config
from src.handlers.db_handler import db
from src.handlers.embedding_handler import embeddings

def clean_response(text):
    """Remove <think> sections and unwanted formatting from model responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def get_context_from_history(history):
    """Extract relevant context from conversation history."""
    context = []
    mentioned_products = set()
    last_query = None
    
    for message in history:
        if message['role'] == 'user':
            last_query = message['content']
        elif message['role'] == 'assistant':
            urls = re.findall(rf'{Config.BASE_URL}/([^)\s]+)', message['content'])
            for url in urls:
                product_name = url.split('-')[0].replace('-', ' ').title()
                if product_name not in mentioned_products:
                    mentioned_products.add(product_name)
                    context.append(f"Previously discussed: {product_name}")
    
    return {
        'context': "\n".join(context),
        'last_query': last_query,
        'mentioned_products': mentioned_products
    }

def generate_response(query, products, conversation_history=[]):
    """Generate a chatbot response using Ollama with conversation context."""
    history_info = get_context_from_history(conversation_history)
    context = history_info['context']
    last_query = history_info['last_query']
    
    # Determine if this is a follow-up question
    is_followup = False
    if last_query:
        followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'the product']
        is_followup = any(indicator in query.lower() for indicator in followup_indicators)
    
    prompt = f"""You are a professional seller specializing in windows and doors at Aikon Distribution.
    
    RULES:
    1. ONLY talk about products that are explicitly provided in the product list below
    2. If a product is not in the list, say you don't have information about it
    3. NEVER make up or invent product features, specifications, or details
    4. When mentioning products, use EXACTLY the same names and links as provided
    5. If asked about a product's details, ONLY discuss it if it's in the current product list
    6. If you don't have enough information, ask the customer for clarification
    7. ALWAYS include the exact product links when mentioning specific products
    8. If the user asks about a new product, focus on that product even if different from previous ones
    9. Only reference previous products if the user specifically asks about them

    Previous context:
    {context}
    
    {"Previous question: " + last_query if is_followup else ""}
    
    Available products for this conversation:
    {products}

    Current question: "{query}"

    Remember: 
    - Focus on answering the current question about products from the list above
    - If the user is asking about a new product, don't be biased by previously discussed products
    - Only reference previous products if the user specifically asks about them
    """
    
    response = ollama.chat(model=Config.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return clean_response(response['message']['content'])

def chat_with_bot(query, conversation_history=[]):
    """Handle chatbot conversation logic with context."""
    
    # Encode query and search for products
    query_vector = embeddings.encode_query(query)
    products = db.search_products(query_vector, conversation_history)

    if not products:
        return "I apologize, but I couldn't find any matching products in our current catalog. Could you please provide more details about what you're looking for? For example, are you interested in specific types of windows or doors?"
    
    # Add debug information
    print(f"🔍 Found {len(products)} relevant products")
    for product in products:
        print(f"  {product}")
    
    response = generate_response(query, "\n".join(products), conversation_history)
    
    # Verify response contains product links when products are mentioned
    if any(name in response for name in [p.split('**')[1].strip() for p in products]) and not '[View Product]' in response:
        response += "\n\nHere are the direct links to the products mentioned:\n" + "\n".join(products)
    
    return response
