import re
import ollama
from src.config.config import Config
from src.handlers.db_handler import db
from src.handlers.embedding_handler import embeddings

def clean_response(text):
    """Remove <think> sections and unwanted formatting from model responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_query_components_llm(query):
    """Use the model to extract product category and attributes."""
    
    prompt = f"""
    Extract key information from the following user query. Identify:
    1. The product category (e.g., 'garage door', 'window')
    2. Relevant attributes (e.g., 'color', 'size', 'insulation')
    3. Any additional requirements (e.g., 'passive house compatible')

    Example query: "I'm looking for garage doors for my passive house. What are the available colors?"
    
    Expected JSON output:
    {{
      "product": "garage doors",
      "attributes": ["color"],
      "special_requirements": ["passive house compatible"]
    }}

    Now analyze the following query:
    {query}
    """

    response = ollama.chat(
        model=Config.OLLAMA_MODEL, 
        messages=[{"role": "user", "content": prompt}]
    )

    #extract the json from the response
    # we need to only extract the json from the response
    # the response is a string and we need to extract the json from the string
    # the json is between ```json and ```
    # so we need to extract the json from the string

    # extract the json from the response
    import json
    json_response = re.search(r'```json(.*?)```', response['message']['content'], re.DOTALL)
    if json_response:
        extracted_data = json.loads(json_response.group(1))
    else:
        extracted_data = {"product": None, "attributes": [], "special_requirements": []}

    return extracted_data

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
    """Handles chatbot conversation logic with improved query understanding."""
    
    extracted_info = extract_query_components_llm(query)
    product = extracted_info["product"]
    attributes = extracted_info["attributes"]
    special_requirements = extracted_info["special_requirements"]

    if not product:
        return {
            "response": "I couldn't determine the product you're looking for. Could you clarify?",
            "debug_info": {
                "extracted_info": extracted_info,
                "query": query,
                "products_found": [],
                "prompt": None,
                "attributes_found": []
            }
        }

    # Step 1: Search for the main product category (e.g., "garage doors")
    product_vector = embeddings.encode_query(product)
    product_results = db.search_products(product_vector, conversation_history)

    if not product_results:
        return {
            "response": f"Sorry, I couldn't find any {product} in our catalog.",
            "debug_info": {
                "extracted_info": extracted_info,
                "query": query,
                "products_found": [],
                "prompt": None,
                "attributes_found": []
            }
        }

    # Step 2: Search for attributes within the found products
    attribute_results = []
    for attr in attributes:
        attr_vector = embeddings.encode_query(attr)
        attr_matches = db.search_products(attr_vector, conversation_history)
        
        # Filter attributes to match found product category
        filtered_attrs = [a for a in attr_matches if product in a.lower()]
        attribute_results.extend(filtered_attrs)

    # Merge product and attribute results
    combined_results = list(set(product_results + attribute_results))

    # Generate the prompt for debugging purposes
    history_info = get_context_from_history(conversation_history)
    context = history_info['context']
    last_query = history_info['last_query']
    
    is_followup = False
    if last_query:
        followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'the product']
        is_followup = any(indicator in query.lower() for indicator in followup_indicators)
    
    debug_prompt = f"""You are a professional seller specializing in windows and doors at Aikon Distribution.
    
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
    {combined_results}

    Current question: "{query}"
    """

    # Step 3: Generate chatbot response
    response = generate_response(query, "\n".join(combined_results), conversation_history)

    return {
        "response": response,
        "debug_info": {
            "extracted_info": extracted_info,
            "query": query,
            "products_found": combined_results,
            "prompt": debug_prompt,
            "attributes_found": attribute_results
        }
    }
