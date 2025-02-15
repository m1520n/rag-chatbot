import re
import ollama
import logging
from datetime import datetime
from src.config.config import Config
from src.handlers.db_handler import db
from src.handlers.embedding_handler import embeddings
import json

#todo add translation from and to english from any language

#todo add compression of context to the llm 

logging.basicConfig(
    filename='chat_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_response(text):
    """Remove <think> sections and unwanted formatting from model responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_query_components_llm(query, conversation_history):
    """
    Use the model to extract possible multiple products, relevant attributes, 
    and special requirements. This is a more flexible approach to handle 
    scenarios where user may mention more than one product in a single query.
    """

    logger.info(f"Extracting query components for: {query}")

    prompt = f"""
    You are a helpful assistant that extracts structured information from user queries about building products.

    Please identify:
    1. All product categories the user is asking about 
       (e.g., ['garage doors', 'windows', 'interior doors']).
    2. Relevant attributes 
       (e.g., ['color', 'size', 'insulation', 'interior/exterior']).
    3. Any special requirements 
       (e.g., ['passive house compatible', 'security glass']).

    Return the answer strictly in valid JSON with keys: 
    "products" (list of strings), 
    "attributes" (list of strings), 
    "special_requirements" (list of strings).

    Example response:
    {{
      "products": ["garage doors", "windows"],
      "attributes": ["color", "insulation"],
      "special_requirements": ["passive house compatible"]
    }}

    Analyze the user query:
    {query}

    Conversation history:
    {conversation_history}
    """

    logger.info(f"Prompt to LLM (extract_query_components_llm): {prompt}")

    response = ollama.chat(
        model=Config.OLLAMA_MODEL, 
        messages=[{"role": "user", "content": prompt}]
    )

    logger.info(f"Raw LLM response: {response}")

    # Attempt to extract JSON from the response
    json_data = {"products": [], "attributes": [], "special_requirements": []}
    json_match = re.search(r'```json(.*?)```', response['message']['content'], re.DOTALL)
    
    if json_match:
        try:
            extracted_data = json.loads(json_match.group(1))
            json_data.update(extracted_data)
            logger.info(f"Successfully extracted JSON: {json_data}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
    else:
        # If no JSON is found, fallback to an empty structure
        logger.warning("No valid JSON found in LLM response for extraction")

    return json_data


def get_context_from_history(history):
    """Extract relevant context from conversation history, such as previously mentioned products."""
    context = []
    mentioned_products = set()
    last_query = None
    
    for message in history:
        if message['role'] == 'user':
            last_query = message['content']
        elif message['role'] == 'assistant':
            # Try to find any previously shared product links in the assistant messages
            urls = re.findall(rf'{Config.BASE_URL}/([^)\s]+)', message['content'])
            for url in urls:
                # Just a naive example: parse the link structure to get a display name
                product_name = url.split('-')[0].replace('-', ' ').title()
                if product_name not in mentioned_products:
                    mentioned_products.add(product_name)
                    context.append(f"Previously discussed: {product_name}")
    
    return {
        'context': "\n".join(context),
        'last_query': last_query,
        'mentioned_products': mentioned_products
    }

def generate_response(query, products_list, conversation_history):
    """
    Generate a chatbot response using Ollama with the conversation context.
    This is where we craft a persona and guidance for the LLM's final message.
    """
    history_info = get_context_from_history(conversation_history)
    context = history_info['context']
    last_query = history_info['last_query']

    # Determine if this is a follow-up
    is_followup = False
    if last_query:
        followup_indicators = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'the product']
        is_followup = any(indicator in query.lower() for indicator in followup_indicators)

    # Create a persona or brand voice in the system prompt
    system_prompt = f"""\
You are a professional but friendly sales representative for Aikon Distribution, specializing in windows and doors.
Speak naturally, with warmth and expertise. Be concise but helpful.

Important constraints:
1. ONLY discuss products explicitly provided in the product list below.
2. If a product is not in the list, say you don't have info on it.
3. NEVER invent product details; only discuss what's in the known product data or ask for clarification.
4. Provide exact product links from the product list whenever referencing a product.
5. If the user is vague or unclear, politely ask clarifying questions (e.g., "Are you looking for interior or exterior doors?").
6. Only reference previously discussed products if the user specifically asks about them or uses pronouns referencing them.
7. If multiple products are requested, handle them gracefully in the same response (e.g., mention each product and its link).
8. End your response in a helpful, friendly manner, offering further assistance.

Conversation history context:
{context}

{"Previous user question: " + last_query if is_followup else ""}

Product list for this conversation:
{products_list}

User's question: "{query}"
"""

    logger.info(f"System prompt for final LLM response:\n{system_prompt}")

    response = ollama.chat(
        model=Config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            # Optionally you can provide recent user messages or partial conversation
            # But we are packing it into system_prompt for demonstration
        ]
    )

    return clean_response(response['message']['content'])


def ask_for_clarification(unresolved_points):
    """
    Craft a response that asks the user for more details or clarifications
    based on unresolved points from the extraction step.
    """
    # Example usage: if the user said "door" but didn't specify interior/exterior
    # you can build clarifications dynamically here.
    clarifications = []
    for point in unresolved_points:
        if point == "door_type":
            clarifications.append(
                "Could you please clarify if you're looking for interior doors or exterior doors?"
            )
    # You can add more logic for other clarifications (dimensions, brand, etc.)
    return " ".join(clarifications)


def chat_with_bot(query, conversation_history=[]):
    """
    Main orchestrator: 
     1. Extract relevant info from the user's query,
     2. Determine if we need clarifications,
     3. Query the vector DB,
     4. Generate a final persona-based response.
    """
    logger.info(f"New chat request received: {query}")

    # 1. Extract structured info from user query
    extracted_info = extract_query_components_llm(query, conversation_history)
    products_mentioned = extracted_info["products"]  # multiple product categories
    attributes = extracted_info["attributes"]
    special_requirements = extracted_info["special_requirements"]

    logger.info(f"Products: {products_mentioned}")
    logger.info(f"Attributes: {attributes}")
    logger.info(f"Special requirements: {special_requirements}")

    # A minimal example of clarifications if user says "door" but no mention of interior/exterior
    # You can expand upon this logic
    clarifications_needed = []
    if "door" in [p.lower() for p in products_mentioned]:
        # If the user didn't specify interior/exterior, we might ask
        if not any(
            a.lower() in ["interior", "exterior", "inside", "outside"] 
            for a in attributes + special_requirements
        ):
            clarifications_needed.append("door_type")

    if len(products_mentioned) == 0:
        # If no product recognized at all, politely ask the user to clarify
        response_text = (
            "I’m not entirely sure which product you’re interested in. "
            "Could you let me know if you're looking for windows, doors, garage doors, or something else?"
        )
        return {
            "response": response_text,
            "debug_info": {
                "extracted_info": extracted_info,
                "query": query,
                "products_found": [],
                "attributes_found": [],
                "prompt": None,
            }
        }
    
    # If clarifications are needed, return a clarifying question
    if clarifications_needed:
        clarification_message = ask_for_clarification(clarifications_needed)
        return {
            "response": clarification_message,
            "debug_info": {
                "extracted_info": extracted_info,
                "query": query,
                "products_found": [],
                "attributes_found": [],
                "prompt": None
            }
        }

    # 2. For each recognized product, we can fetch from the vector DB
    #    We'll combine all returned results into a single list
    all_product_links = []
    for p in products_mentioned:
        vector = embeddings.encode_query(p)
        matched = db.search_products(vector, conversation_history)
        logger.info(f"Matched products: {matched}")
        all_product_links.extend(matched)
        logger.info(f"All product links: {all_product_links}")
    
    # Deduplicate
    all_product_links = list(set(all_product_links))

    if not all_product_links:
        return {
            "response": f"Sorry, I couldn't find any matching products for {products_mentioned}.",
            "debug_info": {
                "extracted_info": extracted_info,
                "query": query,
                "products_found": [],
                "attributes_found": attributes,
                "prompt": None,
            }
        }

    # 3. Generate the final answer from the LLM, injecting the relevant product links
    # We'll join all matching product links in one block
    product_list_for_prompt = "\n".join(all_product_links)

    final_bot_response = generate_response(query, product_list_for_prompt, conversation_history)

    # 4. Return the final structured response
    return {
        "response": final_bot_response,
        "debug_info": {
            "extracted_info": extracted_info,
            "query": query,
            "products_found": all_product_links,
            "attributes_found": attributes,
            "prompt": product_list_for_prompt
        }
    }
