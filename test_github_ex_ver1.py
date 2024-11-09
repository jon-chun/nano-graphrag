from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv
import os
import time
import openai
from openai import RateLimitError

# Load environment variables from .env file
load_dotenv()

# Configure retry settings
MAX_RETRIES = 5
RETRY_DELAY = 60  # seconds

def retry_with_backoff(func):
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError:
                wait_time = RETRY_DELAY * (2 ** retries)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
        raise Exception("Max retries exceeded")
    return wrapper

@retry_with_backoff
def process_text(graph_func, text):
    return graph_func.insert(text)

@retry_with_backoff
def perform_query(graph_func, query, param=None):
    return graph_func.query(query, param=param)

# Initialize GraphRAG
graph_func = GraphRAG(working_dir="./dickens")

# Process the text
with open("./book.txt") as f:
    process_text(graph_func, f.read())

# Perform queries with retries
try:
    # Global search
    print("Performing global search...")
    result1 = perform_query(graph_func, "What are the top themes in this story?")
    print(result1)
    
    # Local search
    print("\nPerforming local search...")
    result2 = perform_query(
        graph_func, 
        "What are the top themes in this story?", 
        param=QueryParam(mode="local")
    )
    print(result2)

except Exception as e:
    print(f"Error occurred: {str(e)}")