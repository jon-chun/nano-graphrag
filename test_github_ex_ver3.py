from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv
import os
import time
from openai import RateLimitError
import asyncio
import logging

# Load environment variables from .env file
load_dotenv()

# Configure rate limiting
REQUESTS_PER_MINUTE = 3
MIN_PAUSE = 20
BACKOFF_MULTIPLIER = 2
MAX_RETRIES = 10

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
    
class RateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.reset_time = time.time() + 60

    async def wait(self):
        now = time.time()
        
        # Ensure minimum pause between requests
        time_since_last = now - self.last_request_time
        if time_since_last < MIN_PAUSE:
            await asyncio.sleep(MIN_PAUSE - time_since_last)

        # Reset counter if minute has passed
        if now > self.reset_time:
            self.request_count = 0
            self.reset_time = now + 60

        # Wait if we've hit our rate limit
        if self.request_count >= REQUESTS_PER_MINUTE:
            wait_time = self.reset_time - now
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.reset_time = time.time() + 60

        self.request_count += 1
        self.last_request_time = time.time()

rate_limiter = RateLimiter()

async def process_with_rate_limit(func, *args, **kwargs):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            await rate_limiter.wait()
            # If the function is a coroutine, await it
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except RateLimitError:
            wait_time = MIN_PAUSE * (BACKOFF_MULTIPLIER ** retries)
            logging.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded")

async def main():
    # Initialize GraphRAG
    graph_func = GraphRAG(working_dir="./dickens")

    # Process the text
    try:
        with open("./book.txt") as f:
            text = f.read()
            # Use ainsert instead of insert
            await process_with_rate_limit(graph_func.ainsert, text)

        print("Performing global search...")
        # Use aquery instead of query
        result1 = await process_with_rate_limit(
            graph_func.aquery,
            "What are the top themes in this story?"
        )
        print(result1)
        
        print("\nPerforming local search...")
        result2 = await process_with_rate_limit(
            graph_func.aquery,
            "What are the top themes in this story?",
            param=QueryParam(mode="local")
        )
        print(result2)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error details: {e}", exc_info=True)

def run_async():
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error details: {e}", exc_info=True)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_async()