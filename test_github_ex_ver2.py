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
REQUESTS_PER_MINUTE = 3  # Very conservative rate
MIN_PAUSE = 20  # Minimum seconds between requests
BACKOFF_MULTIPLIER = 2
MAX_RETRIES = 10

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
    with open("./book.txt") as f:
        text = f.read()
        await process_with_rate_limit(graph_func.insert, text)

    # Perform queries
    try:
        print("Performing global search...")
        result1 = await process_with_rate_limit(
            graph_func.query,
            "What are the top themes in this story?"
        )
        print(result1)
        
        print("\nPerforming local search...")
        result2 = await process_with_rate_limit(
            graph_func.query,
            "What are the top themes in this story?",
            param=QueryParam(mode="local")
        )
        print(result2)

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the async main function
    asyncio.run(main())