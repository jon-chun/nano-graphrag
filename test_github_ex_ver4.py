from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv
import os
import time
import asyncio
import logging
from collections import deque
import aiohttp
import json

load_dotenv()

# Even more conservative settings
BATCH_SIZE = 1  # Process one request at a time
REQUEST_INTERVAL = 2  # Wait n seconds between requests
MAX_RETRIES = 5
MAX_CONCURRENT_REQUESTS = 1

class TokenBucket:
    def __init__(self, tokens_per_minute=1):
        self.tokens = tokens_per_minute
        self.max_tokens = tokens_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + time_passed * (self.max_tokens / 60))
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.max_tokens
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_update = now

class RequestQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.token_bucket = TokenBucket(tokens_per_minute=1)
        
    async def add_request(self, func, *args, **kwargs):
        await self.queue.put((func, args, kwargs))
    
    async def process_queue(self):
        while True:
            try:
                func, args, kwargs = await self.queue.get()
                await self.token_bucket.acquire()
                
                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        result = await func(*args, **kwargs)
                        self.queue.task_done()
                        return result
                    except Exception as e:
                        logging.warning(f"Request failed: {str(e)}")
                        retries += 1
                        if retries < MAX_RETRIES:
                            await asyncio.sleep(REQUEST_INTERVAL)
                
                logging.error(f"Max retries exceeded for request")
                self.queue.task_done()
                return None
            except Exception as e:
                logging.error(f"Error processing request: {str(e)}")
                self.queue.task_done()

request_queue = RequestQueue()

async def process_text_chunk(graph_func, chunk):
    try:
        return await request_queue.add_request(graph_func.ainsert, chunk)
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000):
    """Split text into smaller chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def main():
    try:
        # Initialize GraphRAG
        graph_func = GraphRAG(working_dir="./dickens")
        
        # Read and chunk the text
        with open("./book.txt") as f:
            text = f.read()
            chunks = chunk_text(text)
        
        # Process chunks with rate limiting
        queue_processor = asyncio.create_task(request_queue.process_queue())
        
        logging.info(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)}")
            await process_text_chunk(graph_func, chunk)
            await asyncio.sleep(REQUEST_INTERVAL)
        
        # Wait for all chunks to be processed
        await request_queue.queue.join()
        queue_processor.cancel()
        
        # Perform queries
        logging.info("Performing queries...")
        try:
            result1 = await request_queue.add_request(
                graph_func.aquery,
                "What are the top themes in this story?"
            )
            print("\nGlobal search result:", result1)
            
            await asyncio.sleep(REQUEST_INTERVAL)
            
            result2 = await request_queue.add_request(
                graph_func.aquery,
                "What are the top themes in this story?",
                param=QueryParam(mode="local")
            )
            print("\nLocal search result:", result2)
            
        except Exception as e:
            logging.error(f"Query error: {str(e)}")
    
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)