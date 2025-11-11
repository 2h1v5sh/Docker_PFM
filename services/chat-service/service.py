import os
import google.generativeai as genai
import httpx
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('../..')

from shared.models import User

# --- Configuration ---
# 1. Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Vector DB (Milvus)
connections.connect("default", host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
COLLECTION_NAME = "financial_knowledge"
collection = Collection(COLLECTION_NAME)
try:
    collection.load()
except Exception as e:
    print(f"Warning: Could not load Milvus collection. Is it populated? Error: {e}")

# 3. Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class ChatService:

    async def _fetch_user_context(self, user_id: int, token: str) -> str:
        """Fetches the user's financial dashboard and transactions."""
        headers = {"Authorization": f"Bearer {token}"}
        context_parts = []
        
        # Define service URLs from env
        analytics_url = f"http://analytics-service:{os.getenv('ANALYTICS_SERVICE_PORT')}"
        tx_url = f"http://transaction-service:{os.getenv('TRANSACTION_SERVICE_PORT')}"
        
        try:
            async with httpx.AsyncClient() as client:
                # Get Dashboard Summary
                dashboard_res = await client.get(f"{analytics_url}/analytics/dashboard", headers=headers)
                if dashboard_res.status_code == 200:
                    context_parts.append(f"User's Dashboard Summary: {dashboard_res.json()}")
                
                # Get Last 5 Transactions
                tx_res = await client.get(f"{tx_url}/transactions?limit=5", headers=headers)
                if tx_res.status_code == 200:
                    context_parts.append(f"User's Last 5 Transactions: {tx_res.json()}")

            return "\n".join(context_parts)
        except Exception as e:
            print(f"Error fetching user context: {e}")
            return "Could not fetch user's financial data."

    def _fetch_knowledge_base(self, query: str) -> str:
        """Fetches relevant financial knowledge from Milvus."""
        try:
            # 1. Embed the user's query
            query_vector = embedding_model.encode(query).tolist()
            
            # 2. Search Milvus
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["text_chunk"]
            )
            
            # 3. Format results
            knowledge = "\n".join([hit.entity.get("text_chunk") for hit in results[0]])
            return knowledge
        except Exception as e:
            print(f"Error fetching from Milvus: {e}")
            return "Could not retrieve financial knowledge."

    async def generate_response(self, user: User, message: str, token: str) -> str:
        """Generates a context-aware response using RAG and Gemini."""
        
        # Step 1: Retrieve User Context (Your data)
        user_context = await self._fetch_user_context(user.id, token)
        
        # Step 2: Retrieve Financial Knowledge (Vector DB)
        financial_knowledge = self._fetch_knowledge_base(message)
        
        # Step 3: Augment (Build the Prompt)
        prompt = f"""
        You are a professional, helpful, and concise financial advisor for the PFM app.
        Your name is 'Fin.ai'.
        You must base your answers *only* on the context provided.
        If the answer isn't in the context, say "I do not have enough information to answer that."
        Analyze the user's data to give personalized insights.
        Keep your answers short and to the point.

        ---
        [USER'S FINANCIAL CONTEXT]
        {user_context}
        ---
        [FINANCIAL KNOWLEDGE BASE]
        {financial_knowledge}
        ---

        [USER'S QUESTION]
        {message}

        [YOUR ANSWER]
        """
        
        # Step 4: Generate (Call Gemini API)
        try:
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "I'm having trouble connecting to my AI brain right now. Please try again later."