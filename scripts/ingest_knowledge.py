import os
import sys
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "financial_knowledge"
EMBEDDING_DIM = 384  # Dimension for 'all-MiniLM-L6-v2'
MODEL_NAME = 'all-MiniLM-L6-v2'

# 1. Sample Knowledge Base (Expand this significantly)
knowledge_data = [
    "A Systematic Investment Plan (SIP) is a method of investing in mutual funds. It allows investors to invest a fixed amount of money at regular intervals, such as monthly or quarterly.",
    "The 50/30/20 rule is a simple budgeting framework. It suggests allocating 50% of your income to Needs (rent, groceries, utilities), 30% to Wants (dining out, entertainment), and 20% to Savings and Debt Repayment.",
    "A mutual fund is an investment vehicle that pools money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities.",
    "Debt-to-income ratio (DTI) is your total monthly debt payments divided by your gross monthly income. Lenders use it to assess your ability to manage monthly payments and repay debts.",
    "An Emergency Fund is a stash of money set aside to cover unexpected financial emergencies, such as job loss, medical expenses, or car repairs. It is recommended to have 3-6 months' worth of living expenses saved.",
    "Compounding is the process where your investment returns themselves begin to earn returns, often described as 'interest on your interest'.",
    "A Fixed Deposit (FD) is a financial instrument provided by banks which provides investors with a higher rate of interest than a regular savings account, until the given maturity date.",
    "A Credit Score is a number between 300-900 that depicts a customer's creditworthiness. A higher score increases your chances of getting a loan.",
    "Asset allocation is an investment strategy that aims to balance risk and reward by apportioning a portfolio's assets according to an individual's goals, risk tolerance, and investment horizon."
]

def main():
    # 2. Initialize Embedding Model
    print(f"Loading embedding model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have network access or the model is cached.")
        return

    # 3. Connect to Milvus
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        print("Please ensure Milvus is running and accessible.")
        return

    # 4. Define Collection Schema
    if utility.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, description="Financial Knowledge Base")
    
    print(f"Creating collection: {COLLECTION_NAME}...")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # 5. Create Index
    print("Creating index for embedding field...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")

    # 6. Embed and Insert Data
    print("Embedding and inserting knowledge data...")
    embeddings = model.encode(knowledge_data)
    
    entities = [
        knowledge_data,  # text_chunk
        embeddings       # embedding
    ]
    
    try:
        collection.insert(entities)
        collection.flush()
        print(f"Successfully inserted {len(knowledge_data)} documents into Milvus.")
    except Exception as e:
        print(f"Error inserting data into Milvus: {e}")
        return

    # 7. Load collection for searching
    print("Loading collection into memory for searching...")
    collection.load()
    print("Collection loaded and ready for search.")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()