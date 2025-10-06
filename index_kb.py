import json
import os
from azure.ai.inference import EmbeddingsClient
import pinecone

# Load KB
with open('self_critique_loop_dataset.json', 'r') as f:
    kb_entries = json.load(f)

# Azure Embeddings setup
embeddings_client = EmbeddingsClient(
    endpoint=os.environ['AZURE_EMBEDDINGS_ENDPOINT'],
    api_key=os.environ['AZURE_EMBEDDINGS_KEY'],
    model="text-embedding-3-small"
)

# Pinecone setup
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment='gcp-starter')
index = pinecone.Index('kb-index')

# Indexing
for entry in kb_entries:
    text = entry['content']
    embedding = embeddings_client.embed(text)
    index.upsert([(entry['id'], embedding, {'text': text})])

print("Indexing complete.")
