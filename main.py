# Import libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

class QueryPhrase(BaseModel):
    query: str

class MCCMatch(BaseModel):
    mcc_code: str
    name: str
    mcc_match: float
    mcc_description: str

# Get model from sentence_transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_csv("mcc_descriptions.csv", dtype=str)
# Load mcc embedding
mcc_embeddings = torch.load('mcc_embeddings_mini_bert.pt')

@app.get("/")
def read_root():
    return {"Curious about what MCC Code to use?": "Search now at the /predict endpoint!"}

@app.post("/predict")
async def predict_mcc(payload: QueryPhrase):
    # Embed query in BERT space
    embeddings_query = model.encode([payload.query], convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(mcc_embeddings, embeddings_query)
    # Convert tensor to numpy
    match_df = df.copy()
    match_df['matches_to_query'] = cosine_scores.numpy()    
    # Rename columns
    match_df = match_df.rename(columns={'mcc': 'mcc_code', 'short_name': 'name', 'unaltered_description': 'mcc_description', 'matches_to_query': 'mcc_match'})
    # Get top 5
    output_df = match_df[['mcc_code', 'name', 'mcc_match', 'mcc_description',]] \
        .sort_values('mcc_match', ascending=False) \
        .head(5) \
        .astype({'mcc_match': 'float'}) \
        .round({'mcc_match': 5})
  
    return output_df.to_dict('records')