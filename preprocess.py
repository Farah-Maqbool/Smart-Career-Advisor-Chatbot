#load json file
import json #we use json to load the json file
import faiss #faiss is vector database store vectors or embeddings
import numpy as np #numpy is used due to convert the embedding into float32 which is required for faiss
from sentence_transformers import SentenceTransformer #sentence transformer is a library or toolkit it use pretrained transformer model to convert
#sentence into vectors

with open("job_title_des.json", "r") as f:
    job_data = json.load(f) #the file in json convert it in to dictionary object

#Now convert it in chunks or documents 
documents = []
for i, job in enumerate(job_data):
    text = f"Job Title: {job['Job Title']}\nDescription: {job['Job Description']}"
    documents.append({"id":f"doc_{i}", "text": text})
#each job is a chunk which contain job title and its description

# 1. Load the pre-trained embedding model (all-MiniLM-L6-V2 popular model used for embedding) 
model = SentenceTransformer("all-MiniLM-L6-V2")

#take text from chunks
texts = [document["text"] for document in documents] #chuns have two things id and text so we are taking text

#do embedding
embeddings = model.encode(texts)

#convert in float32 which is required for faiss
embeddings = np.array(embeddings).astype("float32")

# now we store the embeddings in faiss 
#FAISS index is a structure that helps search among vectors efficiently.
#FAISS index helps you find similar vectors fast without scanning them all. Its a smart search engine
index = faiss.IndexFlatL2(embeddings.shape[1])


index.add(embeddings) #Add vectors to the index

faiss.write_index(index, "index.faiss") #storing faiss index 
#we are not storing embeddings seprately because it is already stored in index

with open("texts.json", "w") as f: #store the text
    json.dump(texts, f)

