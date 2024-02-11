

import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP
import pinecone
# Note To Self: 8000 tokens is about 6500 words, so use as modulo for calculating chunks
openai.api_key = "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" # openai api key

with open(r"C:\Users\samsh\Videos\ta.ai\src\courses\Stanford\CS105\Lecture1.3") as f:
                content = f.read()
                response = openai.Embedding.create( # related text (large)
                    model= "text-embedding-ada-002",
                    input=[content])

                query = "What is a binary number?"
                query2 = "do binary numbers have endpoints?"

                response3 = openai.Embedding.create( # related text (chunk)
    	        model= "text-embedding-ada-002",
    	        input=[query])

                response4 = openai.Embedding.create( # related text (chunk)
    	        model= "text-embedding-ada-002",
    	        input=[query2])


                response_data = response["data"][0]["embedding"]
                response3_data = response3["data"][0]["embedding"]
                response4_data = response4["data"][0]["embedding"]


                result = 1 - distance.cosine(response_data, response3_data)
                result2 = 1 - distance.cosine(response_data, response4_data)
                print(result)
                print(result2)