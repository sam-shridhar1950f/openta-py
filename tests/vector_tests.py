import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP
openai.api_key = "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"

school = "Stanford"
course_title = "CS105"
path = r"C:/Users/samsh/Videos/ta.ai/src/courses/"+school+"/"+course_title
stripped_fp = path + "/"
with open(stripped_fp+"myfilexs_jmQt5D0s.txt") as f:
    content = f.read()
    response = openai.Embedding.create( # related text (large)
    model= "text-embedding-ada-002",
    input=[content])
    response1_data = response["data"][0]["embedding"]
    print(len(response1_data))

with open(stripped_fp+"myfile0tAtq1vlU5w.txt") as f2:
    content2 = f2.read()
    response2 = openai.Embedding.create( # related text (large)
    model= "text-embedding-ada-002",
    input=[content2])
    response2_data = response2["data"][0]["embedding"]
    print(len(response2_data))

with open(stripped_fp+"myfileckhu-nU-s9c.txt") as f3:
    content3 = f3.read()
    response3 = openai.Embedding.create( # related text (large)
    model= "text-embedding-ada-002",
    input=[content3])
    response3_data = response3["data"][0]["embedding"]
    print(len(response3_data))