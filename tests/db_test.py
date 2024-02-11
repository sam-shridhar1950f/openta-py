import pinecone
import numpy as np
pinecone.init(api_key="190dabf8-0c0b-4690-8733-0be7fcecdb34", environment="gcp-starter")

def get_ids_from_query(index,input_vector):
  print("searching pinecone...")
  results = index.query(vector=input_vector, top_k=10000,include_values=False)
  ids = set()
  print(type(results))
  for result in results['matches']:
    #print(result)
    ids.add(result['id'])
  return ids

def get_all_ids_from_index(index, num_dimensions, namespace=""):
  num_vectors = index.describe_index_stats()["namespaces"][namespace]['vector_count']
  all_ids = set()
  while len(all_ids) < num_vectors:
    print("Length of ids list is shorter than the number of total vectors...")
    input_vector = np.random.rand(num_dimensions).tolist()
    print("creating random vector...")
    ids = get_ids_from_query(index,input_vector)
    print("getting ids from a vector query...")
    all_ids.update(ids)
    print("updating ids set...")
    print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
    print(all_ids) # set of all ids - can use as iterable later on




print(pinecone.list_indexes())
# pinecone.create_index("stanford-cs105", dimension=1536, metric="euclidean") # name of index needs to be lowercase
#print(pinecone.describe_index("stanford-cs229"))
#pinecone.delete_index("cornell-cs2110")
#index = pinecone.Index("stanford-cs105-a")
#print(index.describe_index_stats())


#all_ids = get_all_ids_from_index(index, num_dimensions=1536, namespace="")
#print(all_ids) # tests number of vectors in index
# helpers

