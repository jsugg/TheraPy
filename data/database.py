import weaviate

class WeaviateClient:
    def __init__(self, endpoint, api_key):
        self.client = weaviate.Client(endpoint, api_key)

    def query(self, query):
        return self.client.query.get(query)
