import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import List

products = pd.read_csv("adidas_usa.csv")

db = lancedb.connect("db")
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class Product(LanceModel):
    name: str 
    description: str
    text: str = model.SourceField()
    url: str 
    color: str
    category: str 
    image_urls: List[str] 
    selling_price: int 
    vector: Vector(model.ndims()) = model.VectorField()
    
    def __str__(self):
        return f"Name:{self.name} \n Description: {self.description} \n Color:{self.color} \n Category:{self.category}"


table = db.create_table("adidas", schema=Product, mode='overwrite')
table.add([{**p, "image_urls":p["images"].split("~"), "text": f'Name: {p["name"]} Description: {p["description"]} Color:{p["color"]} Category: {p["category"]}'} for p in products.to_dict(orient="records")])

query = "beach shorts"
actual = table.search(query).limit(5).to_pydantic(Product)
print(actual)