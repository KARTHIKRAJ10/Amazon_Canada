# src/app.py
from fastapi import FastAPI
from src.recommend import recommend

app = FastAPI()

@app.get("/recommend/{index}")
def get_recommendations(index: int, top_n: int = 5):
    recs, sims = recommend(index, top_n)
    return {
        "input_product_index": index,
        "recommendations": [
            {
                "title": recs.iloc[i]["title"],
                "category": recs.iloc[i]["categoryName"],
                "similarity": float(sims[i])
            }
            for i in range(top_n)
        ]
    }
