#!/usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from typing import cast
import asyncio
import base64
import json
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

MAX_QUERY_TERMS = 64
SAVEDIR = Path(__file__).parent / "output" / "images"
load_dotenv()

# Configure OpenAI client for colqwen2 API
client = OpenAI(
    base_url="http://localhost:7997",  # Update with your colqwen2 API URL
    api_key="sk-1234"  # Update with your API key if required
)

def get_text_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="colqwen2",
        input=[text],
        extra_body={"modality": "text"},
        encoding_format="float"
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def display_query_results(query, response, hits=5):
    query_time = response.json.get("timing", {}).get("searchtime", -1)
    query_time = round(query_time, 2)
    count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    result_text = f"Query text: '{query}', query time {query_time}s, count={count}, top results:\n"

    for i, hit in enumerate(response.hits[:hits]):
        title = hit["fields"]["title"]
        url = hit["fields"]["url"]
        page = hit["fields"]["page_number"]
        image = hit["fields"]["image"]
        _id = hit["id"]
        score = hit["relevance"]

        result_text += f"\nPDF Result {i + 1}\n"
        result_text += f"Title: {title}, page {page+1} with score {score:.2f}\n"
        result_text += f"URL: {url}\n"
        result_text += f"ID: {_id}\n"
        # Optionally, save or display the image
        # img_data = base64.b64decode(image)
        # img_path = SAVEDIR / f"{title}.png"
        # with open(f"{img_path}", "wb") as f:
        #     f.write(img_data)
    print(result_text)

async def query_vespa_default(app, queries, qs):
    async with app.asyncio(connections=1, total_timeout=120) as session:
        for idx, query in enumerate(queries):
            query_embedding = {k: v.tolist() for k, v in enumerate(qs[idx])}
            response: VespaQueryResponse = await session.query(
                yql="select documentid,title,url,image,page_number from pdf_page where userInput(@userQuery)",
                ranking="default",
                userQuery=query,
                timeout=120,
                hits=3,
                body={"input.query(qt)": query_embedding, "presentation.timing": True},
            )
            assert response.is_successful()
            display_query_results(query, response)

async def query_vespa_nearest_neighbor(app, queries, qs):
    # Using nearestNeighbor for retrieval
    target_hits_per_query_tensor = 20  # this is a hyper parameter that can be tuned for speed versus accuracy
    async with app.asyncio(connections=1, total_timeout=180) as session:
        for idx, query in enumerate(queries):
            float_query_embedding = {k: v.tolist() for k, v in enumerate(qs[idx])}
            binary_query_embeddings = dict()
            for k, v in float_query_embedding.items():
                binary_vector = (
                    np.packbits(np.where(np.array(v) > 0, 1, 0))
                    .astype(np.int8)
                    .tolist()
                )
                binary_query_embeddings[k] = binary_vector
                if len(binary_query_embeddings) >= MAX_QUERY_TERMS:
                    print(
                        f"Warning: Query has more than {MAX_QUERY_TERMS} terms. Truncating."
                    )
                    break

            # The mixed tensors used in MaxSim calculations
            # We use both binary and float representations
            query_tensors = {
                "input.query(qtb)": binary_query_embeddings,
                "input.query(qt)": float_query_embedding,
            }
            # The query tensors used in the nearest neighbor calculations
            for i in range(0, len(binary_query_embeddings)):
                query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]
            nn = []
            for i in range(0, len(binary_query_embeddings)):
                nn.append(
                    f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
                )
            # We use an OR operator to combine the nearest neighbor operator
            nn = " OR ".join(nn)
            response: VespaQueryResponse = await session.query(
                body={
                    **query_tensors,
                    "presentation.timing": True,
                    "yql": f"select documentid, title, url, image, page_number from pdf_page where {nn}",
                    "ranking.profile": "retrieval-and-rerank",
                    "timeout": 120,
                    "hits": 3,
                },
            )
            assert response.is_successful(), response.json
            display_query_results(query, response)

def main():
    vespa_app_url = "http://localhost:8080"  # Update with your local Vespa URL
    if not vespa_app_url:
        raise ValueError("Please set the VESPA_APP_URL environment variable")

    # Instantiate Vespa connection
    app = Vespa(url=vespa_app_url)
    status_resp = app.get_application_status()
    if status_resp.status_code != 200:
        print(f"Failed to connect to Vespa at {vespa_app_url}")
        return
    else:
        print(f"Connected to Vespa at {vespa_app_url}")

    # Define queries
    queries = [
        "My test query 1",
        "My test query 2",
        "My test query 3",
    ]

    # Obtain query embeddings using colqwen2 API
    qs = []
    for query in queries:
        embedding = get_text_embedding(query)
        qs.append(embedding)

    # Perform queries using default rank profile
    print("Performing queries using default rank profile:")
    asyncio.run(query_vespa_default(app, queries, qs))

    # Perform queries using nearestNeighbor
    print("Performing queries using nearestNeighbor:")
    asyncio.run(query_vespa_nearest_neighbor(app, queries, qs))

if __name__ == "__main__":
    main()
