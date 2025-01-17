#!/usr/bin/env python3

import argparse
from typing import List
import os
import json
import hashlib
import re
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader
from vespa.application import Vespa
from vespa.io import VespaResponse
from dotenv import load_dotenv
from openai import OpenAI
import base64
from torch.utils.data import DataLoader
from PIL import Image
import io
import torch
import contextlib
import sys

load_dotenv()

# Configure OpenAI client
client = OpenAI(
    base_url="https://gaia-api.dlr-gfr.com",
    api_key="sk-1234"
)

def sanitize_text(text):
    # Remove control characters and non-printable characters
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

def get_image_embedding(image: Image.Image) -> np.ndarray:
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response = client.embeddings.create(
        model="colqwen2",
        input=["data:image/jpeg;base64," + base64_image],
        encoding_format="float"
    )

    # Extract and process embedding
    embedding = response.data[0].embedding
    embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array

def group_images_by_resolution(images: List[Image.Image]) -> dict:
    """Group images by their resolution."""
    resolution_groups = {}
    for img in images:
        resolution = f"{img.size[0]}x{img.size[1]}"
        if resolution not in resolution_groups:
            resolution_groups[resolution] = []
        resolution_groups[resolution].append(img)
    return resolution_groups

def process_images_batch(images: List[Image.Image], batch_size: int = 16) -> List[np.ndarray]:
    """Process images in batches of specified size."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_embeddings = []
        for image in batch:
            embedding = get_image_embedding(image)
            batch_embeddings.append(embedding)
        embeddings.extend(batch_embeddings)
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Feed data into local Vespa application")
    parser.add_argument(
        "--application_name",
        required=True,
        default="colpalidemo",
        help="Vespa application name",
    )
    parser.add_argument(
        "--vespa_schema_name",
        required=True,
        default="pdf_page",
        help="Vespa schema name",
    )
    parser.add_argument(
        "--pdf_folder",
        required=True,
        help="Path to folder containing PDF files",
    )
    args = parser.parse_args()

    vespa_app_url = "http://localhost:8080"  # Assuming local Vespa deployment
    application_name = args.application_name
    schema_name = args.vespa_schema_name
    pdf_folder = args.pdf_folder

    # Instantiate Vespa connection
    app = Vespa(url=vespa_app_url)
    app.get_application_status()

    def get_pdf_images(pdf_path):
        reader = PdfReader(pdf_path)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text()
            sanitized_text = sanitize_text(text)
            page_texts.append(sanitized_text)

        # Suppress warnings from pdf2image
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                images = convert_from_path(pdf_path)

        assert len(images) == len(page_texts)
        return (images, page_texts)

    # Get all PDF files from the specified folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

    vespa_feed = []  # Initialize vespa_feed here

    # Check if vespa_feed.json exists
    if os.path.exists("vespa_feed.json"):
        print("Loading vespa_feed from vespa_feed.json")
        with open("vespa_feed.json", "r") as f:
            vespa_feed_saved = json.load(f)
        for doc in vespa_feed_saved:
            put_id = doc["put"]
            fields = doc["fields"]
            parts = put_id.split("::")
            document_id = parts[1] if len(parts) > 1 else ""
            page = {"id": document_id, "fields": fields}
            vespa_feed.append(page)
    else:
        print("Generating vespa_feed")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            title = os.path.basename(pdf_file)
            url = pdf_path

            page_images, page_texts = get_pdf_images(pdf_path)

            # Group images by resolution before processing
            resolution_groups = group_images_by_resolution(page_images)

            # Process each resolution group separately
            all_embeddings = []
            for resolution, images in resolution_groups.items():
                print(f"Processing {len(images)} images of resolution {resolution}")

                # Convert PIL images to numpy arrays and resize
                #images_np = [resize_image(np.array(image)) for image in images]
                images_np = [np.array(image) for image in images]
                # Create DataLoader for this resolution group
                dataloader = DataLoader(
                    images_np,
                    batch_size=16,  # Set batch size to 16
                    shuffle=False,
                )

                group_embeddings = []
                for batch_images in tqdm(dataloader, desc=f"Processing {resolution}"):
                    # Convert tensors back to numpy arrays
                    batch_images_np = [image.numpy() for image in batch_images]
                    batch_embeddings = process_images_batch(
                        [Image.fromarray(image) for image in batch_images_np],
                        batch_size=16
                    )
                    group_embeddings.extend(batch_embeddings)

                # Store embeddings in the same order as original images
                for idx, embedding in enumerate(group_embeddings):
                    all_embeddings.append((images[idx], embedding))

            # Sort embeddings back to original page order
            all_embeddings.sort(key=lambda x: page_images.index(x[0]))
            page_embeddings = [emb for _, emb in all_embeddings]

            for page_number, (page_text, embedding, image) in enumerate(
                zip(page_texts, page_embeddings, page_images)
            ):
                # Convert image to base64
                buffered = io.BytesIO()
                scaled_image = image.copy()
                scaled_image.thumbnail((640, 640))
                scaled_image.save(buffered, format="JPEG")
                base_64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Full image
                buffered_full = io.BytesIO()
                image.save(buffered_full, format="JPEG")
                base_64_full_image = base64.b64encode(buffered_full.getvalue()).decode('utf-8')

                # Convert embedding to binary format for Vespa
                embedding_dict = dict()
                for idx, patch_embedding in enumerate(embedding):
                    binary_vector = (
                        np.packbits(np.where(patch_embedding > 0, 1, 0))
                        .astype(np.int8)
                        .tobytes()
                        .hex()
                    )
                    embedding_dict[idx] = binary_vector

                id_hash = hashlib.md5(f"{url}_{page_number}".encode()).hexdigest()
                page = {
                    "id": id_hash,
                    "fields": {
                        "id": id_hash,
                        "url": url,
                        "title": title,
                        "page_number": page_number + 1,
                        "image": base_64_image,
                        "full_image": base_64_full_image,
                        "text": page_text,
                        "embedding": embedding_dict,
                    },
                }
                vespa_feed.append(page)

        # Save vespa_feed to vespa_feed.json
        vespa_feed_to_save = []
        for page in vespa_feed:
            document_id = page["id"]
            put_id = f"id:{application_name}:{schema_name}::{document_id}"
            vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
        with open("vespa_feed.json", "w") as f:
            json.dump(vespa_feed_to_save, f)

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(
                f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
            )
        else:
            print(f"feeding completed successfully")

    # Feed data into Vespa
    app.feed_iterable(vespa_feed, schema=schema_name, callback=callback)

if __name__ == "__main__":
    main()
(base) gfrai@gfr-ai-database01:/opt/llm/sample-code/visual-retrieval/colqwen2-vespa-feed$ ll
-bash: ll: command not found
(base) gfrai@gfr-ai-database01:/opt/llm/sample-code/visual-retrieval/colqwen2-vespa-feed$ ls
application                feed-vespa_colqwen2-api4.py  feed-vespa_colqwen2-api7.py  feed-vespa_colqwen2-api.py         README.md         vespa_feed.json.1
colqwen2_api.py            feed-vespa_colqwen2-api5.py  feed-vespa_colqwen2-api8.py  query_vespa-process-visual-WiP.py  requirements.txt  vespa_feed.json.2
deploy_vespa_app_local.py  feed-vespa_colqwen2-api6.py  feed-vespa_colqwen2-api9.py  query_vespa.py                     vespa_feed.json
(base) gfrai@gfr-ai-database01:/opt/llm/sample-code/visual-retrieval/colqwen2-vespa-feed$ cat requirements.txt
pyvespa
pillow
pdf2image
pypdf
openai
numpy
python-dotenv
torch[cpu](base) gfrai@gfr-ai-database01:/opt/llm/sample-code/visual-retrieval/colqwen2-vespa-feed$ ls
application                feed-vespa_colqwen2-api4.py  feed-vespa_colqwen2-api7.py  feed-vespa_colqwen2-api.py         README.md         vespa_feed.json.1
colqwen2_api.py            feed-vespa_colqwen2-api5.py  feed-vespa_colqwen2-api8.py  query_vespa-process-visual-WiP.py  requirements.txt  vespa_feed.json.2
deploy_vespa_app_local.py  feed-vespa_colqwen2-api6.py  feed-vespa_colqwen2-api9.py  query_vespa.py                     vespa_feed.json
(base) gfrai@gfr-ai-database01:/opt/llm/sample-code/visual-retrieval/colqwen2-vespa-feed$ cat deploy_vespa_app_local.py
import argparse
import os
import subprocess
from pathlib import Path
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    FieldSet,
    SecondPhaseRanking,
)

def create_vespa_application_package(vespa_app_name):
    # Define the Vespa schema
    colpali_schema = Schema(
        name="pdf_page",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"], match=["word"]),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(name="title", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(name="full_image", type="raw", indexing=["summary"]),
                Field(name="text", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(
                    name="embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=["attribute", "index"],
                    ann=HNSW(
                        distance_metric="hamming",
                        max_links_per_node=32,
                        neighbors_to_explore_at_insert=400,
                    ),
                ),
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["title", "url", "page_number", "text"]),
            FieldSet(name="image", fields=["image"]),
        ],
    )

    # Define rank profiles
    colpali_profile = RankProfile(
        name="default",
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
        ],
        first_phase="bm25_score",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_profile)

    # Add retrieval-and-rerank rank profile
    input_query_tensors = []
    MAX_QUERY_TERMS = 64
    for i in range(MAX_QUERY_TERMS):
        input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

    input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
    input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

    colpali_retrieval_profile = RankProfile(
        name="retrieval-and-rerank",
        inputs=input_query_tensors,
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(
                name="max_sim_binary",
                expression="""
                    sum(
                      reduce(
                        1/(1 + sum(
                            hamming(query(qtb), attribute(embedding)) ,v)
                        ),
                        max,
                        patch
                      ),
                      querytoken
                    )
                """,
            ),
        ],
        first_phase="max_sim_binary",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_retrieval_profile)

    # Create the Vespa application package
    vespa_application_package = ApplicationPackage(
        name=vespa_app_name,
        schema=[colpali_schema],
    )

    return vespa_application_package

def deploy_vespa_app(app_package_path):
    # Deploy the application using Vespa CLI
    deploy_command = f"vespa deploy {app_package_path} --wait 300"

    try:
        subprocess.run(deploy_command, shell=True, check=True)
        print("Vespa application deployed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error deploying Vespa application: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy Vespa application locally")
    parser.add_argument("--vespa_application_name", required=True, help="Vespa application name")
    parser.add_argument("--output_dir", default="application", help="Output directory for the application package")

    args = parser.parse_args()
    vespa_app_name = args.vespa_application_name
    output_dir = args.output_dir

    # Create the Vespa application package
    app_package = create_vespa_application_package(vespa_app_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the application package to disk
    app_package.to_files(output_dir)

    # Deploy the application locally
    deploy_vespa_app(os.path.abspath(output_dir))

    print(f"Application deployed locally. You can access it at http://localhost:8080/")

if __name__ == "__main__":
    main()