"""
title: Visual Retrieval
author: Florian Euler
description: Visual Retrieval - colqwen + vespa
version: 0.0.1
"""

import logging
import asyncio
import base64
import json
import numpy as np
from typing import Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from openai import OpenAI
from open_webui.constants import TASKS

# Constants and Setup
name = "Visual Retrieval"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set to DEBUG level
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


@dataclass
class User:
    id: str
    email: str
    name: str
    role: str


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: User
    __model__: str

    class Valves(BaseModel):
        VESPA_APP_URL: str = Field(
            default="http://vespa-example:8080",
            description="URL of the Vespa application.",
        )
        OPENAI_URL: str = Field(
            default="https://server.example.com",
            description="URL of the openai comptible api",
        )
        OPENAI_API_KEY: str = Field(
            default="sk-1234",
            description="api key for the openai compatible api",
        )
        VISION_MODEL: str = Field(
            default="Qwen2-VL-7B",
            description="Vision language model to use for processing images and text.",
        )
        MAX_HITS: int = Field(
            default=6,
            description="Maximum number of document hits to retrieve from Vespa.",
        )
        IMAGE_PREVIEW_SIZE: int = Field(
            default=256,
            description="Size of the image previews to display.",
        )
        EMBEDDING_MODEL: str = Field(
            default="colqwen2-text", description="Model to use for embeddings"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.vespa_app = Vespa(url=self.valves.VESPA_APP_URL)
        self.type = "manifold"
        self.client = OpenAI(
            base_url=f"{self.valves.OPENAI_URL}",
            api_key=f"{self.valves.OPENAI_API_KEY}",
        )

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def emit_message(self, message: str):
        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": message}}
        )

    async def emit_status(self, level: str, message: str, done: bool):
        await self.__current_event_emitter__(
            {
                "type": "status",
                "data": {
                    "status": "complete" if done else "in_progress",
                    "level": level,
                    "description": message,
                    "done": done,
                },
            }
        )

    async def query_vespa(self, query: str) -> List[Dict]:
        logger.debug(f"Starting Vespa query for: {query[:50]}...")
        try:
            # Get text embedding using the same method as query_vespa.py
            response = self.client.embeddings.create(
                model=self.valves.EMBEDDING_MODEL,
                input=[query],
                encoding_format="float",
            )
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            logger.debug(f"Query embedding obtained, shape: {query_embedding.shape}")

            # Convert embedding to the format expected by Vespa
            float_query_embedding = {
                k: v.tolist() for k, v in enumerate(query_embedding)
            }
            logger.debug(
                f"Created float query embedding dict with keys: {float_query_embedding.keys()}"
            )

            async with self.vespa_app.asyncio(
                connections=1, total_timeout=120
            ) as session:
                logger.debug("Established Vespa session")
                try:
                    # Using the same query structure as query_vespa_default
                    response: VespaQueryResponse = await session.query(
                        yql="select documentid,title,url,full_image,page_number from pdf_page where userInput(@userQuery)",
                        ranking="default",
                        userQuery=query,
                        timeout=120,
                        hits=self.valves.MAX_HITS,
                        body={
                            "input.query(qt)": float_query_embedding,
                            "presentation.timing": True,
                        },
                    )
                    logger.debug(
                        f"Vespa query completed. Response status: {response.status_code}"
                    )

                    if not response.is_successful():
                        logger.error(
                            f"Failed Vespa query. Status: {response.status_code}, Response: {response.json}"
                        )
                        return []

                    logger.debug(f"Number of hits: {len(response.hits)}")
                    return response.hits

                except Exception as e:
                    logger.error(
                        f"Error during Vespa query execution: {str(e)}", exc_info=True
                    )
                    raise

        except Exception as e:
            logger.error(f"Error in query_vespa: {str(e)}", exc_info=True)
            raise

    async def generate_vision_response(self, query: str, images: List[str]) -> str:
        logger.debug(f"Generating vision response for query: {query[:50]}...")
        logger.debug(f"Number of images: {len(images)}")

        try:
            # Prepare content list starting with the query
            content = [{"type": "text", "text": query}]

            # Add all images in the correct format
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your task is to write a comprehensive and detailed response based on the user query using only the information from the attached satellite documentation images. If there is not enough information to respond to the user query you say i don't have enough context.",
                },
                {"role": "user", "content": content},
            ]

            logger.debug(f"Created messages structure with {len(images)} images")
            logger.debug("Message content structure:")
            logger.debug(f"Number of content items: {len(content)}")

            response = await generate_chat_completion(
                self.__request__,
                {
                    "model": self.valves.VISION_MODEL,
                    "messages": messages,
                    "max_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "repetition_penalty": 1.05,
                },
                user=self.__user__,
            )
            logger.debug(f"Vision model response type: {type(response)}")
            logger.debug(
                f"Vision model response structure: {json.dumps(response, default=str)[:200]}..."
            )

            if (
                isinstance(response, dict)
                and "choices" in response
                and response["choices"]
            ):
                result = response["choices"][0]["message"]["content"]
                logger.debug(
                    f"Successfully generated vision response: {result[:100]}..."
                )
                return result
            else:
                logger.error(f"Unexpected response format: {response}")
                return "Error generating vision response."

        except Exception as e:
            logger.error(f"Error in generate_vision_response: {str(e)}", exc_info=True)
            return f"Error generating vision response: {str(e)}"

    async def emit_document_preview(self, documents: List[Dict]):
        """Emit all document previews in a single message with side-by-side images"""
        # Create a table-like structure using markdown
        # Each image will be a cell in the first row, followed by metadata cells below

        # First collect all images in one row
        images_row = "|"
        headers_row = "|"
        metadata_row = "|"

        for doc in documents:
            # Add image cell
            images_row += f" ![Document Preview]({doc['image_preview']}) |"
            # Add header cell for alignment
            headers_row += " :---: |"
            # Add metadata cell
            metadata_row += (
                f" **Title:** {doc['title']} **Page:** {doc['page_number']} |"
            )

        # Combine all rows into a markdown table
        markdown_table = f"{images_row}\n{headers_row}\n{metadata_row}"

        await self.__current_event_emitter__(
            {"type": "message", "data": {"content": markdown_table}}
        )

    async def emit_formatted_response(self, response: str):
        """Emit the vision response with proper line break formatting"""
        formatted_html = f"""
            Response: {response}
        """

        await self.__current_event_emitter__(
            {
                "type": "message",
                "data": {"content": formatted_html, "content_type": "text/html"},
            }
        )

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
    ) -> str:
        logger.debug("Starting pipe execution")
        logger.debug(f"User: {__user__}")
        logger.debug(f"Task: {__task__}")
        logger.debug(f"Model: {__model__}")

        if __event_emitter__ is None:
            logger.error("Event emitter is None")
            return ""

        try:
            self.__user__ = User(**__user__)
            self.__request__ = __request__
            self.__current_event_emitter__ = __event_emitter__

            query = body.get("messages", [])[-1].get("content", "").strip()
            logger.debug(f"Extracted query: {query[:50]}...")

            if not query:
                logger.warning("Empty query received")
                await self.emit_status("error", "Please provide a query.", True)
                return ""

            await self.emit_status("info", "Querying documents...", False)
            hits = await self.query_vespa(query)
            logger.debug(f"Received {len(hits)} hits from Vespa")

            if not hits:
                logger.warning("No hits found in Vespa response")
                await self.emit_status("info", "No relevant documents found.", True)
                return ""

            images = []
            document_info = []
            for idx, hit in enumerate(hits):
                logger.debug(f"Processing hit {idx + 1}")
                if "fields" in hit and "full_image" in hit["fields"]:
                    image_data = hit["fields"]["full_image"]
                    images.append(image_data)
                    doc_info = {
                        "title": hit["fields"].get("title", "No title"),
                        "page_number": hit["fields"].get("page_number", "Unknown"),
                        "image_preview": f"data:image/jpeg;base64,{image_data}",
                        "url": hit["fields"].get("url", "No URL"),
                    }
                    document_info.append(doc_info)
                    logger.debug(f"Added document info: {doc_info['title']}")

            # Emit document previews immediately
            await self.emit_document_preview(document_info)

            logger.debug(f"Processed {len(images)} images")
            await self.emit_status("info", "Generating vision response...", False)

            vision_response = await self.generate_vision_response(query, images)
            logger.debug("Vision response generated successfully")

            # Emit formatted vision response
            await self.emit_formatted_response(vision_response)

            await self.emit_status("success", "Processing complete", True)
            return ""

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.emit_status("error", error_msg, True)
            return ""
