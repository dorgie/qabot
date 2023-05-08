#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import pickle

import concurrent.futures
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings


def create_index(input_file: str, output_file: str) -> None:
    if input_file.endswith('.txt'):
        text = open(input_file, encoding='utf8').read()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(text)
    elif input_file.endswith('.json'):
        texts = json.load(open(input_file, encoding='utf8'))
    else:
        raise ValueError(f'unknown file type: {input_file}')
    print(f'split into {len(texts)} chunks')
    embeddings = OpenAIEmbeddings(client=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        index = np.array(list(executor.map(embeddings.embed_query, texts)))
    with open(output_file, 'wb') as f:
        pickle.dump({'index': index, 'texts': texts}, f)

def load_index(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

class MyKNNRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    index: Any
    texts: List[str]
    k: int = 4
    relevancy_threshold: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_file(cls, path: str, **kwargs: Any) -> MyKNNRetriever:
        data = load_index(path)
        index = data['index']
        texts = data['texts']
        embeddings = OpenAIEmbeddings(client=None)
        return cls(embeddings=embeddings, index=index, texts=texts, **kwargs)

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = []
        for row in sorted_ix[0 : self.k]:
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            ):
                top_k_results.append(Document(page_content=self.texts[row]))
        return top_k_results

    async def aget_relevant_documents(self, _: str) -> List[Document]:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file', nargs='?', default='index.pkl')
    args = parser.parse_args()
    create_index(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
