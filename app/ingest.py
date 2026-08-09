from __future__ import annotations
import os, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

DATA_DIR = os.getenv("DATA_DIR", "data")

def _load_docs(base) :
    docs = []

    # walk every file under data/, including subfolders (announcements/, faqs/, guides/, ...)
    all_files = Path(base).rglob("*")

    for path in all_files:
        is_hidden_file = path.name.startswith(".")
        if path.is_dir() or is_hidden_file:
            continue  # skip folders and stuff like .DS_Store

        ext = path.suffix.lower()
        path_str = str(path)

        try:
            if ext == ".md":
                for d in UnstructuredMarkdownLoader(path_str).load():
                    docs.append(d)
            elif ext == ".pdf":
                for d in PyMuPDFLoader(path_str).load():
                    docs.append(d)
            elif ext == ".docx":
                for d in UnstructuredWordDocumentLoader(path_str).load():
                    docs.append(d)
            elif ext == ".txt":
                for d in TextLoader(path_str).load():
                    docs.append(d)
            # anything else (.DS_Store, .json, etc.) just gets skipped

        except Exception:
            print(f"INGEST ERROR: failed to load {path_str}")
            traceback.print_exc()

    return docs
        

def _chunk(docs: List[Document]) -> List[Document]:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120
    )
    try:
        return splitter.split_documents(docs)
    except Exception:
        print(f"INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise



async def run_ingest_async() -> dict:
   docs= _load_docs(DATA_DIR)
   chunks=_chunk(docs)
   store= await get_vector_store()
   await store.aadd_documents(chunks)
   print(F'INGET:{len(docs)} docs, {len(chunks)} chunks')

   return {"documents": len(docs), "chunks":len(chunks)}
