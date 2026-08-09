import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

load_dotenv()

PG_CONN_STR = os.getenv("DATABASE_URL")
PG_ENGINE = PGEngine.from_connection_string(PG_CONN_STR)

embeddings= OpenAIEmbeddings(model="text-embedding-3-small")

async def get_vector_store():
    return await AsyncPGVectorStore.create(
        engine=PG_ENGINE,
        embedding_service=embeddings,
        table_name='langchain_pg_embedding'
        )
