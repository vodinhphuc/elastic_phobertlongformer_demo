"""
This file create elasticsearch index of document in /data
using phobertlong model
"""
from pathlib import Path

from decouple import config
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

from model import Model


def embed_text(batch_text) -> list:
    """
    create embedding for batch of document using model
    """
    batch_embeddings = model.encode(batch_text)
    return batch_embeddings


def index_batch(docs):
    """
    create index in elasticsearch for batch of document
    """
    contents = [doc["content"] for doc in docs]
    doc_vectors = embed_text(contents)

    for i, doc in enumerate(docs):
        body = {"title": doc["title"], "doc_vector": doc_vectors[i]}
        es.index(index=index_name, body=body)


def recreate_index():
    """
    Delete and create new index in elasticsearch
    """
    print(f"Deleting index: {index_name}")
    es.indices.delete(index=index_name, ignore=[404])

    print(f"Creating the {index_name} index.")
    config = {
        "mappings": {
            "dynamic": "true",
            "_source": {"enabled": "true"},
            "properties": {
                "title": {"type": "text"},
                "doc_vector": {"type": "dense_vector", "dims": 768},
            },
        }
    }
    es.indices.create(index=index_name, body=config, ignore=400)


def index():
    """
    index all document in data folder
    """
    docs = []
    count = 0
    data_path = root / "data"
    for file in tqdm(data_path.iterdir()):
        count += 1
        content = file.open().read()
        item = {"title": file.name, "content": content}
        docs.append(item)

        if count % batch_size == 0:
            index_batch(docs)
            docs = []

    if docs:
        index_batch(docs)

    print("Indexed {} documents.".format(count))
    es.indices.refresh(index=index_name)
    print("Done indexing.")


if __name__ == "__main__":
    # get path
    root = Path.cwd().parent

    model_path = config("MODEL_PATH")
    segmenter_path = str(root / "vncorenlp/VnCoreNLP-1.1.1.jar")

    elastic_host = config("ELASTIC_HOST")
    elastic_port = int(config("ELASTIC_PORT"))
    index_name = config("INDEX_NAME")
    re_index = config("RE_INDEX")
    batch_size = config("BATCH_SIZE")

    print("*" * 50)
    print("model_path =", model_path)
    print("segmenter_path =", segmenter_path)

    print("elastic_host =", elastic_host)
    print("elastic_port =", elastic_port)
    print("index_name =", index_name)
    print("reindex =", re_index)
    print("batch_size =", batch_size)
    print("*" * 50)

    model = Model(model_path, segmenter_path)
    es = Elasticsearch(elastic_host, port=elastic_port)

    if re_index == "true":
        recreate_index()
    index()
