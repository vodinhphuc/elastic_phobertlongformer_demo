#! streamlit run
"""
This file run an web app demo for calculate cosin similary
using document's embedding vector
"""

from pathlib import Path

import streamlit as st
from decouple import config
from elasticsearch import Elasticsearch

from model import Model


@st.cache(allow_output_mutation=True)
def load_resource(
    model_path: str, segmenter_path: str, elastic_host: list, elastic_port: int
):
    """
    Load model and elasticsearch client
    """
    model = Model(model_path, segmenter_path)
    es = Elasticsearch(elastic_host, port=elastic_port)
    return model, es


def embed_text(doc: str):
    """
    create embedding for document
    """
    text_embeddings = model.encode(doc)
    return text_embeddings[0]


def search(document: str, type_ranker):
    """
    search in elasticsearch
    """
    if type_ranker == "PhoBert":
        doc_vector = embed_text(doc=document)
        source = "cosineSimilarity(params.query_vector, 'doc_vector') + 1.0"
        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": source,
                    "params": {"query_vector": doc_vector},
                },
            }
        }
    else:
        query = {"match": {"title": {"query": document, "fuzziness": "AUTO"}}}
    global es
    response = es.search(
        index=index_name,
        body={
            "size": 10,
            "query": query,
            "_source": {"includes": ["id", "title"]},
        },
        ignore=[400],
    )

    # early fail
    if not response.get("hits"):
        return []
    if not response["hits"].get("hits"):
        return []

    result = []
    for hit in response["hits"]["hits"]:
        result.append(hit["_source"]["title"])

    return result


def run():
    """
    run the main web app
    """
    st.title("Test semantic search")
    ranker = st.sidebar.radio("Rank by", ["BM25", "PhoBert"], index=0)
    st.markdown("Here is example")
    st.text("")
    search_doc = st.text_area("Write your test content!")
    if st.button("SEARCH"):
        with st.spinner("Searching ......"):
            if search_doc != "":
                if ranker == "PhoBert":
                    _result = search(search_doc, "PhoBert")
                else:
                    _result = search(search_doc, "BM25")

                if not _result:
                    st.warning("No document match")

                for i in _result:
                    st.success(f"{str(i)}")


if __name__ == "__main__":
    # get path
    root = Path.cwd().parent

    segmenter_path = str(root / "vncorenlp/VnCoreNLP-1.1.1.jar")
    model_path = config("MODEL_PATH")

    es_host = config("ELASTIC_HOST")
    es_port = int(config("ELASTIC_PORT"))
    index_name = config("INDEX_NAME")

    print("*" * 50)
    print("model_path =", model_path)
    print("segmenter_path =", segmenter_path)

    print("elastic_host =", es_host)
    print("elastic_port =", es_port)
    print("index_name =", index_name)

    model, es = load_resource(model_path, segmenter_path, [es_host], es_port)

    run()
