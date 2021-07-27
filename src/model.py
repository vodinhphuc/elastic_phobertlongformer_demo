import torch
from transformers import AutoTokenizer
from transformers import LongformerModel as lfm

from vncorenlp import VnCoreNLP


class Model(object):
    """
    This model use to embed document to vector
    """

    def __init__(self, model_path, segmenter_path):
        super(Model, self).__init__()

        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = torch.device(device)

        self.phoBertLong = lfm.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.model_max_length = (
            self.phoBertLong.config.max_position_embeddings
        )

        self.segmenter = VnCoreNLP(
            segmenter_path, annotators="wseg", max_heap_size="-Xmx500m"
        )

    def encode(self, docs) -> list:
        """
        Encode documents to embedding
        """
        if isinstance(docs, str):
            docs = [docs]

        embeddings = []
        for doc in docs:
            doc_segmented = ""
            sentences_segments_tokens = self.segmenter.tokenize(doc)
            for sentence_tokens in sentences_segments_tokens:
                sentence_segmented = " ".join(sentence_tokens)
                doc_segmented += sentence_segmented + " "
            doc_segmented = doc_segmented.strip()

            encoding = self.tokenizer.encode_plus(
                doc_segmented,
                max_length=self.phoBertLong.config.max_position_embeddings,
                truncation=True,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",  # Return PyTorch tensors
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # calculate embedding
            with torch.no_grad():
                output = self.phoBertLong(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )

                arr = output[1].detach().cpu().numpy().flatten().tolist()
                embeddings.append(arr)

        return embeddings
