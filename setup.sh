#!/usr/bin/env bash
mkdir -p vncorenlp/models/wordsegmenter
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -P vncorenlp
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab -P vncorenlp/models/wordsegmenter/
wget -q --show-progress https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr -P vncorenlp/models/wordsegmenter/
