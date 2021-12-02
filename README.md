# awesome-semantic-search

In [Semantic search with embeddings](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c), I described how to build semantic search systems (also called neural search). These systems are being used more and more with indexing techniques improving and representation learning getting better every year with new deep learning papers. The medium post explain how to build them, and this list is meant to reference all interesting resources on the topic to allow anyone to quickly start building systems.

![image](https://user-images.githubusercontent.com/2346494/118412784-38db9480-b69c-11eb-9cf7-d159da16434a.png)


*   **Tutorials** explain in depth how to build semantic search systems
    *   [Semantic search with embeddings](https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c#ef3f) end to end explanation on how to build semantic search pipelines
    *   [google cloud embedding similarity system](https://cloud.google.com/solutions/machine-learning/building-real-time-embeddings-similarity-matching-system)  Use google cloud to build an embedding similarity system
    *   [cvpr 2020 tutorial on image retrieval](https://matsui528.github.io/cvpr2020_tutorial_retrieval/) end to end in depth tutorial focusing on image
*   **Good datasets** to build semantic search systems
    *   [Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/overview) building search systems only requires image or text, many tf datasets are interesting in that regard
    *   [Torchvision datasets](https://pytorch.org/vision/stable/datasets.html) datasets provided for vision are also interesting for this
*   **Pretrained encoders** make it possible to quickly build a new system without training
    *   Vision+Language
        *   [Clip](https://github.com/openai/CLIP) encode image and text in a same space
    *   Image
        *   [Efficientnet b0](https://github.com/qubvel/efficientnet) is a simple way to encode images
        *   [Dino](https://github.com/facebookresearch/dino) is an encoder trained using self supervision which reaches high knn classification performance
        *   [Face embeddings](https://github.com/ageitgey/face_recognition) compute face embeddings
    *   Text
        *   [Labse](https://tfhub.dev/google/LaBSE/2) a bert text encoder trained for similarity that put sentences from 109 in the same space
    *   Misc
        *   [Jina examples](https://github.com/jina-ai/examples) provide example on how to use pretrained encoders to build search systems 
        *   [Vectorhub](https://github.com/vector-ai/vectorhub) image, text, audio encoders
*   **Similarity learning** allows you to build new similarity encoders
    *   [Fine tuning classification with keras](https://keras.io/guides/transfer_learning/) enables adapting an existing image encoder to a custom dataset
    *   [Fine tuning classification with hugging face](https://huggingface.co/transformers/training.html) makes it possible to adapt existing text encoders
    *   [Lightly](https://github.com/lightly-ai/lightly) is a simple way to train image encoders with self supervision
    *   [Pytorch big graph](https://github.com/facebookresearch/PyTorch-BigGraph) library to encode a graph as node and link embeddings
    *   [RSVD](https://github.com/criteo/Spark-RSVD) a spark library to compute large scale svd with spark
    *   [Groknet](https://ai.facebook.com/blog/powered-by-ai-advancing-product-understanding-and-building-new-shopping-experiences/) Using image and categories and many datasets to fine tune product embeddings with many losses
*   **Indexing and approximate knn**: indexing make it possible to create small indices encoding million of embeddings that can be used to query the data in milli seconds
    *   [Faiss](https://github.com/facebookresearch/faiss) Many aknn algorithms (ivf, hnsw, flat, gpu, …) in c++ with a python interface
    *   [Autofaiss](https://github.com/criteo/autofaiss) to use faiss easily
    *   [Nmslib](https://github.com/nmslib/nmslib) fast implementation of hnsw
    *   [Annoy](https://github.com/spotify/annoy) a aknn algorithm by spotify
    *   [Scann](https://github.com/google-research/google-research/tree/master/scann) a aknn algorithm faster than hnsw by google
    *   [Catalyzer](https://arxiv.org/pdf/1806.03198.pdf) training the quantizer with backpropagation
*   **Search pipelines** allow fast serving and customization of how the indices are queries
    *   [Milvus](https://github.com/milvus-io/milvus) end to end similarity engine, on top of faiss and hnswlib
    *   [Jina](https://github.com/jina-ai/jina) flexible end to end similarity engine
    *   [Haystack](https://github.com/deepset-ai/haystack) question answering on text pipeline
*   **Companies**: many companies are being built around semantic search systems
    *   [Jina](https://jina.ai/) is building flexible pipeline to encode and search with embeddings
    *   [Weaviate](https://github.com/semi-technologies/weaviate) is building a cloud-native vector search engine
    *   [Pinecone](https://techcrunch.com/2021/01/27/pinecone-lands-10m-seed-for-purpose-built-machine-learning-database/?guccounter=1) a startup building databases indexing embeddings
    *   [Vector ai](https://hub.getvectorai.com/) is building an encoder hub
    *   [Milvus](https://milvus.io/) builds an end to end open source semantic search system
    *   [FeatureForm's embeddinghub](https://github.com/featureform/embeddinghub) combining DB and KNN
    *   Many other companies are using these systems and releasing open tools on the way, and it would be too long a list to put them here (for example facebook with faiss and self supervision, google with scann and thousand of papers, microsoft with sptag, spotify with annoy, criteo with rsvd, deepr, autofaiss, …)
