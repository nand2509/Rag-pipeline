"""
RAG Pipeline — main entry point
Run this file with: python main.py
"""

import os
from dotenv import load_dotenv
from rag.pipeline import RAGPipeline

load_dotenv()


def demo_with_text():
    print("=" * 60)
    print("RAG PIPELINE INTERACTIVE DEMO")
    print("=" * 60)

    knowledge = """
   ================================================================================
APACHE SPARK - COMPLETE KNOWLEDGE BASE
================================================================================

Apache Spark is an open-source, distributed computing framework designed for
large-scale data processing and analytics. It was originally developed at
UC Berkeley's AMPLab in 2009 by Matei Zaharia and was donated to the Apache
Software Foundation in 2013. Spark has since become one of the most widely
used big data processing engines in the world, with adoption at companies
including Netflix, Uber, Airbnb, Facebook, Microsoft, and Amazon.

Spark's core innovation is its ability to perform in-memory computation,
keeping intermediate data in RAM rather than writing it to disk after every
step. This approach makes Spark up to 100 times faster than Hadoop MapReduce
for certain iterative workloads such as machine learning and graph processing.
For batch processing workloads, Spark is typically 10 to 30 times faster than
MapReduce.

SPARK ARCHITECTURE:
Spark follows a master-worker architecture. Every Spark application consists
of a Driver Program and one or more Executors. The Driver Program runs the
main() function of the application and creates the SparkContext, which
coordinates all distributed operations. The Driver communicates with the
Cluster Manager to negotiate resources and schedule tasks on worker nodes.
Executors are JVM processes that run on worker nodes and are responsible for
executing tasks assigned by the Driver. Each Executor has a fixed number of
CPU cores and a fixed amount of memory. Executors run for the entire lifetime
of the Spark application and return results to the Driver.

Spark supports four cluster managers: Standalone (built into Spark), Apache
YARN (used in Hadoop environments), Apache Mesos, and Kubernetes. In local
mode, Spark runs entirely on a single machine using multiple threads, which
is useful for development and testing.

SPARK COMPONENTS:
Spark Core is the foundation of the entire platform, providing task scheduling,
memory management, fault recovery, and the RDD API. Spark SQL provides
structured data processing with DataFrames, Datasets, and a SQL query
interface. Spark Streaming enables real-time data processing using micro-batch
processing or the newer Structured Streaming engine. MLlib is Spark's built-in
machine learning library containing algorithms for classification, regression,
clustering, collaborative filtering, and feature engineering. GraphX is Spark's
API for graph computation and graph-parallel algorithms.

RDD (RESILIENT DISTRIBUTED DATASETS):
RDDs are the fundamental data abstraction in Spark. An RDD is an immutable,
distributed collection of objects partitioned across the nodes of a cluster.
The word Resilient refers to the fault tolerance mechanism: if a partition is
lost due to node failure, Spark can recompute it from its lineage — the record
of transformations used to build it. This lineage-based fault recovery
eliminates the need for data replication.

RDDs support two types of operations: Transformations and Actions.
Transformations are lazy operations that create a new RDD from an existing one
without immediately executing any computation. Examples include map(), filter(),
flatMap(), groupByKey(), reduceByKey(), and join(). Actions trigger the
execution of the entire transformation chain and return results to the Driver
or write data to storage. Examples include collect(), count(), first(), take(),
saveAsTextFile(), and reduce().

Lazy evaluation is a key design choice in Spark. Because transformations are
not executed until an action is called, Spark can analyze the entire
computation plan and optimize it before execution, combining multiple
operations and reducing data movement across the network.

DATAFRAMES AND SPARK SQL:
DataFrames are the primary API for structured data in Spark 2.x and later.
A DataFrame is a distributed collection of data organized into named columns,
similar to a table in a relational database or a Pandas DataFrame in Python,
but distributed across a cluster. DataFrames are built on top of RDDs and
provide a higher-level abstraction with automatic query optimization through
the Catalyst query optimizer.

The Catalyst optimizer transforms logical query plans into optimized physical
execution plans. It applies rule-based optimizations such as predicate pushdown,
column pruning, and constant folding. It also performs cost-based optimization
to choose the most efficient join strategy, for example choosing a broadcast
hash join when one side of the join is small enough to fit in memory on each
executor.

Spark SQL allows developers and analysts to query structured data using
standard ANSI SQL syntax. DataFrames can be registered as temporary views and
queried using SQL strings. This makes Spark accessible to analysts who are
comfortable with SQL but do not know Scala or Python.

PERFORMANCE TUNING:
Partitioning is the most important factor in Spark performance. Too few
partitions means the cluster is underutilized. Too many partitions creates
excessive scheduling overhead. The recommended number of partitions is 2 to 4
times the total number of CPU cores in the cluster. The configuration property
spark.sql.shuffle.partitions controls the number of partitions created during
shuffle operations like groupBy and join. Its default value is 200, which is
too high for small datasets and too low for very large ones.

Data skew occurs when some partitions have significantly more data than others,
causing certain tasks to take much longer than the rest. Skew can be addressed
by salting keys, using broadcast joins for small tables, or enabling Adaptive
Query Execution (AQE) which was introduced in Spark 3.0. AQE automatically
detects and handles skewed partitions at runtime by splitting large partitions
into smaller ones.

Broadcast joins eliminate the shuffle step entirely by sending a copy of the
small table to every executor. This is triggered automatically when one side
of the join is smaller than the threshold set by
spark.sql.autoBroadcastJoinThreshold, which defaults to 10 megabytes. It can
also be forced manually using the broadcast() hint function.

Caching and persistence allow frequently accessed DataFrames to be stored in
memory or on disk to avoid recomputation. The cache() method stores data in
memory with deserialization. The persist() method allows specifying a
StorageLevel such as MEMORY_ONLY, MEMORY_AND_DISK, or DISK_ONLY. Always call
unpersist() when a cached DataFrame is no longer needed to free up memory.

================================================================================
PYSPARK - PYTHON API FOR APACHE SPARK
================================================================================

PySpark is the official Python API for Apache Spark. It allows Python
developers to write Spark programs using familiar Python syntax while
leveraging Spark's distributed computing engine. PySpark uses the Py4J library
to bridge Python and the Java Virtual Machine, allowing Python code to call
Java and Scala objects running in the JVM.

SETTING UP PYSPARK:
PySpark can be installed using pip: pip install pyspark. A SparkSession is the
entry point for all PySpark functionality. It replaces the older SparkContext,
SQLContext, and HiveContext. A SparkSession is created using the builder
pattern: SparkSession.builder.appName("MyApp").master("local[*]").getOrCreate()
The master("local[*]") setting runs Spark locally using all available CPU cores.

DATAFRAME OPERATIONS IN PYSPARK:
PySpark DataFrames support a rich set of operations. The select() method
chooses specific columns. The filter() or where() method removes rows that
do not satisfy a condition. The withColumn() method adds a new column or
replaces an existing one. The groupBy() method groups rows by one or more
columns, after which aggregation functions like count(), sum(), avg(), min(),
and max() can be applied. The join() method combines two DataFrames based on
a condition, supporting inner, left, right, full outer, left semi, and
left anti join types.

Window functions in PySpark enable calculations across a set of rows related
to the current row, similar to SQL window functions. Common window functions
include rank(), dense_rank(), row_number(), lag(), lead(), sum(), avg(), and
first(). A Window specification defines the partition, ordering, and frame
boundary. For example, Window.partitionBy("department").orderBy("salary")
creates a window that groups rows by department and orders them by salary
within each department.

USER DEFINED FUNCTIONS:
Python UDFs allow applying custom Python logic to DataFrame columns. However,
Python UDFs are slower than built-in Spark SQL functions because each row
must be serialized from the JVM to Python, processed, and serialized back.
Pandas UDFs, also called vectorized UDFs, solve this problem by processing
data in batches using Apache Arrow for zero-copy data transfer between the
JVM and Python. Pandas UDFs can be 10 to 100 times faster than regular
Python UDFs. There are three types of Pandas UDFs: Scalar, Grouped Map,
and Grouped Aggregate.

FILE FORMATS IN PYSPARK:
PySpark supports reading and writing multiple file formats. CSV is human
readable but slow and has no schema. JSON is flexible and supports nested
structures but is also slow. Parquet is a columnar binary format that is
highly compressed and efficient for analytical queries. It stores schema
information within the file and supports predicate pushdown. ORC is similar
to Parquet and is preferred in Hive environments. Avro is a row-based format
suitable for streaming and schema evolution. Delta Lake is an open-source
storage layer that adds ACID transactions, time travel, schema enforcement,
and upsert capability on top of Parquet files stored in cloud object storage.

STRUCTURED STREAMING:
Structured Streaming is a scalable, fault-tolerant stream processing engine
built on Spark SQL. It treats a live data stream as an unbounded table that
is continuously appended. Developers write the same DataFrame and SQL
operations they use for batch processing, and Spark handles the incremental
execution automatically. Structured Streaming supports reading from Kafka,
file systems, and socket sources. It supports three output modes: Append
(only new rows), Complete (entire result table), and Update (only changed
rows). Watermarking allows handling late-arriving data by defining how long
the system should wait before discarding state for a given event time.

================================================================================
ETL PROCESS - EXTRACT TRANSFORM LOAD
================================================================================

ETL stands for Extract, Transform, Load. It is the process of collecting data
from one or more source systems, transforming it to meet business requirements
and quality standards, and loading it into a target system such as a data
warehouse, data lake, or operational database. ETL is the foundation of data
engineering and business intelligence infrastructure.

EXTRACT PHASE:
The Extract phase collects raw data from source systems. Sources can include
relational databases such as PostgreSQL, MySQL, Oracle, and SQL Server,
queried using JDBC or ODBC connections. Other sources include NoSQL databases
like MongoDB, Cassandra, and DynamoDB, REST APIs returning JSON or XML data,
flat files in CSV, JSON, XML, or Excel format stored on FTP servers or cloud
storage, message queues like Apache Kafka and RabbitMQ, and SaaS platforms
like Salesforce, HubSpot, and Google Analytics.

Extraction strategies include Full Load, which extracts all data from the
source every time and is simple but inefficient for large datasets. Incremental
Load extracts only records that are new or changed since the last extraction,
using a watermark such as a timestamp column or an auto-incrementing ID.
Change Data Capture (CDC) reads the database transaction log to capture every
insert, update, and delete as it happens, providing near-real-time data
movement with minimal impact on the source system. Debezium is the most
popular open-source CDC tool and supports PostgreSQL, MySQL, SQL Server,
Oracle, and MongoDB.

TRANSFORM PHASE:
The Transform phase applies business logic and data quality rules to the
extracted data. Common transformations include data cleaning such as handling
null values, removing duplicates, fixing typos, and standardizing formats.
Data type conversion converts strings to dates, floats to integers, or
categorical strings to numeric codes. Filtering removes invalid, test, or
out-of-scope records. Aggregation computes summaries such as daily totals,
weekly averages, and monthly maximums. Joining enriches records by combining
them with reference data from dimension tables. Derived column creation adds
new fields calculated from existing ones, such as revenue equals quantity
multiplied by price minus discount.

Data quality checks should be applied at every stage of the pipeline. Common
checks include null checks to ensure required fields are populated, uniqueness
checks to detect duplicate primary keys, range checks to validate that numeric
values fall within expected bounds, referential integrity checks to verify that
foreign keys exist in the referenced table, and freshness checks to confirm
that data was updated within the expected time window. Great Expectations is
a popular Python library for defining and running data quality tests.

LOAD PHASE:
The Load phase writes transformed data into the target system. Load strategies
include Full Refresh which drops and recreates the target table on every run.
Append Only adds new records without modifying existing ones. Upsert, also
called Merge, inserts new records and updates existing records based on a
primary key match. Slowly Changing Dimensions or SCD handle changes to
dimension attributes over time. SCD Type 1 overwrites the old value with no
history. SCD Type 2 inserts a new row and sets the end date on the old row
to preserve full history. SCD Type 3 adds a previous value column to track
only the most recent change.

DATA WAREHOUSE CONCEPTS:
A data warehouse is a centralized repository designed for analytical queries.
It stores historical data from multiple source systems in a consistent,
integrated format. The star schema is the most common data warehouse design
pattern. It consists of a central fact table surrounded by dimension tables.
Fact tables store quantitative metrics such as sales amounts, order quantities,
and page view counts. Dimension tables store descriptive attributes used for
filtering and grouping, such as customer name, product category, and date.
The snowflake schema normalizes dimension tables into sub-tables, reducing
redundancy but requiring more joins.

ORCHESTRATION:
Apache Airflow is the most widely used workflow orchestration platform for
ETL pipelines. Airflow uses Python code to define Directed Acyclic Graphs
or DAGs. Each DAG represents a pipeline with tasks and dependencies between
them. Airflow provides scheduling, retry logic, alerting, and a web interface
for monitoring pipeline runs. Tasks in Airflow are implemented using Operators,
which are pre-built classes for common operations such as PythonOperator for
running Python functions, BashOperator for running shell commands,
PostgresOperator for executing SQL, and S3ToRedshiftOperator for loading
data from S3 to Amazon Redshift.

dbt, which stands for Data Build Tool, is a SQL-first transformation framework
used in ELT workflows. dbt connects directly to a data warehouse and executes
SQL transformations inside the warehouse engine. dbt manages dependencies
between models, runs tests, generates documentation automatically, and
integrates with version control systems like Git. dbt is widely used with
Snowflake, BigQuery, Redshift, and Databricks.

================================================================================
DEEP LEARNING - COMPLETE KNOWLEDGE BASE
================================================================================

Deep Learning is a subfield of Machine Learning that uses artificial neural
networks with many layers to learn representations from data. The term deep
refers to the number of layers in the network. Each layer learns increasingly
abstract features from the input. Deep Learning has achieved breakthrough
results in image recognition, natural language processing, speech recognition,
drug discovery, and game playing.

NEURAL NETWORK BASICS:
A neural network consists of layers of interconnected neurons. Each neuron
computes a weighted sum of its inputs, adds a bias term, and applies an
activation function. The weights and biases are the learnable parameters of
the network. They are initialized randomly and updated iteratively during
training to minimize a loss function.

The input layer receives the raw data such as pixel values, word embeddings,
or sensor readings. Hidden layers transform the data through successive
nonlinear transformations. The output layer produces the final prediction,
such as a probability distribution over classes for classification tasks
or a continuous value for regression tasks.

ACTIVATION FUNCTIONS:
Activation functions introduce nonlinearity into the network, enabling it to
learn complex patterns that a linear model cannot. The Sigmoid function maps
any input to a value between 0 and 1, making it useful for binary
classification output layers. However, sigmoid suffers from the vanishing
gradient problem because its gradient approaches zero for large positive or
negative inputs. The Tanh function maps inputs to values between -1 and 1
and is zero-centered, making it preferred over sigmoid for hidden layers in
recurrent networks. ReLU, which stands for Rectified Linear Unit, returns
the input if it is positive and zero otherwise. ReLU is the most popular
activation function for hidden layers in feedforward and convolutional
networks because it is simple, computationally efficient, and mitigates
the vanishing gradient problem. However, ReLU neurons can die if they
receive only negative inputs and output zero permanently. Leaky ReLU
addresses this by allowing a small negative slope. GELU, the Gaussian
Error Linear Unit, is used in Transformer models such as BERT and GPT.
Softmax converts a vector of real numbers into a probability distribution
and is used for multi-class classification output layers.

LOSS FUNCTIONS:
The loss function measures how wrong the model's predictions are compared to
the true labels. Mean Squared Error measures the average squared difference
between predicted and true values and is used for regression tasks. Cross
Entropy Loss measures the difference between two probability distributions and
is used for classification tasks. Binary Cross Entropy is a special case of
cross entropy for binary classification problems. Categorical Cross Entropy
is used for multi-class classification where each example belongs to exactly
one class. Focal Loss is a variant of cross entropy that down-weights easy
examples to focus training on hard examples, commonly used in object detection.

BACKPROPAGATION AND OPTIMIZATION:
Backpropagation is the algorithm used to compute the gradient of the loss
function with respect to every weight in the network. It works by applying
the chain rule of calculus repeatedly from the output layer back to the input
layer. The gradient tells the optimizer in which direction and by how much to
adjust each weight to reduce the loss.

Gradient Descent updates all weights using the gradients computed over the
entire training dataset. Stochastic Gradient Descent updates weights using
the gradient computed from a single training example or a small mini-batch.
Mini-batch gradient descent is the most practical approach, typically using
batch sizes of 32, 64, or 128 examples. The Adam optimizer combines momentum,
which accelerates updates in consistent directions, with adaptive learning
rates for each parameter. Adam typically converges faster than vanilla SGD
and requires less learning rate tuning. Its default hyperparameters are a
learning rate of 0.001, beta1 of 0.9, beta2 of 0.999, and epsilon of 1e-8.

CONVOLUTIONAL NEURAL NETWORKS:
Convolutional Neural Networks or CNNs are the dominant architecture for image
processing tasks. CNNs exploit the spatial structure of images through
convolutional layers that apply learnable filters to detect local patterns such
as edges, textures, and shapes. Max pooling layers downsample feature maps
by taking the maximum value in each local region, reducing spatial dimensions
and adding translation invariance. Fully connected layers at the end of the
network aggregate all features and produce the final classification.

Famous CNN architectures include LeNet-5, the first successful CNN developed
by Yann LeCun for handwritten digit recognition. AlexNet won the 2012 ImageNet
competition with a top-5 error rate of 15.3 percent, more than 10 percentage
points better than the second place entry, igniting the modern deep learning
revolution. VGGNet demonstrated that network depth using very small 3x3
convolution filters is a critical component of performance. ResNet introduced
residual connections that allow gradients to flow directly through skip
connections, enabling training of networks with hundreds or even thousands
of layers. EfficientNet uses neural architecture search to systematically
scale network width, depth, and resolution.

RECURRENT NEURAL NETWORKS:
Recurrent Neural Networks or RNNs are designed for sequential data such as
text, audio, and time series. RNNs maintain a hidden state that is updated
at each time step based on the current input and the previous hidden state.
This allows them to model dependencies between elements in a sequence.
However, vanilla RNNs suffer from the vanishing gradient problem when
processing long sequences, making it difficult to learn long-range dependencies.

LSTM, which stands for Long Short-Term Memory, solves the vanishing gradient
problem through a gating mechanism. An LSTM cell contains three gates: the
forget gate controls what information to discard from the cell state, the
input gate controls what new information to store, and the output gate controls
what information from the cell state to output. The cell state acts as a
memory conveyor belt that can carry information across many time steps without
significant gradient degradation.

TRANSFORMERS:
The Transformer architecture was introduced in the 2017 paper Attention Is All
You Need by Vaswani et al. at Google. Transformers replaced RNNs as the
dominant architecture for natural language processing tasks. Unlike RNNs,
Transformers process all tokens in the input sequence in parallel, making them
much faster to train on modern GPU hardware.

The core mechanism of the Transformer is self-attention. Self-attention allows
each token in the sequence to attend to every other token, computing a
weighted sum of all token representations where the weights reflect the
relevance of each token to the current one. Multi-head attention runs multiple
attention operations in parallel, allowing the model to jointly attend to
information from different representation subspaces.

BERT, which stands for Bidirectional Encoder Representations from Transformers,
was introduced by Google in 2018. BERT is a pre-trained Transformer encoder
that reads the entire input sequence bidirectionally. It is trained using
two self-supervised objectives: Masked Language Modeling, which predicts
randomly masked tokens, and Next Sentence Prediction. BERT achieves
state-of-the-art results on many NLP benchmarks when fine-tuned on
task-specific data with a small amount of labeled examples.

GPT, which stands for Generative Pre-trained Transformer, is a family of
autoregressive language models developed by OpenAI. GPT models are Transformer
decoders trained to predict the next token in a sequence. GPT-3 has 175
billion parameters and demonstrated remarkable few-shot learning capabilities.
GPT-4 is a multimodal model that accepts both text and image inputs.

================================================================================
RETRIEVAL AUGMENTED GENERATION (RAG)
================================================================================

Retrieval Augmented Generation, commonly abbreviated as RAG, is a technique
that enhances large language model responses by retrieving relevant information
from an external knowledge base before generating an answer. RAG addresses the
key limitations of LLMs, including knowledge cutoff dates, hallucinations, and
the inability to access private or proprietary data.

RAG ARCHITECTURE:
A RAG system consists of two main pipelines: the indexing pipeline and the
query pipeline. The indexing pipeline runs once or on a schedule to prepare
the knowledge base. It loads documents from files, databases, or APIs, splits
them into smaller chunks using a text splitter, converts each chunk into a
dense vector embedding using an embedding model, and stores the vectors in a
vector database. The query pipeline runs every time a user asks a question.
It converts the user query into a vector embedding using the same embedding
model, performs a similarity search in the vector database to retrieve the
most relevant chunks, constructs a prompt that includes the retrieved context
and the user question, and passes the prompt to a large language model which
generates a final answer grounded in the retrieved information.

TEXT SPLITTING:
Documents must be split into chunks before embedding because embedding models
have a maximum input length and because embedding entire documents produces
vectors that capture the general topic rather than specific facts. The chunk
size controls the amount of text in each chunk. Larger chunks provide more
context but may dilute the embedding signal. Smaller chunks are more focused
but may lack sufficient context for the LLM to generate a complete answer.
The chunk overlap ensures that information near chunk boundaries is not lost
by including the last N characters of each chunk in the beginning of the next.
Common chunk sizes range from 256 to 1024 characters with overlaps of 10 to
20 percent of the chunk size.

EMBEDDING MODELS:
Embedding models convert text into dense numerical vectors in a high-dimensional
space where semantically similar texts are close together. OpenAI's
text-embedding-3-small and text-embedding-3-large are high-quality embedding
models available via API. For local use without an API key, sentence-transformers
provides a large collection of open-source embedding models. The all-MiniLM-L6-v2
model is a popular choice for its balance of quality and speed. The
BAAI/bge-large-en-v1.5 model achieves higher quality at the cost of slower
inference.

VECTOR DATABASES:
Vector databases store embeddings and support efficient similarity search.
FAISS, which stands for Facebook AI Similarity Search, is a library for
efficient similarity search on dense vectors. It supports multiple index types
including Flat (exact search), IVF (inverted file, approximate search), and
HNSW (hierarchical navigable small world graph). FAISS runs entirely in memory
and on a single machine, making it suitable for development and small-to-medium
datasets. Chroma is an open-source embedding database built on top of SQLite
for persistence. It provides a simple Python API and is easy to set up locally.
Pinecone is a managed vector database service offering low-latency search at
scale without requiring infrastructure management. Weaviate and Qdrant are
other production-grade open-source vector databases that support filtering,
multi-tenancy, and hybrid search combining dense and sparse vectors.

LANGCHAIN:
LangChain is a Python and JavaScript framework for building applications with
large language models. It provides abstractions for document loaders, text
splitters, embedding models, vector stores, prompt templates, LLM wrappers,
chains, and agents. LangChain simplifies RAG implementation by providing
pre-built integrations with hundreds of data sources, embedding models, vector
stores, and LLMs. The RetrievalQA chain is a built-in LangChain chain that
combines a retriever and an LLM to answer questions from a knowledge base.
LCEL, which stands for LangChain Expression Language, is a declarative syntax
for composing chains using the pipe operator.

ADVANCED RAG TECHNIQUES:
Hybrid search combines dense vector search with sparse keyword search using
BM25 or TF-IDF to improve retrieval quality, especially for queries containing
rare or domain-specific terms. Reranking applies a cross-encoder model to
reorder the initial retrieval results, improving precision at the cost of
additional latency. HyDE, which stands for Hypothetical Document Embeddings,
generates a hypothetical answer to the query, embeds it, and uses that
embedding for retrieval instead of the query embedding directly, which can
improve retrieval for complex questions. Query decomposition breaks a complex
multi-part question into simpler sub-questions, retrieves context for each,
and synthesizes a final answer. Self-RAG trains the LLM to decide when to
retrieve, evaluate retrieved passages for relevance, and assess its own
generated output for correctness.

================================================================================
MACHINE LEARNING FUNDAMENTALS
================================================================================

Machine Learning is a subfield of artificial intelligence that enables systems
to learn from data without being explicitly programmed. Machine Learning
algorithms build mathematical models from training data to make predictions
or decisions on new, unseen data.

SUPERVISED LEARNING:
In supervised learning, the training data consists of input-output pairs where
the correct output is known. The algorithm learns a mapping from inputs to
outputs. Classification predicts a discrete category label. Regression predicts
a continuous numeric value. Common supervised learning algorithms include
Linear Regression, Logistic Regression, Decision Trees, Random Forests,
Gradient Boosted Trees, Support Vector Machines, and Neural Networks.

UNSUPERVISED LEARNING:
In unsupervised learning, the training data has no labels and the algorithm
must discover structure in the data on its own. Clustering algorithms such as
K-Means, DBSCAN, and Hierarchical Clustering group similar data points together.
Dimensionality reduction algorithms such as PCA, t-SNE, and UMAP project
high-dimensional data into a lower-dimensional space while preserving structure.
Autoencoders learn compressed representations of data by training an encoder
to compress inputs and a decoder to reconstruct them.

EVALUATION METRICS:
For classification tasks, accuracy measures the fraction of correctly classified
examples. Precision measures the fraction of predicted positives that are truly
positive. Recall measures the fraction of true positives that are correctly
identified. The F1 score is the harmonic mean of precision and recall. ROC-AUC
measures the area under the Receiver Operating Characteristic curve, reflecting
the model's ability to discriminate between classes across all decision
thresholds. For regression tasks, Mean Absolute Error measures the average
absolute difference between predictions and true values. Mean Squared Error
penalizes large errors more heavily. R-squared measures the proportion of
variance in the target explained by the model.

OVERFITTING AND REGULARIZATION:
Overfitting occurs when a model learns the training data too well, including
its noise and outliers, resulting in poor generalization to new data. Signs
of overfitting include high training accuracy and low validation accuracy.
Regularization techniques reduce overfitting. L2 regularization adds the
sum of squared weights to the loss function, penalizing large weights and
encouraging simpler models. L1 regularization adds the sum of absolute weights
and produces sparse models where many weights are exactly zero. Dropout
randomly sets a fraction of neuron activations to zero during training,
preventing neurons from co-adapting and acting as an ensemble of many networks.
Data augmentation artificially increases the size of the training set by
applying transformations such as rotation, flipping, cropping, and color
jitter to existing training examples. Early stopping monitors validation loss
during training and stops when it begins to increase, preventing the model
from memorizing the training data.

================================================================================
PYTHON FOR DATA ENGINEERING
================================================================================

Python is the dominant programming language for data engineering, data science,
and machine learning. Its readable syntax, extensive standard library, and rich
ecosystem of third-party packages make it the preferred choice for building
data pipelines, training machine learning models, and deploying AI applications.

KEY LIBRARIES:
NumPy provides efficient multi-dimensional array operations and mathematical
functions. It is the foundation for most scientific computing in Python.
Pandas provides the DataFrame data structure for working with tabular data,
along with tools for reading and writing files, handling missing values,
performing aggregations, and joining datasets. Matplotlib and Seaborn provide
data visualization capabilities. Scikit-learn provides a consistent API for
classical machine learning algorithms including preprocessing, feature
engineering, model training, cross-validation, and evaluation.

SQLAlchemy is the standard library for database connectivity in Python,
providing both a high-level ORM and a low-level SQL expression language.
Psycopg2 is the most popular PostgreSQL adapter for Python. Requests simplifies
making HTTP calls to REST APIs. FastAPI is a modern, high-performance web
framework for building APIs with automatic OpenAPI documentation. Pydantic
provides data validation using Python type annotations.

VIRTUAL ENVIRONMENTS:
A virtual environment is an isolated Python environment that contains its own
Python interpreter and installed packages, separate from the system Python.
Virtual environments prevent package conflicts between different projects.
The built-in venv module creates virtual environments. After creating a virtual
environment with python -m venv venv, it must be activated before installing
packages. On Windows, activation is done with venv\Scripts\activate. On Mac
and Linux, it is done with source venv/bin/activate. Once activated, pip
install commands install packages into the virtual environment rather than the
system Python. Requirements can be exported with pip freeze > requirements.txt
and restored on another machine with pip install -r requirements.txt.

================================================================================
END OF KNOWLEDGE BASE
================================================================================
    """

    # Build RAG pipeline
    rag = RAGPipeline(
        use_openai=False,
        index_path="faiss_index",
        chunk_size=300,
        chunk_overlap=30,
        top_k=3,
    )

    # Build index
    rag.index(knowledge, force_rebuild=True)

    print("\nIndex built successfully.")
    print("You can now ask unlimited questions.")
    print("Type 'exit' or 'quit' to stop.\n")

    print("=" * 60)

    # 🔹 INTERACTIVE LOOP
    while True:

        query = input("\nAsk a question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("\nExiting RAG system...")
            break

        if not query:
            continue

        print("\nSearching knowledge base...")
        answer = rag.ask(query)

        print("\nAnswer:")
        print("-" * 40)
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    demo_with_text()