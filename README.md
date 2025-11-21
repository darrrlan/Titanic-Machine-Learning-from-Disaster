flowchart LR
subgraph External
A[Sistemas Fonte: RH, ERP, Fornecedores, CSV/Excel] -->|API/DB Connect| Ingest
B[Usuário (Web UI / Admin)] -->|HTTPS| API
C[Serviço de IA/LLM (OpenAI/Local)] -->|genera contratos| ContractSvc
end


subgraph Ingestion & Streaming
Ingest[Ingest: Connectors / ETL] -->|events| KafkaTopicA[(Kafka Topics)]
Airflow[(Airflow DAGs)] -->|orquestra| Ingest
end


subgraph Storage & Indexing
KafkaTopicA -->|sink| MongoDB[(MongoDB)]
MongoDB -->|CDC (Debezium)| KafkaTopicB[(Kafka CDC)]
KafkaTopicB -->|sink| Elasticsearch[(Elasticsearch)]
KafkaTopicB -->|sink| MinIO[(S3 / MinIO)]
end


subgraph Query & Access
Dremio[(Dremio / Query Engine)] -->|virtual datasets| API[Quarkus API]
API -->|cache| Redis[(Redis Cache)]
API -->|read/write| Postgres[(Postgres / Primary DB)]
API -->|search| Elasticsearch
API -->|storage| MinIO
ContractSvc[Contract Generator Service] -->|requests| API
ContractSvc -->|uses| LLM[Cognitive / LLM]
end


subgraph Infra
Kafka[(Kafka Cluster)]
Connect[(Kafka Connect)]
Debezium[(Debezium Connector)]
K8s[(Kubernetes)]
Observability[Prometheus + Grafana + ELK]
end


KafkaTopicA --> Kafka
KafkaTopicB --> Kafka
Connect --> Kafka
Debezium --> Kafka
API --> K8s
MongoDB --> K8s
Elasticsearch --> K8s
MinIO --> K8s
Dremio --> K8s
Redis --> K8s


style K8s fill:#f3f4f6,stroke:#333,stroke-width:1px
style Observability fill:#fff7ed,stroke:#333
