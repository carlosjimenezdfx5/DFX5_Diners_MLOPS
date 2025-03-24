# Financial Product Recommendation System for Diners Club

## Purpose

This repository contains the foundational structure for building a modern, secure, and scalable MLOps pipeline to power Diners Club's personalized financial product recommendation engine. The goal is to enable contextual, real-time recommendations under six seconds while maintaining full data governance, version control, and operational observability.

## Context

As part of Diners Club's strategic migration from on-premise infrastructure (e.g., IBM DataStage, DB2) to AWS, this project aims to build a production-grade machine learning system using Amazon SageMaker and supporting services. The system empowers data scientists, engineers, and business domains with autonomy, governance, and accelerated experimentation while ensuring traceability and compliance.

## Guiding Principles

- **Reproducibility**: All training workflows are versioned, parameterized, and tracked.
- **Observability**: End-to-end monitoring of pipelines, data quality, and model drift.
- **Modularity**: Each domain operates independently under a federated DataMesh architecture.
- **Governance**: Permissions, encryption, and access policies are embedded throughout.
- **Scalability**: Elastic infrastructure adapts to workload demand across domains.

## System Overview

### Architecture Highlights

- **Data Ingestion & Storage**: Raw domain data (e.g., transactions, profiles) ingested into Amazon S3.
- **Metadata Management**: AWS DataZone enables version control, access provisioning, and lineage tracking.
- **Preprocessing**: Data wrangling in SageMaker with built-in normalization, encoding, and missing value handling.
- **Feature Engineering**: SageMaker Feature Store centralizes processed variables for reuse across models.
- **Experimentation**:
  - SageMaker Notebooks for exploratory analysis.
  - SageMaker Experiments for hyperparameter tuning.
  - Model tracking and explainability via SageMaker Clarify.
- **Deployment**:
  - Models exposed via SageMaker Endpoints.
  - Triggered through Lambda functions and consumed through API Gateway.
- **Monitoring**:
  - Model performance tracked via SageMaker Model Monitor.
  - Data quality checks with Glue Data Quality.
- **Security & IAM**:
  - Role-based access control via SageMaker Role Manager.
  - Encryption, network constraints, and policy auto-generation embedded into setup.

## MLOps Lifecycle

### 1. DataOps Foundations
- Implemented pipelines for ingesting data from legacy IBM systems.
- Feature pipelines prepared with SageMaker Processing.
- Established data validation and anomaly detection.

### 2. ML System Design
- Defined CI/CD structure for model training and deployment.
- Maintained reproducible training pipelines and dataset versioning.
- IAM policies and model permissions standardized per team.

### 3. Experimentation & Governance
- Each experiment logged and compared within SageMaker Studio.
- Data scientists subscribe to assets via DataZone.
- Published datasets, models, and features to catalog for domain reuse.

### 4. Deployment & Serving
- Real-time API endpoints served via SageMaker.
- Batch scoring jobs integrated into domain workflows.
- Inference time monitored and optimized (<6s target).

### 5. Continuous Monitoring
- Alerting for data drift, feature attribution shifts, and missing monitoring jobs.
- Evaluation results visualized and linked to model cards.

## Team Roles & Responsibilities

- **Data Engineers**: Build pipelines, secure data access, and monitor ingestion.
- **Data Scientists**: Train models, conduct experiments, and publish assets.
- **ML Engineers**: Build CI/CD flows, deploy models, ensure compliance.
- **Software Engineers**: Expose APIs, integrate with apps, and scale endpoints.

## Outcomes & Benefits

- Domain-driven ownership and autonomy over ML pipelines.
- Shorter time-to-market for experiments and deployments.
- Secure, governed, and observable machine learning stack.
- Foundation for scalable experimentation with LLMs and multilingual corpora.

## Getting Started

> This README outlines architectural intent. For hands-on guides, explore the `/infrastructure`, `/pipelines`, and `/models` directories for Terraform scripts, ML pipelines, and training code.




