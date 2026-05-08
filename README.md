# Stocklytics AI

Stocklytics AI is a retail operations platform for small and medium stores. It combines inventory management, billing, customer records, alerts, analytics, and a grounded AI assistant in one workspace built for day-to-day store operations.

The project is implemented as a modular monolith backend with a modern Next.js frontend, and it is designed to run well on Google Cloud using Firestore, BigQuery, Cloud Run, and local Gemma open models.

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Product Modules](#product-modules)
- [AI Assistant](#ai-assistant)
- [Architecture](#architecture)
- [Google Cloud and Firebase Usage](#google-cloud-and-firebase-usage)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Local Development](#local-development)
- [Environment Configuration](#environment-configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Operational Scripts](#operational-scripts)
- [Security Notes](#security-notes)
- [Roadmap Ideas](#roadmap-ideas)

## Hackathon Journey

### The Genesis: Google Solution Challenge
Stocklytics AI was originally built for the **Solution Challenge by Hack2skill and Google**. In its initial version, the platform relied heavily on cloud-based AI, specifically the **Gemini API (`gemini-2.0-flash` and `gemini-embedding-001`)** for both product vectorization and chat generation. This allowed us to build a robust proof-of-concept for intelligent retail operations.

### The Evolution: Gemma for Good Hackathon
To participate in the **Gemma for Good Hackathon**, the architecture was entirely re-engineered and optimized for Edge AI. We disconnected all external cloud AI services and pivoted the codebase to run **100% locally using Open Weights**.
- **Before**: Cloud-dependent. Data sent to Google's Gemini API for processing and embedding.
- **After**: Edge-optimized. The AI Assistant now runs natively on local hardware using `google/gemma-4-4b-it` (via PyTorch/Transformers) and `sentence-transformers/all-MiniLM-L6-v2` for offline embeddings.

### Target Track: Digital Equity & Inclusivity
We are submitting this project under the **Digital Equity & Inclusivity** track. Small and medium retail operators (mom-and-pop shops) often lack access to enterprise-grade AI or the technical skills to implement it. By bringing a powerful, intuitive, edge-ready AI directly to their operational dashboard, Stocklytics AI bridges the AI skills gap, democratizing advanced retail analytics without relying on expensive cloud subscriptions.

## Overview

Stocklytics AI helps a retail operator answer practical questions such as:

- What products are low on stock?
- Which items are expiring soon?
- What sold best today?
- Which customers are buying the most?
- What should I restock next?
- What changed in the business since the last update?

Instead of splitting those jobs across multiple tools, the platform keeps operational data, analytics snapshots, and AI guidance in one system.

At a high level:

- `frontend/` provides the operator-facing web app
- `backend/` provides the API, auth, domain logic, analytics access, alerts, and AI services
- `infra/cloudrun/` contains deployment templates for Google Cloud Run and Cloud Build
- `backend/scripts/` contains seeding, sync, transform, and E2E helper scripts

## Core Features

- Retail dashboard with sales, transactions, alert counts, and stock health
- Inventory management with create, update, stock adjustments, and low-stock workflows
- Billing and transaction creation with idempotency protection
- Customer management with profile creation and purchase history
- Alerts engine for low stock, expiry risk, high demand, and non-selling products
- Analytics views backed by BigQuery mart tables
- Grounded AI assistant that answers using live store context and retrieval-augmented product evidence
- Authenticated multi-user access using Firebase Authentication and backend claim validation
- Cloud-ready deployment model for frontend and backend as separate services

## Product Modules

The backend is organized into domain modules, each with its own router, schemas, service, and repository layer where appropriate.

### Inventory

- Product CRUD
- Stock adjustment tracking
- Reorder threshold monitoring
- Expiry metadata and inventory status handling

### Billing

- Transaction creation
- Stock deduction during billing
- Idempotency handling to prevent duplicate charges
- Integration with downstream analytics and alert workflows

### Customers

- Customer profile creation and listing
- Purchase history lookup
- Data used by both analytics and AI assistant features

### Alerts

- Low stock alert generation
- Expiry-soon and expired-item monitoring
- Not-selling and high-demand detection
- Acknowledge and resolve workflows

### Analytics

- Dashboard summary
- Sales trends
- Product performance
- Customer insights
- Freshness metadata for downstream consumers

### AI

- Chat sessions with persisted message history
- Grounded retrieval from inventory, analytics, alerts, customers, and transactions
- Product embedding sync into BigQuery
- Multi-model routing for fast answers and deeper reasoning

### Data Pipeline

- Incremental sync from Firestore to BigQuery raw tables
- Mart transformations for analytics views
- Failure tracking and repair workflows
- Admin-triggered sync endpoints

## AI Assistant

The AI assistant is designed for operational retail questions, not open-ended general chat.

Current design goals:

- Answer from store data instead of generic LLM behavior
- Use exact and semantic product retrieval for product-specific questions
- Run 100% locally on edge hardware using the `google/gemma-4-4b-it` model via PyTorch/Transformers
- Return answer provenance such as intent, retrieval confidence, and grounding metadata
- Fall back gracefully when parts of the context are unavailable

Current backend AI flow includes:

1. Query intent detection
2. Selective context loading
3. Hybrid retrieval using Firestore snapshots plus BigQuery vector search
4. Evidence-pack prompt construction
5. Model routing and answer generation
6. Response normalization and persistence

Relevant AI endpoints:

- `POST /api/v1/ai/chat`
- `GET /api/v1/ai/chat/sessions/{chat_session_id}`
- `POST /api/v1/ai/embed-sync`

## Architecture

The backend follows a modular monolith style. Each domain lives inside `backend/app/modules/<domain>/`.

Common backend layering pattern:

- `router.py`
  FastAPI endpoints and request validation
- `schemas.py`
  Pydantic request/response models
- `service.py`
  Business logic and orchestration
- `repository.py`
  Firestore or BigQuery access

Shared backend infrastructure lives in:

- `backend/app/common/`
  config, auth, middleware, exceptions, logging, shared response helpers
- `backend/app/api/`
  platform and administrative endpoints

The frontend is a Next.js App Router application. Pages live under `frontend/app/`, shared components under `frontend/components/`, and runtime/service helpers under `frontend/lib/`.

### Backend Request Flow

Typical API request flow:

1. Request enters FastAPI app at [main.py](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/backend/app/main.py:1)
2. Middleware attaches request metadata and logging context
3. Firebase-backed auth validates the bearer token and extracts role/store claims
4. Router validates payload and delegates to a domain service
5. Service coordinates repositories, rules, downstream reads, and persistence
6. Standard response helper injects `request_id`

### Frontend Runtime Modes

The frontend supports multiple runtime modes via [runtime.ts](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/frontend/lib/runtime.ts:1):

- `firebase`
  Real Firebase web authentication
- `backend_stub_auth`
  Real backend with local stub auth when Firebase web config is absent
- `mock_api`
  Fully mocked frontend preview mode

This makes it easier to develop independently across backend, frontend, and cloud environments.

## Google Cloud and Firebase Usage

This project is tightly integrated with Google services.

### Firebase

- Firebase Authentication for operator sign-in
- Firebase Admin SDK on backend for token verification
- Custom claims (`role`, `store_id`) used for store-scoped authorization

### Firestore

- Primary operational datastore
- Stores products, customers, transactions, stock adjustments, alerts, metadata, and AI chat sessions
- Supports the real-time store state used by API modules and AI retrieval

### BigQuery

- Raw tables for synced operational data
- Mart tables for analytics summaries and reporting views
- Product embedding storage for vector search
- Retrieval source for analytics and AI grounding

### Gemma / Local AI Models

- Query embeddings using offline `sentence-transformers/all-MiniLM-L6-v2`
- Fast, local chat generation using the native `google/gemma-4-4b-it` model
- Perfect for Edge deployment and 100% compliant with the "Gemma for Good" hackathon
- Used by the AI assistant only after context is assembled from operational data

### Cloud Run

- `stocklytics-backend` runs the FastAPI API
- `stocklytics-frontend` runs the Next.js standalone app
- Separate deployment surface keeps backend and frontend scaling concerns clean

### Cloud Build

- Builds backend and frontend images
- Injects environment-specific build/runtime configuration during deployment

### Artifact Registry

- Stores backend and frontend container images for Cloud Run deployment

## Repository Structure

```text
Stocklytics-AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── common/
│   │   └── modules/
│   │       ├── ai/
│   │       ├── alerts/
│   │       ├── analytics/
│   │       ├── billing/
│   │       ├── customer/
│   │       ├── data_pipeline/
│   │       └── inventory/
│   ├── scripts/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── package.json
├── infra/
│   └── cloudrun/
├── docs/
└── README.md
```

## Tech Stack

### Frontend

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Lucide icons
- Axios
- Firebase Web SDK
- Vitest for frontend unit tests

### Backend

- FastAPI
- Pydantic v2
- Firebase Admin SDK
- Google Cloud Firestore client
- Google Cloud BigQuery client
- HTTPX
- Python dotenv
- Pytest and pytest-asyncio

## Local Development

### Prerequisites

- Python 3.11+ or compatible runtime for the backend toolchain
- Node.js 20+ recommended for the frontend
- A Firebase project and Google Cloud project if you want to run against real services
- Firestore and BigQuery enabled for full-stack integration

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Backend docs and health endpoints:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/api/v1/ready`

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Frontend dev URL:

- `http://127.0.0.1:3000`

### Local Auth Options

- Use real Firebase web auth by providing Firebase frontend config values
- Use backend stub auth for local backend work when Firebase web config is absent
- Use mock mode for purely frontend exploration

## Environment Configuration

Key backend environment variables are defined in:

- [backend/.env.example](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/backend/.env.example:1)
- [infra/cloudrun/backend.env.yaml.example](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/infra/cloudrun/backend.env.yaml.example:1)

Important backend values:

- `APP_ENV`
- `CORS_ALLOW_ORIGINS`
- `FIREBASE_PROJECT_ID`
- `FIRESTORE_PROJECT_ID`
- `BIGQUERY_PROJECT_ID`
- `GEMMA_MODEL_ID`
- `GEMMA_EMBEDDING_MODEL`

Key frontend environment variables are defined in:

- [frontend/.env.example](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/frontend/.env.example:1)
- [infra/cloudrun/frontend.env.yaml.example](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/infra/cloudrun/frontend.env.yaml.example:1)

Important frontend values:

- `NEXT_PUBLIC_API_BASE_URL`
- `BACKEND_URL`
- `NEXT_PUBLIC_FIREBASE_API_KEY`
- `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
- `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
- `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`
- `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`
- `NEXT_PUBLIC_FIREBASE_APP_ID`
- `NEXT_PUBLIC_USE_MOCKS`
- `NEXT_PUBLIC_AUTO_LOGIN_DEMO`

## Testing

### Backend

```bash
cd backend
pytest
```

### Frontend

```bash
cd frontend
npm test
```

### E2E Helper

There is also a backend E2E validation script:

```bash
python backend/scripts/run_backend_e2e_check.py
```

That script validates:

- platform endpoints
- inventory flows
- customer flows
- billing behavior
- alerts endpoints
- optional analytics and AI checks

## Deployment

Cloud deployment assets live in `infra/cloudrun/`.

Read the detailed cloud guide here:

- [infra/cloudrun/README.md](/run/media/shrey/Data/Team%20369/Solution%20Challenge/Stocklytics-AI/infra/cloudrun/README.md:1)

Deployment model:

- Backend deployed as a Cloud Run FastAPI service
- Frontend deployed as a Cloud Run Next.js service
- Images built with Cloud Build and stored in Artifact Registry

Production recommendation:

- Prefer Cloud Run service account permissions over injecting raw private keys
- Use Secret Manager for sensitive runtime values where possible
- Grant backend service account Firestore and BigQuery roles explicitly

## Operational Scripts

Useful backend scripts in `backend/scripts/`:

- `seed_one_month_data.py`
  Seeds demo-style retail data
- `reset_and_seed_one_month_data.py`
  Clears one store, reseeds one month, rebuilds alerts, analytics, and embeddings
- `run_sync_job.py`
  Runs Firestore to BigQuery sync
- `run_transform_job.py`
  Builds mart tables from raw data
- `run_embedding_sync.py`
  Regenerates product embeddings
- `run_alerts_sweep.py`
  Recomputes alerts from operational state
- `run_repair_job.py`
  Repairs pipeline failures

## Security Notes

- Do not commit service-account JSON files, local auth key notes, or real env files
- Prefer Application Default Credentials on Cloud Run over shipping long-lived private keys
- Rotate any Firebase or Google Cloud credentials immediately if they are ever exposed
- Keep `infra/cloudrun/*.env.yaml` local and out of git
- Review `.gitignore` and `.gcloudignore` before deployment so local secrets are excluded but build-critical source files are included

## Roadmap Ideas

- Stronger admin workflows for store and user provisioning
- Better observability around AI latency, retrieval quality, and model routing
- Richer analytics slices and drill-downs
- Automated CI validation for backend, frontend, and deployment health
- Secret Manager integration for all production-sensitive settings

## ScreenShots
<img width="610" height="1132" alt="303cd486-aab1-416f-b109-ce66db5dc725" src="https://github.com/user-attachments/assets/1610b04b-5c57-4296-9df4-4abe5431d63d" />
<img width="1600" height="845" alt="d3ef7fad-1313-41b9-80b8-4702bc0a6308" src="https://github.com/user-attachments/assets/2353032c-267e-4fee-9496-af80c2755b29" />
<img width="1600" height="845" alt="a891f3e7-f78a-4c8a-bbfe-1c74be5453ab" src="https://github.com/user-attachments/assets/3ea869a9-cffb-441d-b5c0-e0114c03ba5a" />
<img width="1600" height="849" alt="558d80fb-30d4-4ecb-8165-e9c24b2a7da1" src="https://github.com/user-attachments/assets/f98e613a-7eb7-4afd-9036-e8ca0fedc719" />
<img width="1600" height="845" alt="ff02c2a1-912c-416e-a1d3-cc61081a5eb8" src="https://github.com/user-attachments/assets/7659b5e2-c911-4cf8-b1ed-19c4dd5c2922" />

## Demo Video

[Watch the Demo Video](https://youtu.be/e9IBF1SQq4Q?si=X39mQI471dhAx7fV)

**Main repo**: [https://github.com/ShreyData/Stocklytics-AI](https://github.com/ShreyData/Stocklytics-AI)
