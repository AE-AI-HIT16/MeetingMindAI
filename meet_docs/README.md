# MeetASR — Meeting Speech Recognition + LLM Summarization

> **Objective:** Re-write FunASR from scratch, simplify the architecture, and integrate LLMs to summarize meeting content, extract key topics, action items, and sentiment.

---

## Directory Structure

```
meet_docs/
├── README.md                    ← This file
├── docs/
│   ├── 01_goals.md              ← Goals & constraints
│   ├── 02_architecture.md       ← Overall architecture
│   ├── 03_funasr_analysis.md    ← Original FunASR analysis
│   ├── 04_llm_integration.md    ← LLM integration design
│   ├── 05_roadmap.md            ← Development roadmap
│   ├── 06_database.md           ← Database design
│   ├── 07_data_pipeline.md      ← Data processing & loaders
│   ├── 08_team_tasks.md         ← Team task breakdown (Phase 1)
│   ├── 09_realtime_phase2.md    ← [superseded] old realtime-meeting plan
│   ├── 10_realtime_team_tasks.md ← [superseded] old realtime task breakdown
│   ├── 11_phase2_notebooklm_plan.md ← Phase 2: NotebookLM-style plan + team tasks
│   └── 12_frontend_nextjs_plan.md   ← Phase 2: Next.js/TypeScript frontend plan
├── skills/
│   ├── 01_data_processing_skill.md
│   ├── 02_backend_system_api_skill.md
│   ├── 03_frontend_skill.md
│   ├── 04_ai_llm_skill.md
│   ├── 05_agent_reasoning_testing_skill.md
│   └── 06_realtime_streaming_skill.md ← Phase 2: Learning guide per role
├── specs/
│   ├── api_spec.md              ← API specification
│   ├── data_formats.md          ← Input/Output data formats
│   └── model_registry.md        ← Registry pattern specification
└── constraints/
    ├── technical.md             ← Technical constraints
    └── coding_style.md          ← Coding conventions
```

## Quick Start

1. Review the [Goals & Constraints](docs/01_goals.md).
2. Study the [Overall Architecture](docs/02_architecture.md).
3. Understand the [Original FunASR Analysis](docs/03_funasr_analysis.md).
4. Review the [LLM Integration Design](docs/04_llm_integration.md).
5. Follow the development schedule in the [Roadmap](docs/05_roadmap.md).

## Phase 2 — "NotebookLM for Audio/Video" (current)

Phase 1 (offline pipeline: upload → transcript + summary) is complete.
Phase 2: user uploads **video/audio**, the system generates a structured
**document in realtime** (content appears as the media is processed), then the
user chooses **summary** or **full text** and exports **PDF/DOCX**. All media
and documents are stored in a web library. The frontend is rebuilt with
**Next.js + TypeScript**.

1. Read the plan, architecture & team tasks: [11_phase2_notebooklm_plan.md](docs/11_phase2_notebooklm_plan.md)
2. Frontend rebuild plan: [12_frontend_nextjs_plan.md](docs/12_frontend_nextjs_plan.md)
3. Background knowledge: [skills/06_realtime_streaming_skill.md](skills/06_realtime_streaming_skill.md) (sections A, B.3, B.4 still apply)
