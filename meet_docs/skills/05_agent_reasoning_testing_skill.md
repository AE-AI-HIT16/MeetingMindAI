# Agent Reasoning & Testing Skill — MeetASR

This document outlines behavioral guidelines, reasoning loops, and testing standards for AI Agents operating on the MeetASR codebase. It ensures Agents work autonomously, verify changes, and adhere to system specifications.

---

## 1. Research and Analysis Guidelines

Before editing any codebase files or altering the system architecture, the Agent **must** conduct thorough research:

### 1.1 Codebase Inspection
- **Rule:** Do not guess the existence of classes, function parameters, or paths.
- **Action:** Utilize file listing (`list_dir`), search tools (`grep_search`), or file viewer tools (`view_file`) to inspect dependencies, verify active schemas, and understand component interaction before coding.

### 1.2 Documentation Alignment
- Align all planned changes with system constraints and design specifications in the `meet_docs/` directory.
- If a conflict arises between user requests and core architecture design, document the issues and construct an `implementation_plan.md` to align with the user before writing code.

---

## 2. The Agent Reasoning Loop

The Agent must navigate tasks using a closed-loop reasoning process: **Plan -> Act -> Observe -> Correct**.

```
       ┌─────────────────┐
       │     1. Plan     │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │     2. Act      │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │   3. Observe    │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │   4. Correct    │◀─── (If errors occur)
       └─────────────────┘
```

1. **Plan:** Deconstruct complex tasks into small, actionable items and track progress using `task.md`.
2. **Act:** Execute surgical edits, modifying only target lines and leaving unrelated code blocks untouched.
3. **Observe:** Run unit tests, linters, or check log outputs to evaluate the impact of changes.
4. **Correct:** Analyze tracebacks or compiler errors, modify code segments, and re-run tests until all verification checks pass. Never ignore warning blocks.

---

## 3. Automated Testing Standards

Features are only considered completed ("Definition of Done") once accompanied by automated unit or integration tests verifying their behavior.

### 3.1 Tests Directory Layout
All test cases must reside in the root `tests/` directory, and test filenames must use the prefix `test_` (e.g., `test_audio_utils.py`) to be auto-discovered by pytest.

### 3.2 Testing Policies
1. **Isolation:** Unit tests must target a single, isolated function or logical path. Do not mix multiple component assertions in a single test block.
2. **External Mocking:** Stub out network connections, external API dependencies, and heavy computing tasks (such as executing raw neural network inferences on CPU) to guarantee test execution times remain under 2 seconds.
3. **Happy and Edge Cases:** Cover both standard successful execution paths (happy paths) and typical error conditions (e.g., bad inputs, connection failures).

---

## 4. Communication and Reporting Guidelines

- **Clarify Ambiguities:** If requirements are unclear, do not make assumptions. Provide a multi-choice questionnaire or outline proposed actions in an `implementation_plan.md` for review.
- **Progress Tracking:** Update `task.md` progress markers frequently (`[x]` for done, `[/]` for active tasks) so the user can easily monitor the status.
- **Build Walkthroughs:** On task completion, update `walkthrough.md` mapping out changes, testing steps, and execution guidelines.
- **Self-Correction:** If automated tests fail during verification, the Agent must analyze and resolve the issue independently before ending its turn.
