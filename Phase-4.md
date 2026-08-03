---
title: "Phase 4 — LLM 깊이 이해"
type: phase-moc
phase: 4
up: "[[AI-Study-Home]]"
tags:
  - ai-study
  - moc
  - phase/4
---

# Phase 4 — LLM 깊이 이해 (Understanding LLMs)

> [!abstract] 이 단계에서
> Phase 3 에서 부품(Transformer)을 손으로 조립했다면([[Day-037]]), Phase 4 는 그 부품이 **실제 언어모델**이 되는 과정을 다룬다. BPE 토크나이저, 사전학습·파인튜닝, 프롬프팅·In-context learning, 디코딩 전략(온도·top-k·top-p), 정렬(RLHF/DPO). 그리고 📄 정독: **Scaling Laws** / **InstructGPT** / **LoRA**, 🛠️ **B4 빌드**(BPE + char-level 미니 GPT + LoRA). **Day 41–55.**

⬆️ [[AI-Study-Home]] · ⬅️ [[Phase-3]] · ➡️ [[Phase-5]]

## 이 단계의 레슨 (자동 생성)

```dataview
TABLE WITHOUT ID
  ("[[" + file.name + "|Day " + day + "]]") AS "노트",
  title AS "제목",
  date AS "날짜",
  type AS "유형",
  status AS "상태"
FROM "ai-with-claude"
WHERE phase = 4 AND day != null
SORT day ASC
```
