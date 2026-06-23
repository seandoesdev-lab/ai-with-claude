---
title: "🏠 AI with Claude — 홈 (Home / MOC)"
type: home
tags:
  - ai-study
  - moc
---

# 🏠 AI with Claude — 매일 AI 공부

> [!abstract] 이 보관함은?
> **IR · NLP · RAG · LLM** 을 초보→심화로, 매일 한 편씩 쌓는 학습 트랙입니다.
> 문서 관계는 각 노트의 **frontmatter 속성**(`phase`/`type`/`prev`/`next`/`related`/`up`)으로 정의되고, 아래 표는 **Dataview**가 자동 생성합니다.

> [!warning] Dataview 플러그인 필요
> 아래 표가 코드로 보이면, Obsidian → 설정 → 커뮤니티 플러그인에서 **Dataview**를 설치·활성화하세요.

## 🗺️ Phase 지도 (MOC)

- [[Phase-0]] · 준비 & 큰 그림 (Day 1–5)
- [[Phase-1]] · NLP 기초 (Day 6–15)
- [[Phase-2]] · 정보 검색 기초 (Day 16–25)
- [[Phase-3]] · 딥러닝 & 트랜스포머 (Day 26–40)
- [[Phase-4]] · LLM 깊이 이해 (Day 41–55)
- [[Phase-5]] · 임베딩 & 밀집 검색 (Day 56–70)
- [[Phase-6]] · RAG (Day 71–90)
- [[Phase-7]] · 📄 논문 읽기 & 재현 (Day 91–100)
- [[Phase-8]] · 최신 논문 & 실전 (Day 101+)

## 📚 전체 진도 (자동 생성)

```dataview
TABLE WITHOUT ID
  ("[[" + file.name + "|Day " + day + "]]") AS "노트",
  date AS "날짜",
  phase AS "P",
  type AS "유형",
  status AS "상태"
FROM "ai-with-claude"
WHERE day != null
SORT day ASC
```

## 🔭 가장 최근 / 다음

```dataview
TABLE WITHOUT ID ("[[" + file.name + "|Day " + day + "]]") AS "최근 5편", date AS "날짜", phase AS "P"
FROM "ai-with-claude"
WHERE day != null
SORT day DESC
LIMIT 5
```

## 🏷️ 유형별 보기

```dataview
TABLE WITHOUT ID ("[[" + file.name + "|Day " + day + "]]") AS "노트", date AS "날짜"
FROM "ai-with-claude"
WHERE type = "paper" OR type = "build"
SORT day ASC
```

> [!tip] 그래프 뷰
> 좌측 그래프 아이콘 → `tag:#ai-study` 로 필터하면 학습 노트들의 관계망만 볼 수 있습니다.
