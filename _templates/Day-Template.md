<!--
==================================================================
AI with Claude — 일일 레슨 템플릿 (Daily Lesson Template)
이 파일은 매일 자동 생성되는 Day-NNN 노트의 표준 형식이다.
{{...}} 부분을 실제 내용으로 채운다. 주석(<!-- -->)은 결과물에서 제거.

설계 원칙
- 문서 관계는 'frontmatter 속성'으로 정의한다(prev/next/related/up/phase/type).
  → Obsidian 속성 패널·그래프·Dataview가 이 속성을 읽어 관계를 자동 구성.
- 시각적 정리는 'Obsidian 콜아웃'으로 한다(> [!summary] 등).
- 본문에서 다른 편을 언급하면 [[Day-NNN]] 위키링크로 쓴다.
- type 값: lesson(일반) | paper(📄 논문 정독) | build(🛠️ 빌드 마일스톤)
==================================================================
-->
---
title: "Day {{NNN}} — {{한국어 제목}}"
day: {{N}}
date: {{YYYY-MM-DD}}
phase: {{PHASE_NUM}}
phase_name: "{{Phase 이름 예: NLP 기초}}"
type: lesson
status: done
est_min: 35
up: "[[Phase-{{PHASE_NUM}}]]"
prev: "[[Day-{{이전번호}}]]"
next: "[[Day-{{다음번호}}]]"
related:
  - "[[Day-{{관련1}}]]"
  - "[[Day-{{관련2}}]]"
tags:
  - ai-study
  - phase/{{PHASE_NUM}}
  - type/lesson
topics:
  - {{핵심 토픽1}}
  - {{핵심 토픽2}}
---

> [!abstract] 🎯 오늘의 목표
> {{한두 문장으로 오늘 무엇을·왜 배우는지. 핵심 개념은 [[Day-NNN]] 으로 연결.}}

> [!info]- 📦 준비물 (uv)
> [[Day-002]] 의 uv 프로젝트. `uv add {{필요한 패키지}}` 로 추가. (활성화 불필요 — 실행은 `uv run`)

## 0. 들어가며 (Motivation)
{{왜 이걸 배우나. 앞 편([[Day-NNN]])과의 연결, 큰 그림 속 위치.}}

## 1. 핵심 개념 (Core Concepts)
{{소제목(### 1.1 …)으로 나눠 개념을 충실히. 표·수식 적극 활용.}}

## 2. 직관 / 비유 (Intuition)
> [!tip] 비유
> {{기억에 남는 비유 1개. 끝에 영어 한 줄 요약(>) 권장.}}

## 3. 한 장의 그림 (One Picture)
```text
{{핵심을 한눈에 보여주는 ASCII 다이어그램}}
```

## 4. 실습 (Hands-on)
> 환경: **Windows + PowerShell**, 패키지는 **uv** 로 관리.

{{실행 가능한 Python/PowerShell 코드. 파일로 저장할 코드는 같은 폴더에 .py 로도 저장하고
`uv run python {{파일}}.py` 실행을 안내. 새 패키지는 `uv add ...`.}}

## 5. 용어집 (Glossary)
| 용어 | English | 한 줄 뜻 |
|------|---------|---------|
| {{용어}} | {{term}} | {{뜻}} |

## 6. 생각거리 (Reflection)
> [!question] 스스로 답해 보기
> 1. {{질문}}
> 2. {{질문}}
> 3. {{질문}}

## 7. 오늘의 한 줄 요약 (Takeaway)
> [!summary]
> {{한 문장 핵심 요약}}

## 8. 더 읽을거리 (Further Reading)
> [!note]- 펼쳐 보기
> - 📄/📘/🔧 {{자료}} — {{링크}}

## 🔗 연관 노트 (Related)
> [!info]- 이 노트와 이어지는 편 (관계는 frontmatter에 정의됨)
> - [[Day-{{관련1}}]] — {{이유}}
> - [[Day-{{관련2}}]] — {{이유}}

---
> [!example] ✅ 다음 편 — [[Day-{{다음번호}}]]
> {{다음 편에서 무엇을 다루는지 한 문단.}}
