# AI with Claude — 매일 AI 공부 (Daily AI Study)

> 초보에서 심화까지, 매일 한 걸음씩.
> From beginner to advanced, one step every day.

이 폴더는 **IR(정보 검색) · NLP · RAG · LLM** 을 중심으로, 초보 개념부터 최신 논문을 이해하고 실제로 활용하는 수준까지 단계별로 쌓아가는 학습 자료 모음입니다. 매일 한 편씩(`Day-XXX.md`) 자동 생성됩니다.

This folder is a step-by-step study track centered on **Information Retrieval (IR) · NLP · RAG · LLM**, building from beginner concepts up to reading and applying recent papers. One lesson (`Day-XXX.md`) is generated automatically each day.

---

## 🎯 학습 목표 (Goals)

이 트랙을 끝마치면 다음을 할 수 있게 됩니다:

1. **Python을 AI/데이터 작업에 능숙하게** 사용한다. (NumPy, PyTorch, Hugging Face 등)
2. **IR · NLP 기술의 기본 원리와 활용 경험**을 갖춘다. (토큰화, 임베딩, 색인, 랭킹, 평가)
3. **IR · RAG · LLM 관련 기술의 충분한 배경 지식**을 보유하고, **최신 논문을 이해하며 실제 시스템에 적용**할 수 있다.
4. **논문을 스스로 읽고 소화한다** — 3-pass 독해법으로 핵심을 파악하고, 한 장으로 요약하며, 핵심 아이디어를 Python으로 **부분 재현**할 수 있다.
5. **모델을 처음부터 직접 만들어 본다 (build from scratch)** — NumPy로 신경망·역전파부터 미니 Transformer·작은 언어모델·임베딩 인코더·RAG 파이프라인까지 **내 손으로 구현**해 동작시킨다. *(Build core models from scratch — from a NumPy neural net up to a mini-Transformer, a small LM, an embedding encoder, and a RAG pipeline.)*
6. **LLM을 효율적으로 학습·압축·확장하는 시스템을 다룬다** — 저정밀 학습(FP8/MXFP4), 양자화(QAT·W4A8/W4A16), 지식 증류, 커널(Triton/CUDA), 분산 학습(TP/PP/EP·FSDP·Megatron·DeepSpeed), 데이터 품질·페타바이트 처리, 대규모 클러스터 최적화를 **원리·논문·설계부터 실전 프레임워크까지**. *(Train, compress & scale LLMs efficiently — low-precision, quantization, distillation, GPU kernels, distributed training, data & cluster optimization.)*

By the end of this track you will be able to: use Python fluently for AI work, understand and apply core IR/NLP techniques, read papers with the 3-pass method (summarize + partially reproduce them), and implement recent IR/RAG/LLM papers.

---

## 🗺️ 전체 로드맵 (Curriculum Roadmap)

각 단계(Phase)는 앞 단계 위에 쌓입니다. 일 수는 대략치이며, 주제별로 조정됩니다.

| Phase | 주제 (Topic) | 범위 (Days) | 핵심 (Core) |
|-------|-------------|-------------|-------------|
| **0** | 준비 & 큰 그림 / Foundations & Big Picture | Day 1–5 | AI·ML·DL·NLP·IR·LLM 개념 지도, Python 환경(uv), NumPy, 확률·통계, ML 기본 |
| **1** | NLP 기초 / NLP Basics | Day 6–15 | 텍스트 전처리, 토큰화, BoW·TF-IDF, 단어 임베딩(Word2Vec/GloVe), 텍스트 분류 · 📄 **논문 읽기 입문**(논문 구조·3-pass 독해법) |
| **2** | 정보 검색 기초 / IR Fundamentals | Day 16–25 | 검색 시스템 구조, 역색인, BM25, 랭킹, 평가지표(MAP·nDCG·MRR), 미니 검색엔진 · 📄 정독: BM25/Probabilistic IR |
| **3** | 딥러닝 & 트랜스포머 / DL & Transformers | Day 26–40 | 신경망·역전파, PyTorch, RNN/LSTM, Attention, Transformer, BERT·GPT · 📄 정독: **Attention Is All You Need**/BERT/GPT |
| **4** | LLM 깊이 이해 / Understanding LLMs | Day 41–55 | 사전학습·파인튜닝, BPE 토크나이저, 프롬프팅·In-context learning, 디코딩, 정렬(RLHF/DPO) · 📄 정독: InstructGPT/LoRA/Scaling Laws |
| **5** | 임베딩 & 밀집 검색 / Embeddings & Dense Retrieval | Day 56–70 | 문장 임베딩(SBERT), DPR, 벡터 DB·ANN(FAISS/HNSW), 하이브리드 검색, 리랭킹 · 📄 정독: DPR/Sentence-BERT/ColBERT |
| **6** | RAG / Retrieval-Augmented Generation | Day 71–90 | RAG 아키텍처, 청킹, 검색+생성 통합, 평가(RAGAS), 고급 RAG(HyDE·multi-hop·rerank), Agentic RAG · 📄 정독: RAG(Lewis 2020)/FiD/Self-RAG |
| **7** | 📄 논문 읽기 & 재현 / Reading & Reproducing Papers | Day 91–100 | 비판적 독해, notation·수식 읽기, 실험·ablation 분석, 베이스라인·재현성 점검, 논문→코드 매핑, **부분 재현**, 요약 노트 템플릿, 동향 추적(arXiv·학회), Claude로 논문 읽기 |
| **8** | 최신 논문 리뷰 & 실전 프로젝트 / Recent Papers & Capstone | Day 101+ | 최근 1–2년 IR/RAG/LLM 논문 로테이션 리뷰, 에이전트·도구 사용, 종합 실전 프로젝트 |
| **9** | ⚙️ LLM 학습·효율화 시스템 / Training Systems & Efficiency | Day 110+ | 저정밀 학습(FP8·MXFP4·loss scaling), 양자화(QAT·W4A8/W4A16·STE·SmoothQuant/QuaRot), 지식 증류, LLM 커널(Triton/CUDA), 분산 학습(DP/TP/PP/CP/EP·ZeRO·FSDP·Megatron·DeepSpeed), 데이터 품질·페타바이트 처리, 클러스터 최적화(communication overlap·activation recomputation·memory-efficient optimizer) · 📄 정독: FlashAttention/GPTQ/AWQ/SmoothQuant/ZeRO/Megatron-LM |

> 로드맵은 학습 진행에 따라 보강·조정됩니다. The roadmap is refined as we progress.

### 📄 논문 읽기 트랙 (Paper-Reading Thread)

논문 읽기는 Phase 7에 몰아넣지 않고 **트랙 전체에 엮습니다** — 일찍 시작해, 배운 직후 그 개념의 원전을 읽습니다.

1. **방법론 미니레슨** — Phase 1에서 *논문 읽기 입문*(논문의 해부학, 3-pass 독해법, 핵심 5질문)을 먼저 익힙니다.
2. **배운 직후 정독** — 각 Phase에서 그 주제의 **landmark 논문**을 정독합니다 (위 표의 📄 항목).
3. **정기 리듬** — 약 **10편마다 1편**을 "📄 논문 정독(Paper Deep-Dive)" 회차로 고정합니다.
4. **정독 깊이** — *이해 + 부분 재현*: 3-pass 중 2단계(이해)까지 깊게 가고, 핵심 아이디어를 Python으로 직접 **부분 재현**합니다.
5. **요약 노트** — 매 정독마다 한 장짜리 요약(문제·아이디어·검증·한계)을 남깁니다.

> Paper reading is woven throughout, not deferred. Start early (Phase 1 methodology), read each topic's landmark paper right after learning it, run a "Paper Deep-Dive" every ~10 lessons, and aim for *understand + partial reproduction*.

### 🛠️ 직접 만들기 트랙 (Build-From-Scratch Thread)

"읽고 이해"를 넘어 **"내 손으로 구현"** 하는 트랙입니다. **기존 Phase 진행과 로드맵 표는 그대로 두고**, 각 Phase가 끝나는 지점에 **"🛠️ 빌드 마일스톤(Build Milestone)"** 한 편을 *추가로* 끼워 그 단계의 핵심을 **처음부터(from scratch)** 직접 만들어 동작시킵니다. (기존 커리큘럼을 바꾸지 않고 더하는 병행 트랙)

| 마일스톤 | 시점 (해당 Phase 마무리) | 내 손으로 만드는 것 |
|---------|------------------------|---------------------|
| **B0** | Phase 0 후 | NumPy만으로 **선형회귀 + 경사하강법(gradient descent)** 직접 구현 (sklearn 없이) |
| **B1** | Phase 1 후 | **TF-IDF + 로지스틱 회귀 텍스트 분류기** from scratch (학습 루프 직접) |
| **B2** | Phase 2 후 | **BM25 검색엔진**과 평가지표(nDCG)를 직접 구현 |
| **B3** ⭐ | Phase 3 후 | NumPy로 **MLP + 역전파(backprop)** 직접 구현 → PyTorch로 **미니 Transformer(Self-Attention 손코딩)** 구축 |
| **B4** | Phase 4 후 | **BPE 토크나이저** 구현 + **char-level 미니 GPT** 학습(nanoGPT 스타일) + LoRA 파인튜닝 |
| **B5** | Phase 5 후 | **대조학습으로 임베딩 인코더** 학습 + (FAISS 없이) brute-force 벡터검색 직접 구현 |
| **B6** | Phase 6 후 | **미니 RAG 파이프라인 end-to-end** 직접 조립 (청킹→임베딩→검색→생성→평가) |
| **B7** | Phase 7–8 | 논문 1편의 핵심을 **from scratch 재현** → 종합 캡스톤으로 연결 |

원칙 (Principles):
1. **동작이 우선** — 외부 고수준 라이브러리를 최소화하고, 작동하는 **최소 구현(working minimal implementation)** 을 목표로 한다. (규모가 크면 2편으로 나눔)
2. **이전 레슨 재사용** — 그 Phase에서 배운 개념·코드를 그대로 끌어와 조립한다.
3. **표시** — 빌드 마일스톤 편은 제목과 진도표에 **🛠️** 로 표시한다.
4. **연결** — 옵시디언 `[[ ]]` 링크로 관련 개념 레슨과 묶는다.

> A parallel **build-from-scratch** thread, *added on top of* (not replacing) the existing phases. After each phase, an extra "🛠️ Build Milestone" lesson has you implement that phase's core idea by hand — a NumPy neural net, a mini-Transformer, a tiny GPT, an embedding encoder, a full mini-RAG — aiming for a *working minimal implementation*.

### ⚙️ LLM 학습·효율화 시스템 트랙 (Training Systems & Efficiency Thread)

채용 요건급 **LLM 학습 인프라·효율화** 역량을 다루는 고급 트랙입니다. 메인 단계는 **Phase 9**, 동시에 관련 Phase 지점(정밀도·커널→Phase 3·4)에서 **⚙️ 표기**로 당겨 다룹니다. **선행: Phase 3(Transformer·PyTorch)·Phase 4(LLM).**

| 모듈 | 다루는 것 | 요건 |
|------|----------|------|
| **S1 수치 정밀도·혼합정밀** | FP32/BF16/FP16 → **FP8(E4M3/E5M2)·MXFP4**, loss scaling, **tensor-wise vs block-wise(microscaling) 스케일링**, 학습 수치 안정성 | ① |
| **S2 양자화 (QAT/저비트)** | PTQ vs QAT, **W4A16·W4A8**, STE 기반 학습, GPTQ/AWQ, **SmoothQuant·QuaRot(rotation/smoothing)**, PTQ 대비 품질 회복 | ② |
| **S3 지식 증류 (KD)** | logit/feature-level, **on-policy/sequence-level KD**, teacher–student 파이프라인 설계 | ③ |
| **S4 LLM 커널** | GPU 실행 모델, **Triton custom kernel**, fused·FlashAttention, CUDA 입문 | ④ |
| **S5 분산 학습** | **DP/TP/PP/CP/EP**, ZeRO, **FSDP**, Megatron-LM/DeepSpeed 구조·기여 흐름 | ⑤ |
| **S6 학습 데이터** | **페타바이트** 텍스트 수집·dedup·필터링·품질 평가, 분산 처리 | ⑥ |
| **S7 클러스터 최적화** | **communication overlap, activation recomputation, memory-efficient optimizer(8-bit Adam·ZeRO)**, 프로파일링 | ⑦ |

원칙 (Principles):
1. **원리·논문·설계 + 실전·프레임워크, 둘 다.** 각 모듈은 (a) 원리·수치/수식 → (b) 핵심 논문 정독(📄) → (c) 소규모 재현(🛠️, 단일 GPU/CPU·에뮬레이션) → (d) 실제 프레임워크 코드리딩·구성(Megatron-LM/DeepSpeed/FSDP) 순으로 깊게.
2. **하드웨어 현실**: FP8/MXFP4·멀티노드는 장비가 필요 → 소규모는 직접 구현/에뮬레이션으로, 대규모는 설계·코드리딩·설정으로 익힌다.
3. **📄/🛠️ 트랙과 연동** — FlashAttention·GPTQ·AWQ·SmoothQuant·ZeRO·Megatron 논문 정독, STE 양자화·Triton 커널 직접 빌드.

> An advanced **LLM training-systems & efficiency** thread (job-level skills): low-precision (FP8/MXFP4), quantization (QAT, W4A8/W4A16), distillation, Triton/CUDA kernels, parallelism (TP/PP/EP, FSDP/Megatron/DeepSpeed), petabyte data, cluster optimization. Each module goes principle → paper → small-scale build → real-framework code. Main home is **Phase 9**, pulled earlier via ⚙️ tags where relevant.

---

## 📚 진도표 (Progress Index)

| Day | 날짜 (Date) | 제목 (Title) | Phase | 상태 |
|-----|------------|-------------|-------|------|
| [001](./Day-001/Day-001.md) | 2026-06-08 | AI의 큰 그림 — ML·DL·NLP·IR·LLM은 어떻게 연결되는가 / The Big Picture | 0 | ✅ |
| [002](./Day-002/Day-002.md) | 2026-06-08 | AI 공부를 위한 Python 환경 만들기 (with uv) / Setting Up Your Python Environment | 0 | ✅ |
| [003](./Day-003/Day-003.md) | 2026-06-09 | NumPy 본격 입문 — 배열·인덱싱·브로드캐스팅 / NumPy Deep Dive: Arrays, Indexing, Broadcasting | 0 | ✅ |
| [004](./Day-004/Day-004.md) | 2026-06-09 | 확률·통계 최소 지식 for AI — 평균·분산·정규분포, 거리 vs 유사도 / Just-Enough Statistics: Distance vs Similarity | 0 | ✅ |
| [005](./Day-005/Day-005.md) | 2026-06-09 | 머신러닝의 기본기 — 지도/비지도, 데이터 분할, 과적합·일반화, 정밀도·재현율 / ML Foundations: Overfitting & Precision-Recall | 0 | ✅ |
| [006](./Day-006/Day-006.md) | 2026-06-12 | 텍스트 전처리와 토큰화 — 토큰·정규화·불용어, 한국어/서브워드 / Text Preprocessing & Tokenization | 1 | ✅ |
| [007](./Day-007/Day-007.md) | 2026-06-15 | 단어를 세어 문서를 벡터로 — Bag-of-Words & TF-IDF / Bag-of-Words & TF-IDF | 1 | ✅ |
| [008](./Day-008/Day-008.md) | 2026-06-15 | 의미를 담은 벡터로 — 분포 가설과 단어 임베딩 입문 / Distributional Hypothesis & Word Embeddings | 1 | ✅ |
| [009](./Day-009/Day-009.md) | 2026-06-16 | Word2Vec 은 어떻게 학습되나 — Skip-gram & 네거티브 샘플링 / Training Skip-gram with Negative Sampling | 1 | ✅ |
| [010](./Day-010/Day-010.md) | 2026-06-17 | 텍스트 분류 첫걸음 — 임베딩으로 감정 분석 베이스라인 & Phase 1 정리 / Sentiment Baseline & Phase 1 Wrap-up | 1 | ✅ |
| [011](./Day-011/Day-011.md) | 2026-06-19 | 언어모델의 첫 직관 — n-gram 으로 "다음 단어 맞히기" / n-gram Language Models, Smoothing & Perplexity | 1 | ✅ |
| [012](./Day-012/Day-012.md) | 2026-06-19 | 📄 논문 읽기 입문 — 논문의 해부학·3-pass 독해법·핵심 5질문 / Paper-Reading 101 | 1 | ✅ |
| [013](./Day-013/Day-013.md) | 2026-06-23 | 확률로 분류하기 — 나이브 베이즈(Naïve Bayes) 텍스트 분류기 / Naïve Bayes Text Classification | 1 | ✅ |
| [014](./Day-014/Day-014.md) | 2026-06-24 | 🛠️ B1 빌드 — TF-IDF + 로지스틱 회귀 분류기 from scratch (NumPy) / B1 Build: TF-IDF + Logistic Regression from Scratch | 1 | ✅ 🛠️ |
| [015](./Day-015/Day-015.md) | 2026-06-25 | Phase 1 총정리 & 정보 검색(IR)으로 가는 다리 / Phase 1 Recap & Bridge to Information Retrieval | 1 | ✅ |
| [016](./Day-016/Day-016.md) | 2026-06-26 | 검색 시스템의 뼈대 & 역색인(Inverted Index) 직접 만들기 / Search System Anatomy & Building an Inverted Index | 2 | ✅ |
| [017](./Day-017/Day-017.md) | 2026-06-29 | 랭킹의 핵심, BM25 — tf 포화·문서길이 정규화로 점수 매기기 / Ranking with BM25: TF Saturation & Length Normalization | 2 | ✅ |
| [018](./Day-018/Day-018.md) | 2026-06-30 | 검색을 평가하다 — IR 평가지표(Precision@k·MAP·MRR·nDCG) / IR Evaluation Metrics | 2 | ✅ |
| [019](./Day-019/Day-019.md) | 2026-07-02 | 정답지는 어디서 오나 — 테스트 컬렉션·qrels·pooling·판단자 일치도(Cohen's κ) / Where Judgments Come From: Test Collections, Pooling & Annotator Agreement | 2 | ✅ |
| [020](./Day-020/Day-020.md) | 2026-07-06 | 검색 파이프라인으로 잇기 — 색인·랭킹·평가 통합 & 평가-주도 개발(EDD) / Wiring the Retrieval Pipeline & Evaluation-Driven Development | 2 | ✅ |
| [021](./Day-021/Day-021.md) | 2026-07-07 | 🛠️ B2 빌드 — BM25 검색엔진 + nDCG·MAP 평가, from scratch / B2 Build: BM25 Engine + Evaluation Harness from Scratch | 2 | ✅ 🛠️ |
| [022](./Day-022/Day-022.md) | 2026-07-08 | 📄 BM25의 원전 — 확률적 적합성 프레임워크(PRP·BIM·RSJ·2-Poisson) / Paper Deep-Dive: The Probabilistic Relevance Framework (BM25 and Beyond) | 2 | ✅ 📄 |
| [023](./Day-023/Day-023.md) | 2026-07-09 | 질의 확장과 의사 적합 피드백 — 상위 문서로 질의를 넓히기(Rocchio·PRF) / Query Expansion & Pseudo-Relevance Feedback | 2 | ✅ |
| [024](./Day-024/Day-024.md) | 2026-07-10 | 벡터 공간 모델(VSM)과 코사인 유사도 랭킹 — 문서=벡터, 각도로 매기는 순위 / The Vector Space Model & Cosine Similarity | 2 | ✅ |
| [025](./Day-025/Day-025.md) | 2026-07-13 | Phase 2 총정리 & 딥러닝/트랜스포머로 가는 다리 — 어휘 불일치의 벽 / Phase 2 Recap & Bridge to Deep Learning | 2 | ✅ |
| [026](./Day-026/Day-026.md) | 2026-07-14 | 신경망의 첫 원리 — 퍼셉트론에서 다층 신경망(MLP)까지, 순전파와 비선형성 / Neural Networks from First Principles: Perceptron → MLP & the Forward Pass | 3 | ✅ |
| [027](./Day-027/Day-027.md) | 2026-07-15 | 역전파(Backpropagation) — 손실·연쇄법칙·경사하강으로 신경망이 스스로 배우는 법 / How Neural Networks Learn: Loss, the Chain Rule & Gradient Descent | 3 | ✅ |
| [028](./Day-028/Day-028.md) | 2026-07-16 | PyTorch 입문 — 텐서와 autograd 로 backprop 자동화하기 / PyTorch 101: Tensors & Autograd | 3 | ✅ |
| [029](./Day-029/Day-029.md) | 2026-07-20 | nn.Module 과 torch.optim — 모델과 학습을 깔끔하게 캡슐화하기 / Encapsulating Models & Optimizers | 3 | ✅ |
| [030](./Day-030/Day-030.md) | 2026-07-20 | RNN 입문 — 순서가 있는 데이터(시퀀스)를 다루는 신경망 / Recurrent Neural Networks | 3 | ✅ |
| [031](./Day-031/Day-031.md) | 2026-07-21 | LSTM 과 GRU — 게이트로 장기 기억을 지키기 / Long Short-Term Memory & GRU | 3 | ✅ |
| [032](./Day-032/Day-032.md) | 2026-07-22 | Attention 메커니즘 — 순환의 병목을 넘어 '모든 위치를 한 번에' 보기 / The Attention Mechanism | 3 | ✅ |

> 새 자료가 생성될 때마다 이 표에 한 줄이 추가됩니다. A new row is added here whenever a lesson is generated.

---

## 🧭 이 자료 사용법 (How to use)

- 매일 `Day-XXX.md` 한 편을 읽습니다. 위에서 아래로, 코드 블록은 직접 실행해 보세요.
- 각 자료는 **개념(Concept) → 직관·비유(Intuition) → 실습(Hands-on) → 생각거리(Reflection) → 더 읽을거리(Further reading)** 순서로 구성됩니다.
- 모르는 용어가 나오면 그날 자료의 **용어집(Glossary)** 을 먼저 확인하세요.
- 한국어가 본문, 핵심 기술 용어는 `영어(English)` 로 병기합니다.

Each lesson follows: Concept → Intuition → Hands-on → Reflection → Further reading. Korean is the main text; key technical terms are given in English.

---

## ⚙️ 자동 생성 (Automation)

이 자료는 매일 자동으로 한 편씩 생성되도록 `/schedule` 로 예약되어 있습니다. 생성 작업은:

1. 이 폴더에서 가장 최근 `Day-XXX.md` 를 찾고,
2. 위 로드맵을 따라 다음 편(`Day-(XXX+1).md`)을 작성하며 — **약 10편마다 한 편은 📄 논문 정독(Paper Deep-Dive) 회차**로, 각 Phase의 landmark 논문을 *이해 + 부분 재현* 깊이로 다룹니다,
3. 위 **진도표** 에 새 줄을 추가합니다.

A scheduled task generates the next lesson daily: it finds the latest `Day-XXX.md`, writes the next one following this roadmap (with a 📄 Paper Deep-Dive every ~10 lessons), and appends a row to the progress index above.
