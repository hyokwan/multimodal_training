# 수정 로그

## 2026-08-04 (추가 4)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py / .ipynb (Llama 3.2 Vision 혼합 배치 지원 — cross_attention_mask 등)

**문제:** 실제 `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`(Mllama, 비게이트 미러) 모델로
end-to-end 검증 중, 이미지 샘플과 텍스트 샘플이 섞인 배치에서 `_mergeMixedBatch()`가
`input_ids`/`attention_mask`/`token_type_ids`와 `pixel_values`만 명시적으로 처리하고 있어
Mllama가 요구하는 `aspect_ratio_ids`, `aspect_ratio_mask`, `cross_attention_mask` 텐서가
누락되어 forward pass에서 `ValueError: aspect_ratio_ids must be provided if pixel_values is
provided` 로 실패함. 순수 이미지 전용 배치(혼합 아님)에서는 이 문제가 없었음 — 혼합 배치
전용 병합 함수의 하드코딩된 키 목록이 원인.

**변경 내용:**

| 파일 | 위치 | 변경 내용 |
|------|------|-----------|
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `getCollateFn` 내 `_mergeMixedBatch` | `cross_attention_mask` 전용 처리 추가(텍스트 축 패딩 + 텍스트 전용 샘플 자리는 0으로 채움), 이미지 서브배치의 나머지 키(`pixel_values`, `aspect_ratio_ids`, `aspect_ratio_mask` 등)를 하드코딩 대신 자동으로 훑어 통과시키도록 일반화 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb` | 셀 `35954667` (`_mergeMixedBatch`) | 동일하게 수정 |
| `codeset/docs/★02. 파인튜닝_멀티모달데이터_Gemma3.md` | — | `_mergeMixedBatch` 동작 설명 갱신, 변경 이력 추가 |

**실제 검증 (B200 GPU, unsloth 4bit 미러 모델 다운로드 후 직접 실행):**
- Gemma3(`google/gemma-3-4b-it`, config-only): 회귀 없음 — 이미지 토큰 256개 전부 정상 마스킹, batch shape 동일
- Llama 3.2 Vision 이미지+텍스트 혼합 배치: 수정 전 `aspect_ratio_ids` 에러로 실패 → 수정 후 forward pass 성공 (loss=4.64)

**참고:** `meta-llama/Llama-3.2-11B-Vision-Instruct` 공식 레포는 테스트 시점 기준 `.env`의 HF_TOKEN
계정에서 게이트 미승인(401) 상태라 비게이트 커뮤니티 미러로 검증함. 승인 완료 후 공식 레포로도
동일하게 동작할 것으로 예상.

---

## 2026-08-04 (추가 3)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py / .ipynb (system 메시지 + 이미지 호환성 — Llama 3.2 Vision 대응)

**문제:** `buildMessages()`가 항상 `system`/`user`/`assistant` 3턴을 만들었는데, Llama 3.2 Vision
(Mllama)의 공식 채팅 템플릿은 이미지가 포함된 대화에서 system 역할 사용을 금지함
(`jinja2.exceptions.TemplateError: Prompting with images is incompatible with system messages.`).
Gemma3는 이 조합이 허용되어 문제없었지만, `BASE_MODEL`을 Llama 3.2 Vision으로 바꾸면 학습이
템플릿 렌더링 단계에서 바로 실패함.

**변경 내용:**

| 파일 | 위치 | 변경 내용 |
|------|------|-----------|
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `buildMessages` | `mergeSystemToUser` 파라미터 추가 — `True`+이미지 샘플이면 system 텍스트를 user 턴 앞에 합치고 별도 system 턴을 생략 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | 신규 `systemMsgSupportsImage(processor)` | 더미 이미지+system 메시지로 실제 채팅 템플릿을 1회 프로브해서 지원 여부 판별 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `getCollateFn` | 시작 시 프로브 실행해 `mergeSystemToUser` 결정, `collateFn`의 user 턴 탐색을 `messages[1]` 인덱스 고정 → `role == "user"` 탐색으로 변경(병합 시 메시지 순서가 바뀌므로) |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb` | 셀 `3fcbdab0`(`buildMessages`), `35954667`(`collate_fn`) | 동일하게 수정 |

**실제 검증 (B200 GPU, `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` 비게이트 미러로 실제 로드):**
- Gemma3: `systemMsgSupportsImage=True` → 기존과 동일하게 system 턴 유지, 이미지 토큰 256개 정상 마스킹 (회귀 없음)
- Llama 3.2 Vision: `systemMsgSupportsImage=False` 자동 감지 → system이 user 턴에 병합되어 템플릿 렌더링 성공, 이미지 토큰(128256) 정상 마스킹

---

## 2026-08-04 (추가 2)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py / .ipynb (이미지 토큰 loss 마스킹 하드코딩 제거)

**문제:** `collateFn`(노트북은 `collate_fn`) 안에 `labels[labels == 262144] = -100` 라는 Gemma3 전용
이미지 토큰 ID가 하드코딩되어 있어, `BASE_MODEL` 을 다른 모델 계열(예: Llama 3.2 Vision, Llama4)로
바꾸면 이미지 토큰이 loss에서 제외되지 않는 문제가 있었음. 에러 없이 조용히 학습 품질이 저하되는
방식이라 발견하기 어려움.

**변경 내용:**

| 파일 | 위치 | 변경 전 | 변경 후 |
|------|------|---------|---------|
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `getCollateFn` 내 `collateFn` | `labels[labels == 262144] = -100` 하드코딩 | `getattr(model.config, "image_token_id", None)` → 없으면 `getattr(model.config, "image_token_index", None)` 순으로 동적 조회 후 마스킹 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb` | 셀 `35954667` (`collate_fn`) | 동일 하드코딩 | 동일하게 동적 조회로 수정 |

**근거:** Gemma3Config는 `image_token_id`(=262144)를 직접 노출하지만, Mllama(Llama 3.2 Vision)와
Llama4Config는 `image_token_index`(각각 128256, 200092)만 제공 — 모델 계열마다 속성명이 달라
하드코딩 대신 두 속성을 순서대로 조회하도록 변경. 상세는 `codeset/docs/★02. 파인튜닝_멀티모달데이터_Gemma3.md`
참고.

---

## 2026-08-04

### SimulApp — 멀티모달 추론 서비스 전면 제거 (경량화)

**문제:** `server.js`가 기동 시 `inference_service.py`(FastAPI, 포트 9999)를 자동 spawn했는데,
이 프로세스가 `.venvg3` 환경에서 병합 모델을 HuggingFace `from_pretrained()`로 로드하면서
(torch/transformers checkpoint shard 로딩, deprecated `on_event` 경고 등) SimulApp 웹 UI가
제대로 로딩되지 않는 경우가 발생. 추론 서비스는 별도 프로젝트로 분리 개발하기로 결정.

**변경 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `SimulApp/inference_service.py` | 삭제 |
| `SimulApp/server.js` | `INFER_SERVICE_SCRIPT`/`INFER_SERVICE_PORT`/`inferServiceProc`/`inferServiceAvailable` 변수 제거, `startInferService()` 함수 제거, `app.listen()` 콜백에서 호출 제거, `GET /api/infer/status` 라우트 제거, 프로세스 종료 핸들러(`exit`/`SIGINT`/`SIGTERM`)에서 추론서비스 kill 로직 제거, 미사용 `spawnSync` import 제거 |
| `SimulApp/public/index.html` | "🤖 멀티모달 추론" 탭 버튼 및 `tab-ollama` 패널(채팅 테스트 UI) 제거, 관련 JS(`OLLAMA_API`, `checkOllamaService`, `previewChatImage`, `clearChatImage`, `appendChatBubble`, `clearChatHistory`, `sendChat`) 제거, `switchTab()`의 `ollama` 분기 및 `init()`의 `/api/infer/status` 확인 로직 제거, 이미 죽은 이전 Ollama 등록 탭 잔재 CSS(`.service-ok/err`, `.chat-*`, `.arch-badge`, `.model-list-table`, `#registerLog`, `#modelfileContent`)도 함께 정리 |
| `codeset/docs/ollama_service.md` | 삭제된 서비스로 문서 갱신 (모델정의서 → 제거 이력 기록) |

**효과:** SimulApp은 이제 DB 연결만 되면 파인튜닝/병합/GGUF 변환 관리 기능이 torch/transformers
로딩 없이 즉시 기동됨. 포트 9999 프로세스가 더 이상 실행되지 않음.

---

## 2026-06-25 (추가 4)

### SimulApp — Ollama 잔재 제거 + inference_service.py 코딩 스타일 정비

**변경 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `SimulApp/ollama_service.py` | 삭제 (더 이상 사용 안 함) |
| `SimulApp/server.js` | `OLLAMA_SERVICE_SCRIPT/PORT` → `INFER_SERVICE_SCRIPT/PORT` 이름 변경, `ollamaServiceProc` → `inferServiceProc`, `startOllamaService()` → `startInferService()`, `/api/ollama/status` 라우트 → `/api/infer/status`, 콘솔 로그 "Ollama서비스" → "추론서비스" |
| `SimulApp/public/index.html` | 초기화 시 `/api/ollama/status` → `/api/infer/status` 호출 변경 |
| `SimulApp/inference_service.py` | 추론 로직을 `inferGemma3()` 함수로 분리, `pydantic.BaseModel`(`StatusResponse`) 추가, `for i in range(0, len(...))` 루프 스타일 적용, 모든 함수에 docstring 추가 |

---

## 2026-06-25 (추가 3)

### SimulApp — Ollama 탭 → 멀티모달 추론 서비스 전환

**변경 배경:** Ollama + GGUF 방식은 멀티모달(이미지+텍스트) 미지원. `.venvg3` 환경의 HuggingFace 패키지를 직접 활용해 파인튜닝 병합 모델을 추론.

**변경 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `SimulApp/inference_service.py` | 신규 생성 — FastAPI 포트 9999, `.venvg3` 환경, `/status` + `/chat` 엔드포인트, `.env`의 `MERGED_MODEL_REPO` 로드 |
| `SimulApp/server.js` | `OLLAMA_SERVICE_SCRIPT` → `inference_service.py` 변경, `startOllamaService()` Python 실행 경로 `VENVGGUF_PY_NIX` → `PYTHON_CMD`(.venvg3) 변경, `/api/ollama/status` health check URL `/gguf/list` → `/status` 변경 |
| `SimulApp/public/index.html` | 탭 버튼명 "Ollama 테스트" → "멀티모달 추론" 변경, 탭 내용 재구성 (GGUF 등록 카드①②제거 → 채팅 단일 카드, 모델 선택 제거, maxNewTokens/temperature 추가), `checkOllamaService()` → `/status` 엔드포인트 호출로 변경, `sendChat()` → `/chat` 엔드포인트 호출 (modelName 파라미터 제거) |

**엔드포인트 변경:**
- `GET /status` → `{success, status, ready, model}` (status: loading/ready/no_model/error)
- `POST /chat` → FormData: message, systemPrompt, image(선택), maxNewTokens, temperature

---

## 2026-06-25 (추가 2)

### ★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py / .ipynb (safetensors 0.8.0 shared tensor 에러 수정 + Hub 업로드 방식 변경)

**문제:** `merge_and_unload()` 후 `save_pretrained()` 호출 시 `safetensors 0.8.0` 에서 shared tensor 에러 발생
- Gemma3의 `lm_head.weight`와 `embed_tokens.weight`가 동일 메모리를 공유 (tie_word_embeddings=True)
- 구버전 safetensors는 묵인, 0.8.0부터 strict 검사로 에러 → 업그레이드 이후 신규 발생

**변경 내용:**

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py` | import | `from huggingface_hub import login, HfApi` (`HfApi` 추가) |
| `★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py` | `saveMerged()` | 양자화 잔재 검증 로직 이동 + `mergedModel.tie_weights()` 추가 (save 직전) |
| `★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py` | `uploadToHub()` | 시그니처 변경 (`mergedModel, processor` 제거 → `mergedLocalDir` 만 받음), `push_to_hub` → `HfApi.upload_folder()` 로 변경, 업로드 완료 후 `shutil.rmtree()` 로 로컬 삭제 |
| `★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb` | cell `7b03fc04` | `import shutil`, `HfApi` 임포트 추가 |
| `★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb` | cell `6b70c0da` | 양자화 잔재 검증 + `merged_model.tie_weights()` 추가 (save 직전) |
| `★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb` | cell `c83c9c93` | `HfApi.upload_folder()` + `shutil.rmtree()` 로 변경 |

**동작 흐름 (변경 후):**
1. `saveMerged()`: 양자화 검증 → `tie_weights()` → `save_pretrained()` → 로컬 저장 완료
2. `uploadToHub()`: `HfApi.upload_folder()` → Hub 업로드 → `shutil.rmtree()` 로컬 삭제

**`tie_weights()` 설명:** PEFT merge 후 `_tied_weights_keys` 가 초기화될 수 있어, transformers가 중복 가중치를 state_dict에서 제거하지 못하는 문제 방지. `tie_weights()` 호출로 명시적 재등록.

---

## 2026-06-25 (추가)

### SimulApp — Ollama 테스트 탭 신규 추가

**변경 파일:**

| 파일 | 변경 내용 |
|------|----------|
| `SimulApp/ollama_service.py` | 신규 생성 — FastAPI 서비스 (포트 8001), .venvgguf 환경 |
| `SimulApp/server.js` | `startOllamaService()` 추가 — 서버 시작 시 ollama_service.py 자동 실행 |
| `SimulApp/public/index.html` | "🤖 Ollama 테스트" 탭 추가 |

**ollama_service.py 주요 엔드포인트:**

| 엔드포인트 | 역할 |
|-----------|------|
| `GET /gguf/list` | 프로젝트 내 .gguf 파일 스캔 (10MB 미만·llama.cpp 내부 제외) |
| `POST /gguf/metadata` | GGUF 헤더에서 `general.architecture` 읽기 → Modelfile 초안 생성 |
| `GET /ollama/models` | ollama list 결과 반환 |
| `POST /ollama/register` | Modelfile 저장 + `ollama create` 실행 |
| `DELETE /ollama/model/{name}` | `ollama rm` 실행 |
| `POST /ollama/chat` | Ollama 채팅 — 이미지 업로드 시 base64 변환 후 멀티모달 전달 |

**지원 아키텍처 자동 감지:** gemma3, gemma, gemma4, llama, qwen2, phi3, mistral, command-r

**사용 전 준비:** `.venvgguf` 에 `fastapi uvicorn python-multipart` 설치 필요

---

## 2026-06-24 (추가)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py / .ipynb (혼합 배치 collateFn 최종 수정)

**문제 1차:** `PER_DEVICE_TRAIN_BATCH_SIZE` 를 2 이상으로 올리면 `ValueError: Invalid input type` 발생 (빈 리스트 `[]` 구조)
**문제 2차:** flat list `[img1, img3]` 방식 → `ValueError: Received inconsistently sized batches of images (1) and text (N)` — `make_nested_list_of_images` 가 PIL Image 리스트 전체를 1개 배치로 인식
**문제 3차:** None 방식 `[img, None, ...]` → `is_valid_image(None) = False` 로 여전히 에러

**원인 최종 (확정):** `make_nested_list_of_images([img1, img2, ..., imgN])` 은 PIL Image 리스트를 "이미지 N개짜리 배치 1건" 으로 해석 → `len(batched_images) = 1 ≠ len(texts) = N` → 에러. 올바른 구조는 `[[img1], [img2], ..., [imgN]]` (샘플당 [img] 중첩).

**변경 내용 (최종 v3):**

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `getCollateFn` | `nestedImages` 에 `[sampleImage]` (중첩 리스트) 또는 `None` 추가. 전부-이미지/전부-텍스트/혼합 3가지 경우 분기 처리 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.py` | `_mergeMixedBatch` | 혼합 배치용 내부 함수 추가 — 이미지 서브배치와 텍스트 서브배치를 패딩 맞춰 원본 순서로 병합 |
| `★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb` | Cell 15 (`collate_fn`) | 동일하게 반영 |

**동작 원리:**
- 전부-이미지 배치: `images = [[img1],[img2],...]` → `make_nested_list_of_images` 가 각 `[img_i]` 를 별도 샘플로 인식 → `len = N == len(texts) = N`
- 혼합 배치: 이미지 샘플/텍스트 샘플 분리 → 각각 `processor()` 호출 → `_mergeMixedBatch` 로 원본 인덱스 순서로 텐서 합산

---

## 2026-06-24

### SimulApp/server.js + SimulApp/public/index.html (프로세스 지속성 및 로그 영구 보존)

**문제:** 브라우저 창 또는 서버 터미널 창을 닫고 재접속하면 학습 로그가 사라지고 학습이 중단됨

**원인:**
1. Python 프로세스가 Node.js 의 자식 프로세스로 실행되어 Node 종료 시 함께 종료
2. 로그(`trainLogs`, `mergeLogs`, `ggufLogs`)가 Node.js 메모리에만 존재 → 서버 재시작 시 소멸
3. 브라우저 `init()` 함수가 재접속 시 기존 로그를 즉시 표시하지 않고 2초 폴링 대기 후 표시

**변경 내용:**

| 파일 | 변경 내용 |
|------|-----------|
| `SimulApp/server.js` | Python 프로세스를 `bash -c '... & echo $!'` + `detached: true` + `unref()` 방식으로 Node.js 와 분리 실행 — Node 재시작 후에도 학습 유지 |
| `SimulApp/server.js` | 각 프로세스 PID 를 `codeset/logs/train.pid`, `merge.pid`, `gguf.pid` 파일에 저장 |
| `SimulApp/server.js` | Python 출력을 `codeset/logs/train.log`, `merge.log`, `gguf.log` 파일로 리디렉션하고 1초마다 파일 감시(`startLogWatch`) |
| `SimulApp/server.js` | 서버 시작 시 `recoverProcesses()` 호출 — PID 파일 확인 후 프로세스 재연결 또는 완료된 로그 복원 |
| `SimulApp/server.js` | `currentProc` (프로세스 객체) → `currentProcPid` (PID 정수) 방식으로 프로세스 추적 전환, `isPidRunning()` 으로 생존 여부 확인 |
| `SimulApp/public/index.html` | `init()` 함수 개선 — 재접속 시 `/api/run/status?offset=0` 응답의 로그를 즉시 화면에 표시 후 올바른 offset 으로 폴링 시작 (병합·GGUF 동일하게 적용) |

**동작 흐름 (서버 재시작 후):**
1. 서버 시작 → `recoverProcesses()` 실행
2. PID 파일 존재 + 프로세스 살아있음 → 로그 파일 로드 + 파일 감시 재개
3. PID 파일 존재 + 프로세스 없음 → 완료된 로그만 복원, PID 파일 삭제
4. 브라우저 접속 → `init()` 에서 로그 즉시 렌더링 + 실행 중이면 폴링 자동 시작

**추가 파일:** `codeset/logs/` 디렉터리 (서버 최초 시작 시 자동 생성)

---

## 2026-06-23 (추가)

### SimulApp/server.js + SimulApp/public/index.html (학습 중지 버튼 + 시뮬레이션 테이블 데이터셋 컬럼)

**변경 내용:**

| 파일 | 변경 내용 |
|------|-----------|
| `SimulApp/server.js` | `POST /api/run/stop` 엔드포인트 신규 추가 — `currentProc.kill('SIGTERM')` 후 DB `simulation_runs` 중 `status='running'` 레코드를 `stopped`/`end_timestamp=NOW()`로 업데이트 |
| `SimulApp/public/index.html` | `.btn-stop` / `.s-stopped` CSS 추가; `■ 학습 중지` 버튼 추가 (학습 중일 때만 표시); `stopTraining()` 함수 추가; `updateBadge()` — Stop 버튼 show/hide 연동; 시뮬레이션 결과 테이블 `데이터셋` 컬럼(11번째) 추가 — `dataset_repo` 표시(마우스오버 시 전체 경로 tooltip) |

**동작 흐름:**
1. 학습 시작 → `■ 학습 중지` 버튼 활성화(빨간색)
2. 중지 버튼 클릭 → 확인 다이얼로그 → `POST /api/run/stop` 호출
3. 서버: 파이썬 프로세스 SIGTERM 종료 → DB status='stopped' 업데이트
4. 로그박스에 "■ 학습 중지됨 (사용자 요청)" 표시, 배지 대기 중으로 복귀

---

## 2026-06-23

### ★04. GGUF 모델 변환.py / ★04. GGUF 모델 변환.ipynb

**문제:** GGUF 변환 subprocess 실행 시 `ModuleNotFoundError: No module named 'transformers'` 오류 발생

**원인:** `convertToGguf()` 함수 내 `subprocess.run(['python', ...])` 호출이 `.venvgguf` 가상환경 파이썬이 아닌 시스템 파이썬(`/usr/bin/python`)을 사용. 시스템 파이썬에는 `transformers` 미설치.

**변경 내용:**

| 파일 | 위치 | 변경 전 | 변경 후 |
|------|------|---------|---------|
| `★04. GGUF 모델 변환.py` | 임포트 | `import os, subprocess` | `import sys` 추가 |
| `★04. GGUF 모델 변환.py` | `convertToGguf()` L112 | `cmd = ['python', convertScript, ...]` | `cmd = [sys.executable, convertScript, ...]` |
| `★04. GGUF 모델 변환.ipynb` | 셀 `15e06be4` | `cmd = ['python', convertScript, ...]` | `import sys` 추가 + `cmd = [sys.executable, convertScript, ...]` |

**효과:** `sys.executable`은 현재 스크립트를 실행 중인 파이썬 인터프리터 경로를 반환하므로, `server.js`가 `.venvgguf/bin/python`으로 스크립트를 실행하면 subprocess도 동일하게 `.venvgguf/bin/python`을 사용하여 `transformers` 정상 인식

---

## 2026-06-12

### SimulApp/public/index.html (UI 개선)

**변경 내용:** ⚙ 학습 설정 탭에 데이터셋 포맷 안내 카드 및 하이퍼파라미터 설명 툴팁 추가

| 추가 항목 | 내용 |
|-----------|------|
| **📋 필수 데이터셋 포맷 카드** | modality / image / instruction / input / output / source / label 7개 컬럼에 대한 설명 표(타입, 필수 여부, 예시 값) — 파인튜닝 전 데이터 구조 확인 목적 |
| **ℹ 툴팁 (모든 파라미터)** | 각 하이퍼파라미터 레이블 옆 `?` 아이콘 — 마우스오버 시 한국어 설명 팝업 |

**툴팁 적용 파라미터:**
- 학습 하이퍼파라미터: `MAX_SEQ_LENGTH`, `TRAIN BATCH SIZE`, `EVAL BATCH SIZE`, `GRAD_ACCUM`, `NUM_EPOCHS`, `LEARNING_RATE`, `LOGGING_STEPS`, `SAVE_STEPS`
- LoRA 설정: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`
- 데이터/모델: `DATASET_REPO`, `BASE_MODEL`, `OUTPUT_BASE_DIR`, `MAX_TRAIN/EVAL_SAMPLES`
- 병합 탭: `BASE_MODEL`, `ADAPTER_PATH`, `MERGED_LOCAL_DIR`, `MERGED_MODEL_REPO`, `TEST_IMAGE_PATH`

**구현 방식:** CSS only (JS 없음) — `.tip-icon:hover .tip-box { display: block }` + `position: absolute` 팝업

---

## 2026-06-03

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb

**문제:** 대용량 OCR 데이터 처리 시 PIL 이미지를 메모리에 전부 누적하다 약 5분 후 OOM으로 커널 사망

**변경 내용:**

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `f8a9b0c1` | 전체 루프 → `image_data` 리스트에 PIL 객체 누적 | 500개 단위 청크 처리 → JPEG bytes로 변환 후 pickle 저장, resume 지원 |
| `a9b0c1d2` | `Dataset.from_list(image_data)` | `Dataset.from_generator(ocrDataGenerator)` — 스트리밍 방식 |
| `d2e3f4a5` | `push_to_hub(repo, token)` | `push_to_hub(repo, token, max_shard_size="500MB")` — 샤드 분할 |

**효과:**
- 청크별 저장으로 중단 시 이어서 재처리 가능 (`./dataset/chunks_ocr/chunk_NNNN.pkl`)
- PIL → JPEG bytes 변환으로 처리 중 메모리 약 60~70% 절감
- HF Hub 업로드 시 대용량 파일 자동 분할 처리

---

## 2026-06-11

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb

**변경 내용:** 하드코딩된 설정값을 `.env` 파일에서 로드하도록 전환

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `b8c9d0e1` | `HF_DATASET_REPO`, `MAX_OCR_SAMPLES` 하드코딩 | `load_dotenv()` 통합, 전체 설정 env화 |
| `c9d0e1f2` | OCR/텍스트 경로 하드코딩 | `os.getenv()` 로 env에서 로드 |
| `d0e1f2a3` | `from dotenv import load_dotenv`, `load_dotenv()` 중복 호출 | `hf_token = HF_TOKEN` 표시만 (중복 제거) |
| `f8a9b0c1` | `CHUNK_SIZE=500`, `instrText`/`inputText` 하드코딩 | `CHUNK_SIZE`, `OCR_INSTR_TEXT`, `OCR_INPUT_TEXT` env화 |
| `471eb9f3` | `HF_DATASET_REPO = 'hyokwan/ocr_dataset'` 중복 선언 | 확인 출력으로 변경 (중복 제거) |
| `c7d019a4` | `ARROW_DIR = "./dataset/ocr_arrow"` 하드코딩 | `os.getenv("ARROW_DIR", ...)` 로 env화 |

**신규 `.env` 항목:** `HF_DATASET_REPO`, `OCR_ROOT`, `TEXT_JSON_PATH`, `MAX_OCR_SAMPLES`, `CHUNK_SIZE`, `DATASET_DIR`, `SEED`, `OCR_INSTR_TEXT`, `OCR_INPUT_TEXT`

### ★01. 데이터 준비_멀티모달데이터_Gemma3_OCR.ipynb (저장 구조 개선)

**변경 내용:** pickle 청크(`CHUNK_DIR`) + Arrow(`ARROW_DIR`) 이중 저장을 Arrow shard 단일 저장(`DATASET_DIR`)으로 통합

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `f8a9b0c1` | JPEG bytes → pickle 저장 | Arrow shard 직접 저장 (`Dataset.save_to_disk`) |
| `402fc61b` | pickle 로드 → PIL 변환 → concatenate | Arrow shard 로드 → concatenate |
| `3dad128c`, `a9b0c1d2`, `f4ce9aec`, `c7d019a4` | 주석 처리된 대안 코드·구 Arrow 로드 셀 | 삭제 |

**`.env` 변경:** `CHUNK_DIR` + `ARROW_DIR` → `DATASET_DIR` 단일화

---

## 2026-06-11 (추가 2)

### ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.ipynb

**변경 내용:** `.env` 전체 통합, 하드코딩 제거, 코드 정리

| 셀 ID | 변경 내용 |
|-------|-----------|
| `460134ee` | 미사용 `import json`, `from pathlib import Path` 제거, 섹션 주석 정리 |
| `ba6a9f1b` | `load_dotenv()` + 전체 설정값 통합 (`IMAGE_ROOT`, `IMAGE_META_ROOT`, `TEXT_JSON_PATH`, `IMAGE_INSTRUCTION`, `IMAGE_INPUT`, `MAX_IMAGE_SAMPLES_PER_CLASS`, `SEED`) |
| `eb199edf` | 삭제 (경로 하드코딩 셀 → `ba6a9f1b`에 통합) |
| `d8200afb` | 삭제 (하드코딩 토큰 셀 → `ba6a9f1b`에 통합) |
| `8c245bf9` | `text_json_path` → `TEXT_JSON_PATH` 변수 사용 |
| `60d477a4` | `text_dataset` → `textDataset` (camelCase) |
| `35332348` | `image_meta_root` → `IMAGE_META_ROOT`, 로드 결과 출력 추가 |
| `3f14f6bb` | `image_data` → `imageData`, `for i in range(0, len(df))` 통일, `IMAGE_INSTRUCTION`/`IMAGE_INPUT` env 변수 사용, `MAX_IMAGE_SAMPLES_PER_CLASS` 실제 동작 구현 |
| `867ef52b` | `image_dataset` → `imageDataset` (camelCase) |
| `afa9b05d` | `text_dataset`/`image_dataset`/`all_dataset` → camelCase 변수명 통일 |
| `703f56f6` | `all_dataset` → `allDataset`, 하드코딩 `42` → `SEED` 변수 사용 |
| `1bfb3eee` | `all_dataset` → `allDataset` |

**신규 `.env` 항목:** `IMAGE_ROOT`, `IMAGE_META_ROOT`, `MAX_IMAGE_SAMPLES_PER_CLASS`, `IMAGE_INSTRUCTION`, `IMAGE_INPUT`

### ★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py (신규 생성)

**변경 내용:** 노트북을 독립 실행 가능한 `.py` 스크립트로 변환

| 함수 | 내용 |
|------|------|
| `loadConfig()` | `.env` 전체 파싱 → 딕셔너리 단일 진입점 |
| `loadTextData(cfg)` | 텍스트 JSON 로드 → Dataset 반환, try-except 에러 처리 |
| `loadImageData(cfg)` | 이미지 CSV+PIL 로드, `MAX_IMAGE_SAMPLES_PER_CLASS` 제한, try-except |
| `mergeAndUpload(textDataset, imageDataset, cfg)` | 캐스팅·병합·셔플·Hub 업로드, try-except |
| `main()` | 파이프라인 조립 (각 단계 None 체크 포함) |

**실행:** `python "★01. 데이터 준비_멀티모달데이터_Gemma3_스포츠.py"`

---

## 2026-06-11 (추가)

### 02. 파인튜닝_멀티모달데이터_Gemma3.ipynb

**변경 내용:** 구조 전면 개선 — 라이브러리 통합, 한글 폰트, .env 하이퍼파라미터 시뮬레이션, TensorBoard, DB 저장

| 셀 ID | 변경 내용 |
|-------|-----------|
| `f1fa00ac` | 임포트 최상단 통합 (matplotlib, PIL, mysql.connector 추가) |
| `8ca492e5` (신규) | `setKoreanFont()` — matplotlib 한글 깨짐 방지 (맑은 고딕 우선) |
| `34708266` | 하드코딩 → 전체 `.env` 로드로 전환 (`TENSORBOARD_DIR` 추가) |
| `3fcbdab0` | `build_messages` → `buildMessages` (camelCase), 독스트링 추가 |
| `35954667` | `collate_fn` — `buildMessages` 호출로 갱신, for 루프 정비 |
| `8cfdace8` | `r/lora_alpha/dropout` → 변수 참조, `report_to='tensorboard'`, `logging_dir=TENSORBOARD_DIR` |
| `88c1d0dd` | `train_result = trainer.train()` (결과 변수 저장) |
| `90bee0ad` (신규) | `plotTrainingLoss()` — 학습 손실 그래프 (한글 폰트 적용) |
| `292bff0d` (신규) | `saveTrainingResultToDB()` — MySQL `training_runs` 테이블 자동 생성 + 저장 |
| `84b4c0d9` | `infer_unified` 독스트링 추가 |
| `c188f5c0` | PIL 중복 임포트 제거 |
| 삭제 | 중복 `load_dataset` 셀, 인라인 테스트 셀 3개, 주석처리 셀 2개 |

**신규 `.env` 항목:** `DATASET_REPO`, `BASE_MODEL`, `OUTPUT_BASE_DIR`, `MAX_SEQ_LENGTH`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `PER_DEVICE_EVAL_BATCH_SIZE`, `GRAD_ACCUM`, `NUM_EPOCHS`, `LEARNING_RATE`, `LOGGING_STEPS`, `SAVE_STEPS`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py (신규 생성)

**변경 내용:** 노트북을 독립 실행 가능한 `.py` 스크립트로 변환

| 함수 | 내용 |
|------|------|
| `loadConfig()` (신규) | `.env` 전체 파싱 → 딕셔너리 반환 (전역 cfg 단일 진입점) |
| `getCollateFn(processor, model)` (신규) | 전역 변수 의존 제거 — 클로저로 processor/model 캡처 |
| `inferUnified(...)` | 파라미터로 model/processor 전달 (전역 의존 제거) |
| `main()` | 전체 학습 파이프라인 (데이터 로드→학습→시각화→DB저장→모델저장) |

**실행:** `python "★02. 파인튜닝_멀티모달데이터_Gemma3.py"`

---

## 2026-06-11 (추가 5)

### ★03. 데이터병합 및 저장_멀티모달데이터_Gemma3.py (신규 생성)

**변경 내용:** 노트북을 독립 실행 가능한 `.py` 스크립트로 변환, SimulApp 병합 탭 연동

| 함수 | 내용 |
|------|------|
| `loadConfig()` | 스크립트 위치 기준 `.env` 로드 |
| `hfLogin(hfToken)` | 로그인 실패 시 경고만 출력 (크래시 없음) |
| `loadBaseModel(cfg)` | 4-bit 양자화 베이스 모델 + 프로세서 로드 |
| `mergeAdapter(baseModel, adapterPath)` | PeftModel → merge_and_unload() |
| `saveMerged / uploadToHub` | 로컬 저장 + Hub 업로드 |
| `inferFromHub` | 텍스트/이미지 추론 검증 |

### SimulApp 🔀 모델 병합 탭 (신규)

**변경 내용:** `server.js` API 추가 + `index.html` 탭 추가

| 파일 | 변경 내용 |
|------|-----------|
| `SimulApp/server.js` | `MODELS_DIR`, `PY_MERGE` 경로 추가; `mergeProc`/`mergeLogs` 상태 추적; `/api/models`, `/api/merge/config GET·POST`, `/api/merge/start`, `/api/merge/status` API 신규 |
| `SimulApp/public/index.html` | 🔀 모델 병합 탭 추가 — 어댑터 폴더 드롭다운, 설정 폼, 병합 실행·로그 폴링 |

---

## 2026-06-11 (추가 4)

### ★03_데이터병합 및 저장_멀티모달데이터_Gemma3.ipynb

**변경 내용:** `.env` 전체 통합, 하드코딩 토큰 제거

| 셀 ID | 변경 내용 |
|-------|-----------|
| `51267a36` | 하드코딩 토큰 제거 → `load_dotenv()` + `os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")` + `login(hf_token)` |
| `d71acf41` | 경로 5개 전부 `.env` 전환 (`BASE_MODEL`, `ADAPTER_PATH`, `MERGED_LOCAL_DIR`, `MERGED_MODEL_REPO`, `TEST_IMAGE_PATH`) |
| `c433b74f` | 베이스 모델·프로세서 로드에 `token=hf_token` 추가 |

**신규 `.env` 항목:** `ADAPTER_PATH`, `MERGED_LOCAL_DIR`, `MERGED_MODEL_REPO`, `TEST_IMAGE_PATH`

### codeset/docs/★03_데이터병합 및 저장_멀티모달데이터_Gemma3.md (신규 생성)

**변경 내용:** 노트북 모델 정의서 작성 (셀 구성, .env 설정, 처리 흐름, 알려진 경고)

---

## 2026-06-11 (추가 3 — SimulApp)

### ★02. 파인튜닝_멀티모달데이터_Gemma3.ipynb (DB 연동 적용)

**변경 내용:** .py와 동일한 DB 연동 구조를 노트북에도 반영

| 셀 ID | 변경 내용 |
|-------|-----------|
| `f1fa00ac` | `mysql.connector`, `TrainerCallback`, `matplotlib.font_manager`, `gridspec` 임포트 추가 |
| `ea23b15e` (신규) | `setKoreanFont()` — matplotlib 한글 폰트 자동 설정 (맑은 고딕 우선) |
| `053d9fd6` | 하드코딩 토큰 제거 → `load_dotenv()` + `os.getenv("HF_TOKEN")` 만 유지 |
| `34708266` | 전체 `.env` 설정 전환 (DB_CFG, TENSORBOARD_DIR, 모든 LoRA·학습 하이퍼파라미터) |
| `eea23c1a` (신규) | `getDbConn`, `getNextSimSeq`, `insertRunStart`, `updateRunEnd`, `DbLogCallback` + `SIM_SEQ` 발급 |
| `3fcbdab0` | `build_messages` → `buildMessages` (camelCase), 독스트링 추가 |
| `35954667` | `collate_fn` — `buildMessages` 호출, `for i in range(0, len())` 루프 통일 |
| `8cfdace8` | LoRA r/α/dropout → 변수 참조, `report_to='tensorboard'`, `logging_dir=TENSORBOARD_DIR`, `DbLogCallback` 추가 |
| `88c1d0dd` | `train_result = trainer.train()` + `updateRunEnd(SIM_SEQ, ...)` |
| `b7a785d3` | 그래프 한글 레이블 적용, 변수명 camelCase 통일, LoRA r/α 동적 표시 |
| `20a38b0d` | CSV 저장 → DB 조회 확인으로 교체 |
| 삭제 | `5981f985`, `53b3eb10`, `1913f7f7`, `7ba914c8`, `52acc4cb`, `95fa7bb1`, `04353c5c` (중복·테스트·주석 셀) |

---

### ★02. 파인튜닝_멀티모달데이터_Gemma3.py (DB 연동 재구축)

**변경 내용:** sim_seq 기반 DB 저장, DbLogCallback, SimulApp 연동 지원으로 전면 재작성

| 추가/변경 내용 | 설명 |
|----------------|------|
| `getNextSimSeq(dbCfg)` | `MAX(sim_seq)+1` 조회로 시뮬레이션 번호 자동 발급 |
| `insertRunStart(simSeq, cfg)` | 학습 시작 시 `simulation_runs` INSERT (status=running) |
| `updateRunEnd(simSeq, ...)` | 완료/실패 시 결과 UPDATE (train_loss, runtime_sec 등) |
| `DbLogCallback` | `TrainerCallback` 서브클래스 — 스텝마다 `simulation_logs` INSERT |
| `loadConfig()` | `os.path.dirname(os.path.abspath(__file__))` 기준 `.env` 로드 (SimulApp 다른 cwd에서 spawn 시에도 정상 동작) |
| `try-except` | `trainer.train()` 실패 시 status=failed로 DB 기록 후 예외 재발생 |

### SimulApp/ (신규 생성)

**변경 내용:** Express 기반 파인튜닝 시뮬레이션 웹 앱 신규 구성

| 파일 | 내용 |
|------|------|
| `SimulApp/db_create.sql` | `simulation_runs`, `simulation_logs` 테이블 생성 SQL |
| `SimulApp/package.json` | express, mysql2, dotenv 의존성 정의 |
| `SimulApp/server.js` | Express API 서버 — `.env` 읽기/쓰기, Python 스크립트 spawn, DB 조회, 로그 폴링 |
| `SimulApp/public/index.html` | HTML/JS UI — 설정 탭(파라미터 저장·학습 실행·로그 실시간 표시) + 결과 탭(이력 테이블·Chart.js 손실 그래프) |

**SimulApp/server.js 주요 API:**

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/config` | GET | `.env` 하이퍼파라미터 조회 |
| `/api/config` | POST | `.env` 하이퍼파라미터 저장 |
| `/api/run/start` | POST | Python 파인튜닝 스크립트 spawn |
| `/api/run/status` | GET | 실행 상태 + 로그 폴링 (offset 파라미터) |
| `/api/simulations` | GET | `simulation_runs` 이력 조회 (최근 50건) |
| `/api/simulations/:seq` | GET | 특정 시뮬레이션 상세 |
| `/api/simulations/:seq/logs` | GET | 스텝별 손실 로그 조회 |

**실행:**
```bash
cd SimulApp
npm install
node server.js
# 브라우저: http://localhost:3000
```

### codeset/docs/★02. 파인튜닝_멀티모달데이터_Gemma3.md (신규 생성)

**변경 내용:** 파인튜닝 스크립트 모델 정의서 작성 (함수 구조, .env 설정, DB 스키마, 출력 경로)

---

## 2026-06-07

### ★01. 데이터 준비_멀티모달데이터_Gemma3.ipynb

**문제:** `concatenate_datasets([image_dataset, text_dataset])` 실행 시 `ValueError: The features can't be aligned` 발생

**원인:**
- `Dataset.from_list()` (image_dataset): 문자열 컬럼 → `Value('string')`
- `Dataset.from_pandas()` (text_dataset): 문자열 컬럼 → `Value('large_string')` (최신 datasets 라이브러리 변경 사항)
- 두 타입이 불일치하여 concatenate 불가

**변경 내용:**

| 셀 ID | 변경 전 | 변경 후 |
|-------|---------|---------|
| `afa9b05d` | `concatenate_datasets([image_dataset, text_dataset])` 직접 호출 | `text_dataset.cast(image_dataset.features)` 로 타입 통일 후 concatenate |

**효과:**
- `large_string` → `string` 타입 캐스팅으로 피처 정렬 오류 해결
- Python 버전 무관, datasets 라이브러리 버전 업그레이드 시에도 안정적으로 동작
