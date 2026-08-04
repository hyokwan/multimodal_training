# 모델 정의서: ollama_service.py / inference_service.py (제거됨)

## 상태: 삭제됨 (2026-08-04)

`SimulApp/ollama_service.py` (포트 8001, GGUF+Ollama 등록용)는 2026-06-25에 이미 삭제되었고,
그 후속으로 `SimulApp/inference_service.py`(포트 9999, `.venvg3` 환경에서 HuggingFace 모델을
직접 로드해 채팅 추론을 제공하던 FastAPI 서비스)가 사용되고 있었습니다.

`inference_service.py`가 `server.js`에 의해 자동 spawn되면서, 모델 로드(torch/transformers
checkpoint shard 로딩)가 SimulApp 기동 과정에 결합되어 웹 UI가 제대로 로딩되지 않는 문제가
발생했습니다. 이에 따라 SimulApp을 파인튜닝/병합/GGUF 변환 관리 전용으로 가볍게 유지하기 위해
멀티모달 추론 관련 코드를 전부 제거했습니다.

**제거된 항목:**

| 파일 | 처리 |
|------|------|
| `SimulApp/inference_service.py` | 삭제 |
| `SimulApp/server.js` | `INFER_SERVICE_SCRIPT`/`INFER_SERVICE_PORT`, `inferServiceProc`, `inferServiceAvailable`, `startInferService()`, `GET /api/infer/status` 라우트, 프로세스 종료 시 추론서비스 kill 로직 제거 |
| `SimulApp/public/index.html` | "🤖 멀티모달 추론" 탭 버튼·패널, 채팅 UI(`chatHistory`, `sendChat()` 등 관련 JS/CSS) 전체 제거 |

**현재 상태:** SimulApp은 DB(`my8003.gabiadb.com/hkcodedb`) 연결만 있으면 파인튜닝/병합/GGUF
관리 기능이 정상 동작하며, 추론 관련 프로세스를 spawn하지 않습니다. 멀티모달 추론 서비스는
별도 프로젝트로 분리하여 개발 예정입니다.

**관련 로그:** `docs/rev_log.md` — "SimulApp — 멀티모달 추론 서비스 전면 제거" 항목 참고.
