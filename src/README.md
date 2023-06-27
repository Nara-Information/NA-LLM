# `src` for NA-LLM

NA-LLM을 개발하는데 사용한 스크립트들의 모음입니다.

데이터 구축 과정 중 일부 과정은 수동 처리가 진행되었습니다:

- 공공데이터포털 데이터에 대한, 내용에 기반한 정제 (인삿말 제거 등)
- OpenAI 생셩 결과의 검토 및 내용 추출 

## 주의사항 

- HuggingFace Hub의 비공개 레포지토리나 WanDB를 사용하기 위해서는, 코드를 실행하기 전에 미리 쉘 환경을 만들어 주세요.

## 파일 소개 

- main.py: 스크립트 실행을 위한 시작점. 아래 모드를 받아서 실행합니다. 모드는 `python main.py`로 실행한 뒤 CLI로 선택할 수 있고, `python main.py fetch`와 같이 `argv`로 넘길수도 있습니다. 
    - fetch: 공공데이터포털에서 민원 데이터 획득 
    - augment: OpenAI API를 통한 데이터 증강 
    - trainTranslate: seq2seq 목적을 통한 BART 훈련 
    - trainCausal: CausalLM 목적을 통한 polyglot-ko 훈련
- config.yaml: 패키지 전반에서 사용되는 설정들을 정리한 파일입니다. `yaml` 문법에 맞추어 값을 변경해 주세요.
- ./tests: 몇몇 unittest 파일들 
- ./train: 훈련에 사용된 스크립트 
- ./utils: 자잘한 도구 함수들 
    - ./utils/data: 데이터의 수집, 정제, 증강 등에 관련한 모듈 
- allow.txt, ignore.txt: 증강에서 사용하는 allowlist 및 ignorelist 파일. 해당 파일에 등장하는 qano를 각각 allowlist와 ignorelist로 처리합니다  ('#'로 시작하는 줄은 무시합니다). 이러한 파일의 위치를 변경하려면 `config.yaml`에서 해당 경로를 변경해 주세요.
- requirements.txt: 훈련에 사용한 pypi 패키지. transformers 등 일부 패키지는 peft 및 qlora 지원을 위해 GitHub 버전을 가져옵니다.