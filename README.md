<p align="center">
  <img src="https://github.com/Nara-Information/NA-LLM/assets/136791147/a9b9328e-4e1a-45c2-9d47-61540949d48d"/>
</p>

## Update Logs
- 2023.06.27: [Polyglot-ko 3.8B 기반 NALLM-polyglot-ko-3.8B 모델](https://huggingface.co/nojiyoon/nallm-polyglot-ko-3.8b-base) 공개
- 2023.06.26: [Polyglot-ko 1.3B 기반 NALLM-polyglot-ko-1.3B 모델](https://huggingface.co/nojiyoon/nallm-polyglot-ko-1.3b-base) 공개
- 2023.06.26: [KoBART 기반 NALLM-kobart 모델]() 공개
- - -

# NA-LLM(나름): NAra information Large Language Model
### 👆 하나는 나름 잘 하는 대화형 언어모델

- NA-LLM(나름)은 [나라지식정보](http://narainformation.com/)가 개발한 한국어 Large Language Model (LLM) 입니다.

- 모든 것을 말할 수 있는 언어 모델은 아닙니다. 하지만, 한국 사람들이 공기관에 제기하는 민원과 같은 텍스트를 넣으면, 공기관과 유사한 방식으로 대답하는 것 하나만은 나름 잘 하는 언어모델입니다.

  
## Example
  (이미지 필요)


## Backbone Model: KoBART, Polyglot-ko
NA-LLM(나름)은 Backbone Model로 [KoBART](https://github.com/SKT-AI/KoBART), [Polyglot-ko](https://github.com/EleutherAI/polyglot)를 사용하여 학습을 진행하였습니다.
  
  
## NA-LLM 모델 실행 예시 코드
### Huggingface Pipeline으로 실행
  (코드 필요)  
  
  
## Dataset
### 나름 데이터셋 v1
나름 데이터셋 v1은 [공공데이터포털 국민권익위원회_민원정책 질의응답조회서비스](https://www.data.go.kr/data/15074671/openapi.do)를 기반으로 합니다.
