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
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("nojiyoon/nallm-polyglot-ko-1.3b-base")

model = AutoModelForCausalLM.from_pretrained("nojiyoon/nallm-polyglot-ko-1.3b-base")
```
  
## Dataset
### 나름 데이터셋 v1
나름 데이터셋 v1은 [공공데이터포털 국민권익위원회_민원정책 질의응답조회서비스](https://www.data.go.kr/data/15074671/openapi.do)를 기반으로 합니다. (설명 필요)
```json
{
    "qano": 6837003,
    "organization": "경찰청",
    "title": "정보공개청구 방법에 대한 문의(전남)",
    "question": [
            "정보공개 청구방법은 어떻게 하는건가요!"
    ],
    "answer": [
            "1. 공공기관의 정보공개에 관한 법률(약칭: 정보공개법)에서의 공개 및 비공개 대상 정보와 정보공개청구 절차에 대하여 답변드리겠습니다.",
            "​",
            " 가.  공공기관의 정보공개에 관한 법률(이하 “정보공개법) 제3조에 의하면 “공공기관이 보유·관리하는 정보는 이 법이 정하는 바에 따라 공개하여야 한다”라고 하여 원칙적으로 정보를 공개한다고 되어 있습니다.",
            "  나. 그러나 국민의 이 정보공개청구권은 모든 정보에 대하여 인정되는 것은 아니고, 이른바 일반적인 사항에 대해서만 인정이 되고 국가안전보장에 관련되는 정보 및 보안업무를 관장하는 기관에서 국가안전보장과 관련된 정보분석을 목적으로 수집되거나 작성된 정보에 대하여는 이를 인정하지 않고 있습니다.",
            "  # 통상 정보공개가 거부되는 경우는 동법 제9조(비공개 대상) 4항 재판 및 수사와 관련되거나 6항 및 7항에 따라 거부되는 경우가 대부분",
            " 다. 정보공개 청구 절차는 정보공개 청구인이 청구인의 이름, 주민등록번호, 주소 및 연락처, 공개를 청구하는 정보의 내용과 공개방법을 기재한 정보공개청구서(인터넷 또는 공공기관 방문)를 제출 하게되면",
            "  - 공공기관은 10일 이내에 정보공개 여부를 결정(부득이 한 경우 10일 이내의 범위에서 연장가능)하여 청구인에게 공개일시·공개장소 등을 명시하여 청구인에게 통지하게 됩니다(동법 제10조부터 제16조)",
            "  - 정보의 공개 및 우송 등에 소요되는 비용은 실비의 범위 안에서 청구인의 부담으로 하게 되어 있으나, 공개를 청구하는 정보의 사용목적이 공공복리의 유지·증진을 위하여 필요하다고 인정되는 경우에는 비용을 감면할 수도 있습니다(동법 제17조 제1항, 제2항).",
            "2. 정보공개결정에 대하여는 이의가 있을 시 정보공개법 제18조(이의신청)에 따라",
            " - 청구인이 정보공개와 관련한 공공기관의 비공개 결정 또는 부분 공개 결정에 대하여 불복이 있거나",
            " - 정보공개 청구 후 20일이 경과하도록 정보공개 결정이 없는 때에는 공공기관으로부터 정보공개 여부의 결정 통지를 받은 날 또는 정보공개 청구 후 20일이 경과한 날부터 30일 이내에 해당 공공기관에 이의신청을 할 수 있습니다."
    ],
    "gen1": [
            "경찰청에서 정보공개 청구 방법에 대해서 문의드립니다. 저는 전남 지역에 살고 있어서, 이 지역에서 정보공개 청구를 하려면 어떻게 해야 할까요? 알려주시면 감사하겠습니다."
    ],
    "gen2": [
            "정보공개 청구 방법을 몰라서 고민하는 중입니다. 경찰청에서 정보를 얻고 싶은데, 어떻게 해야할까요? 전남 지역에 사는데, 전남 지역에서 정보공개 청구를 하는 방법이 궁금합니다. 부탁드립니다."
    ],
    "gen3": [
            "경찰청에서 정보공개 청구 방법에 대해서 문의드립니다. 전남 지역에서 정보공개 청구를 하고 싶은데, 어떻게 해야 할까요? 공식적인 양식이나, 처리 절차와 기간, 수수료 등이 궁금합니다. 자세한 설명을 부탁드립니다."
   ]
},
```
