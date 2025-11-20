
[HF link](https://huggingface.co/collections/ai-sage/gigachat3)
----

# GigaChat 3 Ultra & Lightning

Открытые MoE-модели нового поколения: **GigaChat 3 Ultra Preview (702B-A36B)** и **GigaChat 3 Lightning (10B-A1.8B)**.

В этом году мы уже представили линейку GigaChat 2, добавили Reasoning в Web-версию (giga.chat), открыли GigaChat Lite и Giga-Embeddings и заняли первое место на бенчмарке ruMTEB.  
Теперь мы публикуем открытые веса MoE-моделей нового поколения, обученных с нуля, без зависимости от зарубежных весов.

Мы открываем веса двух моделей, доступных сообществу с лицензией MIT и возможностью коммерческого использования:

- **GigaChat 3 Ultra Preview (702B-A36B)** — флагманская instruct-модель.
- **GigaChat 3 Lightning (10B-A1.8B)** — компактная MoE-модель для локального и высоконагруженного использования.

---

## Список моделей и чекпоинтов

### GigaChat 3 Ultra Preview

- Instruct-модель:  
  - [`ai-sage/GigaChat3-702B-A36B-preview`](https://huggingface.co/ai-sage/GigaChat3-702B-A36B-preview)
- Чекпоинт для дообучения (bf16):  
  - [`ai-sage/GigaChat3-702B-A36B-preview-bf16`](https://huggingface.co/ai-sage/GigaChat3-702B-A36B-preview-bf16)

### GigaChat 3 Lightning

- Base-модель (pretrain):  
  - [`ai-sage/GigaChat3-10B-A1.8B-base`](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base)
- Instruct-модель (для диалогов и инструкций):  
  - [`ai-sage/GigaChat3-10B-A1.8B`](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)

---

## Общая архитектура

Обе модели используют кастомную архитектуру **Mixture-of-Experts (MoE)** с поддержкой:

- **Multi-head Latent Attention (MLA)**  
- **Multi-Token Prediction (MTP)**  

### Multi-head Latent Attention (MLA)

Вместо стандартного Multi-head Attention используется **Multi-head Latent Attention**:

- Key-Value (KV) кэш сжимается в латентное представление, что:
  - снижает требования к памяти;
  - уменьшает размер KV-кэша;
  - ускоряет обработку длинных контекстов.

Это особенно заметно на больших контекстах (десятки и сотни тысяч токенов) и при кластерном инференсе.

### Multi-Token Prediction (MTP)

Обе модели обучены с задачей **Multi-Token Prediction (MTP)**:

- модель учится предсказывать несколько токенов за один проход;
- это позволяет использовать спекулятивную/параллельную генерацию и ускорять инференс примерно до +40 %;
- в типичных конфигурациях (vLLM / SGLang) MTP даёт пропускную способность, сопоставимую с более мелкими dense-моделями.

---

## GigaChat 3 Ultra Preview (702B-A36B)

### Краткое описание

`GigaChat3 Ultra Preview` — флагманская **instruct-модель** семейства GigaChat на архитектуре **Mixture-of-Experts (MoE)**.

Ключевые характеристики:

- около 702B параметров, из них примерно 36B активируются на токен за счёт разрежённой архитектуры;
- сочетает качество моделей «топ-класса» с практически применимой скоростью инференса;
- архитектура вдохновлена DeepSeek V3 (MoE + MLA + MTP), но модель обучена с нуля на собственном корпусе;
- топ-1 на MERA;
- контекст: до 131k токенов;
- демонстрирует скорость выше, чем GigaChat 2 Max.

Для дообучения доступен bf16-чекпоинт:  
[`ai-sage/GigaChat3-702B-A36B-preview-bf16`](https://huggingface.co/ai-sage/GigaChat3-702B-A36B-preview-bf16).

Подробнее про архитектуру и обучение будет в статье на Habr (to do).

---

## GigaChat 3 Lightning (10B-A1.8B)

### Краткое описание

`GigaChat3-10B-A1.8B` — компактная MoE-модель следующего поколения.

Ключевые характеристики:

- 10B общих параметров, 1.8B активных на токен;
- по качеству достигает уровня лидера open-source своего класса Qwen3-4B;
- по скорости генерации примерно в 1.5 раза быстрее, сопоставима с Qwen3-1.7B;
- подходит для локального использования:
  - офлайн-ассистенты;
  - прототипирование и RAG;
  - LLM-классификаторы и high-load RAG-сценарии;
- хорошая база для высокопроизводительного эмбеддера на CPU (за счёт MoE-архитектуры);
- поддерживаемый контекст: до 256k токенов.

### Base vs Instruct

- **Base (pretrain)** — `GigaChat3-10B-A1.8B-base`  
  Используется для дообучения и кастомных задач.
- **Instruct** — `GigaChat3-10B-A1.8B`  
  Рекомендуется для диалоговых сценариев, ассистентов и выполнения инструкций.

---

## Данные и обучение

Обе модели обучены с нуля на многоязычном и разнообразном корпусе.

Общие свойства корпуса:

- более 10 языков (включая русский, английский, китайский, арабский, узбекский, казахский и другие);
- источники:
  - книги и нон-фикшн;
  - академические данные;
  - датасеты по коду и математике;
  - диалоговые и инструктивные датасеты;
- предобработка:
  - дедупликация;
  - языковая фильтрация;
  - автоматические проверки качества при помощи эвристик и классификаторов.

Существенный вклад в качество дала синтетика:

- около 5.5T токенов синтетических данных;
- в корпус входят:
  - вопросы-ответы к текстам;
  - цепочки reverse-prompt для структурирования данных;
  - комментарии и заметки модели внутри текстов;
  - миллионы синтетических задач с решениями по математике и олимпиадному программированию (с автогенерируемыми тестами) на основе PromptCot.

---

## Инференс и деплой

### Ultra (702B-A36B)

`GigaChat3 Ultra Preview` ориентирована на кластерные и on-prem-сценарии с серьёзной инфраструктурой:

- поддерживаются популярные inference-движки:
  - vLLM
  - SGLang
  - LMDeploy
  - TensorRT-LLM
  - другие фреймворки;
- поддерживаются режимы BF16 и FP8 (для FP8 требуется отдельная сборка и настройки GPU);
- MLA и MTP уменьшают размер KV-кэша и ускоряют генерацию;
- рекомендуется использовать прокси/gateway-слой для интеграции с внешними сервисами, тулзами и агентными фреймворками.

Для ориентировочной конфигурации можно смотреть гайды по моделям схожего масштаба:

- DeepSeek-V3 — раздел *How to run locally*:  
  <https://github.com/deepseek-ai/DeepSeek-V3?tab=readme-ov-file#6-how-to-run-locally>
- Kimi-K2-Instruct — рекомендации по деплою (vLLM / SGLang / LMDeploy):  
  <https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/deploy_guidance.md>

### Lightning (10B-A1.8B): производительность

Одно из ключевых преимуществ `GigaChat3-10B-A1.8B` — скорость инференса.

Модель (особенно в MTP-режиме) по пропускной способности сопоставима с гораздо более мелкими dense-моделями, оставаясь при этом заметно сильнее по качеству.

Измерения проводились через vLLM v0.11.0, dtype bfloat16, `batch_size=1`.  
Код бенчмарка: <https://gist.github.com/ajpqs/ce941aa6f0f48ef36a65cb87a2a1d726>.

| Модель                          | request_throughput | output_throughput | total_token_throughput | mean_ttft_ms |
|---------------------------------|--------------------|-------------------|------------------------|--------------|
| `Qwen3-1.7B`                    | 1.689              | 357.308           | 726.093                | 11.824       |
| `mtp-GigaChat3-10B-A1.8B-base`  | 1.533              | 333.620           | 678.894                | 26.345       |
| `GigaChat3-10B-A1.8B-base`      | 1.077              | 234.363           | 476.912                | 31.053       |
| `Qwen3-4B`                      | 0.978              | 206.849           | 420.341                | 14.947       |
| `Qwen3-8B`                      | 0.664              | 140.432           | 285.375                | 16.663       |
| `YandexGPT-5-Lite-8B-pretrain`  | 0.641              | 147.305           | 300.269                | 16.711       |

Несмотря на 10B параметров, по скорости и стоимости инференса модель можно рассматривать как альтернативу dense-моделям на 3–4B параметров, а в MTP-режиме — сравнивать и с меньшими.

---

## Общие свойства моделей

Обе модели:

- не являются специализированными reasoning-моделями (но поддерживают базовый уровень рассуждений);
- умеют предсказывать несколько токенов за один шаг (MTP);
- используют MLA, что уменьшает размер KV-кэша и снижает требования по памяти;
- обучены с нуля, без инициализации чужими весами;
- совместимы с:
  - Hugging Face;
  - vLLM / SGLang / LMDeploy;
  - стандартными пайплайнами инференса и дообучения;
- распространяются по лицензии MIT и могут использоваться в коммерческих продуктах.

---

## Quickstart: GigaChat3-10B-A1.8B-base

### 1. `transformers`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "ai-sage/GigaChat3-10B-A1.8B-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.generation_config = GenerationConfig.from_pretrained(model_name)

prompt = "Ниже я написал подробное доказательство теоремы о неподвижной точке:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=400,
    )

result = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=False,
)
print(result)
````

### 2. `vLLM`

Запуск сервера:

```bash
vllm serve ai-sage/GigaChat3-10B-A1.8B-base \
  --disable-sliding-window \
  --dtype "auto" \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "disable_padded_drafter_batch": false}'
```

Пример запроса:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-sage/GigaChat3-10B-A1.8B-base",
    "prompt": "Ниже я написал подробное доказательство теоремы о неподвижной точке:",
    "max_tokens": 400,
    "temperature": 0
  }'
```

### 3. `SGLang`

Запуск сервера:

```bash
python -m sglang.launch_server \
  --model-path ai-sage/GigaChat3-10B-A1.8B-base \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype auto \
  --mem-fraction-static 0.88 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1
```

Пример запроса:

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-sage/GigaChat3-10B-A1.8B-base",
    "prompt": "Ниже я написал подробное доказательство теоремы о неподвижной точке:",
    "max_tokens": 400,
    "temperature": 0
  }'
```

---

## Лицензия

Модели распространяются по лицензии **MIT**.
Вы можете использовать их в исследовательских и коммерческих проектах при сохранении текста лицензии.

