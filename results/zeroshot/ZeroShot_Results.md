# Zero-Shot LLM Evaluation Results

Prompt: `zeroshot`, temperature 0, no training data used. Positive class = fake. *Real acc.* = accuracy on human_real reviews (true-negative rate). *Majority* = accuracy of always predicting the majority class. Best accuracy and F1 per dataset in bold.

| Dataset | Model | Accuracy | Precision | Recall | F1 | Real acc. | Test | Majority |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **d1** human_real vs human_fake | gemma4:e4b | 0.568 | 0.729 | 0.216 | 0.333 | 0.920 | 398 | 0.500 |
|  | doomgrave/ministral-3:8b | 0.580 | 0.695 | 0.286 | 0.406 | 0.874 |  |  |
|  | llama3.2:3b | 0.595 | 0.636 | 0.447 | **0.525** | 0.744 |  |  |
|  | qwen3.5:9b | **0.606** | 0.839 | 0.261 | 0.398 | 0.950 |  |  |
| **d2** sampled, human_real vs synthetic | gemma4:e4b | **0.578** | 0.772 | 0.221 | **0.344** | 0.935 | 398 | 0.500 |
|  | doomgrave/ministral-3:8b | 0.467 | 0.349 | 0.075 | 0.124 | 0.859 |  |  |
|  | llama3.2:3b | 0.377 | 0.111 | 0.035 | 0.053 | 0.719 |  |  |
|  | qwen3.5:9b | **0.578** | 0.792 | 0.211 | 0.333 | 0.945 |  |  |
| **d2.5** sampled, human_real vs synthetic | gemma4:e4b | **0.492** | 0.440 | 0.055 | **0.098** | 0.930 | 398 | 0.500 |
|  | doomgrave/ministral-3:8b | 0.440 | 0.000 | 0.000 | 0.000 | 0.879 |  |  |
|  | llama3.2:3b | 0.382 | 0.000 | 0.000 | 0.000 | 0.764 |  |  |
|  | qwen3.5:9b | 0.490 | 0.300 | 0.015 | 0.029 | 0.965 |  |  |
| **d2.5** unsampled, human_real vs synthetic | gemma4:e4b | **0.318** | 0.681 | 0.067 | **0.121** | 0.925 | 679 | 0.707 |
|  | doomgrave/ministral-3:8b | 0.252 | 0.000 | 0.000 | 0.000 | 0.859 |  |  |
|  | llama3.2:3b | 0.221 | 0.000 | 0.000 | 0.000 | 0.754 |  |  |
|  | qwen3.5:9b | 0.284 | 0.312 | 0.010 | 0.020 | 0.945 |  |  |
| **d3** sampled, human_real vs mixed | gemma4:e4b | 0.570 | 0.759 | 0.206 | 0.324 | 0.935 | 398 | 0.500 |
|  | doomgrave/ministral-3:8b | 0.535 | 0.609 | 0.196 | 0.297 | 0.874 |  |  |
|  | llama3.2:3b | 0.460 | 0.418 | 0.206 | 0.276 | 0.714 |  |  |
|  | qwen3.5:9b | **0.613** | 0.846 | 0.276 | **0.417** | 0.950 |  |  |
| **d3** unsampled, human_real vs mixed | gemma4:e4b | 0.477 | 0.852 | 0.261 | 0.399 | 0.909 | 598 | 0.667 |
|  | doomgrave/ministral-3:8b | 0.430 | 0.790 | 0.198 | 0.317 | 0.894 |  |  |
|  | llama3.2:3b | 0.388 | 0.634 | 0.195 | 0.299 | 0.774 |  |  |
|  | qwen3.5:9b | **0.505** | 0.881 | 0.298 | **0.446** | 0.920 |  |  |
| **d3.5** sampled, human_real vs mixed | gemma4:e4b | 0.523 | 0.596 | 0.141 | 0.228 | 0.904 | 398 | 0.500 |
|  | doomgrave/ministral-3:8b | 0.500 | 0.500 | 0.131 | 0.207 | 0.869 |  |  |
|  | llama3.2:3b | 0.447 | 0.376 | 0.161 | 0.225 | 0.734 |  |  |
|  | qwen3.5:9b | **0.550** | 0.750 | 0.151 | **0.251** | 0.950 |  |  |
| **d3.5** unsampled, human_real vs mixed | gemma4:e4b | **0.300** | 0.856 | 0.113 | **0.200** | 0.935 | 878 | 0.773 |
|  | doomgrave/ministral-3:8b | 0.263 | 0.716 | 0.078 | 0.141 | 0.894 |  |  |
|  | llama3.2:3b | 0.249 | 0.583 | 0.103 | 0.175 | 0.749 |  |  |
|  | qwen3.5:9b † | 0.254 | 0.923 | 0.091 | 0.166 | 0.967 |  |  |

† qwen3.5:9b on d3.5 unsampled returned an unparseable answer for 555 of 878 test rows; its metrics cover only the 323 scored rows.

## Notable patterns

- All four models are strongly biased toward predicting *real*: real-review accuracy is 0.71–0.97, while fake recall never exceeds 0.45.
- No model beats the majority-class baseline on the unsampled (imbalanced) datasets; on balanced sets the best accuracy is only 0.61.
- LLM-generated (synthetic) reviews are almost undetectable zero-shot: on d2.5 recall is ≤ 0.07, and ministral/llama catch 0 of them.
- d1 (human-written fakes) is the easiest setting; llama3.2:3b has the best F1 there (0.525) because it is the least biased toward *real*.
- qwen3.5:9b and gemma4:e4b have the best accuracy/F1 on most datasets, with precision up to 0.92 but recall never above 0.30.
- Compared with RAG (see [../rag/RAG_Results.md](../rag/RAG_Results.md)), zero-shot is far weaker, especially on synthetic reviews (e.g. qwen3.5:9b d2.5 sampled F1 0.885 with RAG vs 0.029 zero-shot).
