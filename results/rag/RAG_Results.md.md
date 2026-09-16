# Dataset–Model Evaluation Results

| Dataset | Model | Accuracy | Precision | Recall | F1 | Train | Test |
|---|---|---:|---:|---:|---:|---:|---:|
| **d1** human_real vs human_fake | retrieval_only | 0.714 | 0.664 | 0.864 | 0.751 | 1194 | 398 |
| | gemma4:e4b | 0.631 | 0.760 | 0.382 | 0.508 | | |
| | doomgrave/ministral-3:8b | 0.688 | 0.660 | 0.779 | 0.714 | | |
| | llama3.2:3b | 0.472 | 0.420 | 0.146 | 0.216 | | |
| | qwen3.5:9b | **0.739** | 0.717 | 0.789 | **0.751** | | |
| **d2** sampled, human_real vs synthetic | retrieval_only | 0.902 | 0.885 | 0.925 | 0.904 | 1194 | 398 |
| | gemma4:e4b | 0.648 | 0.748 | 0.447 | 0.560 | | |
| | doomgrave/ministral-3:8b | 0.709 | 0.686 | 0.769 | 0.725 | | |
| | llama3.2:3b | 0.317 | 0.146 | 0.075 | 0.099 | | |
| | qwen3.5:9b | 0.844 | 0.837 | 0.854 | 0.846 | | |
| **d2.5** sampled, human_real vs synthetic | retrieval_only | 0.869 | 0.798 | 0.990 | 0.883 | 1194 | 398 |
| | gemma4:e4b | 0.661 | 0.758 | 0.472 | 0.582 | | |
| | doomgrave/ministral-3:8b | 0.648 | 0.637 | 0.688 | 0.662 | | |
| | llama3.2:3b | 0.319 | 0.100 | 0.045 | 0.062 | | |
| | qwen3.5:9b | 0.877 | 0.829 | 0.950 | 0.885 | | |
| **d2.5** unsampled, human_real vs synthetic | retrieval_only | 0.869 | 0.844 | **1.000** | 0.915 | 2037 | 679 |
| | gemma4:e4b | 0.629 | 0.868 | 0.560 | 0.681 | | |
| | doomgrave/ministral-3:8b | 0.610 | 0.767 | 0.644 | 0.700 | | |
| | llama3.2:3b | 0.214 | 0.179 | 0.031 | 0.053 | | |
| | qwen3.5:9b | **0.890** | 0.873 | 0.988 | **0.927** | | |
| **d3** sampled, human_real vs mixed | retrieval_only | 0.709 | 0.671 | 0.819 | 0.738 | 1194 | 398 |
| | gemma4:e4b | 0.616 | 0.735 | 0.362 | 0.485 | | |
| | doomgrave/ministral-3:8b | 0.673 | 0.652 | 0.744 | 0.695 | | |
| | llama3.2:3b | 0.467 | 0.393 | 0.121 | 0.185 | | |
| | qwen3.5:9b | 0.721 | 0.698 | 0.779 | 0.736 | | |
| **d3** unsampled, human_real vs mixed | retrieval_only | 0.764 | 0.765 | 0.932 | 0.841 | 1794 | 598 |
| | gemma4:e4b | 0.585 | 0.879 | 0.439 | 0.585 | | |
| | doomgrave/ministral-3:8b | 0.722 | 0.793 | 0.789 | 0.791 | | |
| | llama3.2:3b | 0.360 | 0.611 | 0.110 | 0.187 | | |
| | qwen3.5:9b | 0.763 | 0.811 | 0.840 | 0.825 | | |
| **d3.5** sampled, human_real vs mixed | retrieval_only | 0.683 | 0.655 | 0.774 | 0.710 | 1194 | 398 |
| | gemma4:e4b | 0.611 | 0.708 | 0.377 | 0.492 | | |
| | doomgrave/ministral-3:8b | 0.603 | 0.587 | 0.693 | 0.636 | | |
| | llama3.2:3b | 0.455 | 0.371 | 0.131 | 0.193 | | |
| | qwen3.5:9b | 0.761 | 0.739 | 0.809 | 0.772 | | |
| **d3.5** unsampled, human_real vs mixed | retrieval_only | 0.830 | 0.830 | 0.982 | 0.900 | 2634 | 878 |
| | gemma4:e4b | 0.584 | **0.913** | 0.511 | 0.655 | | |
| | doomgrave/ministral-3:8b | 0.629 | 0.805 | 0.686 | 0.741 | | |
| | llama3.2:3b | 0.230 | 0.518 | 0.065 | 0.115 | | |
| | qwen3.5:9b | 0.845 | 0.884 | 0.920 | 0.902 | | |
