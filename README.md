# **Data Science Tips** 
by João Raposo ist197377

---
## USEFUL TIPS

- I'd recommend first training all classifiers with the required parameters, and [saving them](https://scikit-learn.org/stable/model_persistence.html) using joblib. This way even if the graphics come out wrong, you don't have to retrain the models.
- Store the CSVs of the modifications of the datasets at each change, and with **clear naming of what was changed**. 
- If you care about your sanity, update the word/latex report after each delivery.
- If you have a Nvidia GPU, try to also use Random Forest and Gradient Boosting in all of your earlier decisions, even if you only test with fewer parameters. They won't take that long to compute, and provide better insights regarding the impact of your changes. 
---
## CUDA installation
For running using WSL on Windows, do the following:
1. Install latest Nvidia game ready driver from [Nvidia's website](https://www.nvidia.com/download/index.aspx)
2. Install the cuda toolkit WITHOUT DRIVERS on wsl. Go to the [download page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0) and select Linux as target OS, architecture of your cpu, and WSL-Ubuntu distribution

For installing on Linux directly, simply install the Nvidia game ready driver and [cuda toolkit for linux]() 

---
## Recommended frameworks 
- Naive Bayes: sklearn
- KNN: RapidsAI cuml (over 100 times faster than sklearn)
- Decision Trees: sklearn
- Random Forest: RapidsAI cuml (use sklearn to get feature importances for the best config given by cuml)
- MLP: sklearn
- Gradient Boosting: XGBOOST (gpu_hist is insanely faster than sklearn, even with huge bin sizes )

Relevant pages:
- [RapidsAI](https://github.com/rapidsai/cuml)
- [cuml](https://docs.rapids.ai/api/cuml/stable/)
- [XGBOOST](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [using cuda with xgboost](https://xgboost.readthedocs.io/en/stable/gpu/index.html)