# Models

Place the Colab-exported model artifact here:

```text
models/secom_fault_detection_model.joblib
```

This specific file is allowed by `.gitignore` so it can be committed to GitHub and loaded by Streamlit Cloud. Other temporary Joblib files are ignored.

The current artifact was trained using scikit-learn `1.6.1`. The exact version is pinned in `requirements.txt` to avoid incompatible Joblib deserialisation on Streamlit Cloud.
