# Machine Learning-Based Industrial Fault Detection System

This repository contains a final year individual research project prototype for:

**Design and Development of a Machine Learning-Based Industrial Fault Detection System Using Sensor Data**

The system is a Streamlit web application that uses the UCI SECOM semiconductor manufacturing dataset to detect possible fault/fail production records from sensor readings.

The recommended workflow is:

1. Keep the project files in GitHub.
2. Run model training in Google Colab using `notebooks/SECOM_Model_Training_Colab.ipynb`.
3. Export `models/secom_fault_detection_model.joblib`.
4. Commit that model artifact to GitHub, then deploy the Streamlit app on Streamlit Cloud.

## Main Features

- Loads and combines `secom.data`, `secom_labels.data`, and `secom.names`
- Handles missing sensor readings with median imputation
- Removes high-missing and constant sensor features
- Handles class imbalance using class weights and balanced sample weights
- Provides Colab-ready training code for multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - XGBoost automatically if installed
- Reports accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, and classification report
- Provides prediction, sensor insight, and decision-support dashboard pages
- Selects the deployment model using fault-class F1, then recall, PR-AUC, and balanced accuracy
- Provides an adjustable operating threshold for sensitivity/false-alarm trade-offs
- Validates uploaded CSV files and allows prediction/risk-register downloads
- Loads the trained Joblib artifact in Streamlit Cloud for prediction and decision support

## Current Tested Model

The committed artifact was regenerated with the final workflow and selects **Logistic Regression**. On the stratified 25% hold-out split it produced:

- Accuracy: `0.7908`
- Balanced accuracy: `0.6557`
- Precision: `0.1585`
- Recall: `0.5000`
- F1-score: `0.2407`
- ROC-AUC: `0.6880`
- PR-AUC: `0.1609`

These values are reproducible development results, not claims of production readiness. Re-run the Colab notebook and use that final output in the dissertation after supervisor review.

## Project Structure

```text
.
|-- app.py
|-- pages/
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- feature_selection.py
|   |-- model_training.py
|   |-- evaluation.py
|   |-- prediction.py
|   |-- visualizations.py
|   `-- utils.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|   `-- SECOM_Model_Training_Colab.ipynb
|-- tests/
|-- assets/
|-- docs/
|-- runtime.txt
|-- requirements.txt
`-- README.md
```

## Dataset

The project uses the UCI SECOM dataset. The raw files should be available in `data/raw/`:

- `secom.data`
- `secom_labels.data`
- `secom.names`

The original labels use `-1` for pass and `1` for fail. In this system they are mapped to:

- `0`: Normal / Pass
- `1`: Fault / Fail

## Run Locally

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app.

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

If `models/secom_fault_detection_model.joblib` is not available yet, the overview and data-quality pages will still work. Prediction, model-evaluation, sensor-insight, and decision-support pages need the Colab-exported model artifact.

## Train in Google Colab

1. Push this project to GitHub.
2. Open `notebooks/SECOM_Model_Training_Colab.ipynb` in Google Colab.
3. Set `REPO_URL` in the notebook to your GitHub repository URL.
4. Run all cells.
5. Download `secom_fault_detection_model.joblib`.
6. Place it in this repo under:

```text
models/secom_fault_detection_model.joblib
```

7. Commit and push the model artifact to GitHub.

The notebook should be restarted and run from top to bottom. The upgraded workflow embeds library-version metadata and selects the deployment model using fault-class F1 rather than ordinary accuracy.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover dataset shape/label mapping, saved-artifact inference, risk ranking, and uploaded-input validation.

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Make sure the raw SECOM files are included under `data/raw/`.
3. Make sure the trained model exists at `models/secom_fault_detection_model.joblib`.
4. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
5. Create a new app and connect the GitHub repository.
6. Set the main file path to:

```text
app.py
```

7. Deploy the app. Streamlit Cloud will install packages from `requirements.txt`.

The app and committed Joblib artifact are deployed with the exact tested package versions in `requirements.txt`, including scikit-learn `1.6.1`. Update those pins only when the model is retrained, re-exported, and retested with the replacement environment.

The official deployment model artifact `models/secom_fault_detection_model.joblib` is allowed by `.gitignore` so Streamlit Cloud can load it from GitHub.

After deployment, verify all six pages and replace the marked thesis placeholders using the checklist in `docs/final_results_and_screenshot_checklist.md`.

## Academic Notes

This repository includes short support documents in `docs/` for methodology, dataset explanation, evaluation planning, system overview, and viva preparation. These files are draft learning material and should be edited by the student with supervisor feedback before being used in formal submissions.

## Ethical and Professional Considerations

This project uses a public industrial dataset and does not collect personal data or human participant data. The student should still cite the dataset source correctly, explain limitations, and avoid claiming that the model alone is sufficient for real manufacturing decisions.
