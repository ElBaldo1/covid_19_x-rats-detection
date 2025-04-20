# COVID-19 Chest X-Ray Detection with CNN

This project classifies chest X-ray images as COVID-positive or COVID-negative using a convolutional neural network in PyTorch.

## Structure
- `import_Data.py`: dataset loading using `ImageFolder`
- `covid_classification.py`: training, evaluation, confusion matrix
- `predict.py`: command-line interface for single image inference
- `streamlit_app.py`: interactive demo using Streamlit
- `training_log.csv`: training & validation metrics
- `environment.yml`: Conda environment for reproducibility

## Dataset
- `COVID-Data-Radiography/no/` — 1270 training images (COVID-negative)
- `COVID-Data-Radiography/yes/` — 1369 training images (COVID-positive)
- `manual-test/` — 6 unseen images for final evaluation or demo

## 🔁 Reproduce environment
```bash
conda env create -f environment.yml
conda activate covid_proj
```

## 📥 Download pretrained model and dataset
To run `predict.py` or the Streamlit app, download the pretrained model and unzip the dataset:

```bash
# Download best model (80 MB)
wget "https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EYF1ng7UuOZFnCuQHxyuTN0Brj2FU6G_Scssv3a629am7Q?e=le94A4" -O best_model.pth

# Download dataset (~100 MB zipped)
wget "https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EUsuXgv53vVPuY9Su8gDRIIBCp7u55ULlecR4YuO21hEIA?e=s6A1Fm" -O COVID-Data-Radiography.zip

# Unzip# Unzip\unzip COVID-Data-Radiography.zip
```
If the above commands fail, open the links below in a browser and download manually:
- [best_model.pth](https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EYF1ng7UuOZFnCuQHxyuTN0Brj2FU6G_Scssv3a629am7Q?e=le94A4)
- [COVID-Data-Radiography.zip](https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EUsuXgv53vVPuY9Su8gDRIIBCp7u55ULlecR4YuO21hEIA?e=s6A1Fm)

## 🧪 Train the model
```bash
python covid_classification.py
```

## 🔍 Predict manually (CLI)
```bash
python predict.py manual-test/image1.jpg
```

## 🌐 Streamlit demo (interactive)
```bash
streamlit run streamlit_app.py
```
Upload a chest X-ray image to classify it live as COVID-positive or COVID-negative.
