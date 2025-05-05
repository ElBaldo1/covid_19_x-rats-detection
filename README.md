# COVID-19 Chest X-Ray Detection with CNN

This project classifies chest X-ray images as COVID-positive or COVID-negative using a convolutional neural network in PyTorch.


## 📂 Project layout

```
.
├── covid_classification.py   # training + evaluation
├── predict.py                # CLI inference: python predict.py <img>
├── streamlit_app.py          # Streamlit front‑end
├── import_Data.py            # CTDataset wrapper
├── outputs/                  # auto‑generated weights + metrics
│   ├── best_model.pth
│   ├── classification_report.txt
│   ├── cm.png
│   ├── training_curves.png
│   └── training_log.csv
├── COVID-Data-Radiography/   # dataset root (after unzip)
│   ├── no/
│   └── yes/
├── manual-test/              # 6 images for sanity checks
├── environment.yml           # Conda environment
└── README.md
```


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
# Download best model 
wget "https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/Ebtd2VterepMkv3GaYyifkUB6sPdmryJrQ8C62nkz0VBpQ" -O best_model.pth

# Download dataset (zipped)
wget "https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EUsuXgv53vVPuY9Su8gDRIIBb9qZsLrx1XwDduUH5ScVVA" -O COVID-Data-Radiography.zip

# Unzip# Unzip\unzip COVID-Data-Radiography.zip
```
If the above commands fail, open the links below in a browser and download manually:
- [best_model.pth](https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/Ebtd2VterepMkv3GaYyifkUB6sPdmryJrQ8C62nkz0VBpQ)
- [COVID-Data-Radiography.zip](https://universitelibrebruxelles-my.sharepoint.com/:u:/g/personal/antonio_baldari_ulb_be/EUsuXgv53vVPuY9Su8gDRIIBb9qZsLrx1XwDduUH5ScVVA)

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

---

📌 GitHub Repository: [github.com/ElBaldo1/covid_19_x-rats-detection](https://github.com/ElBaldo1/covid_19_x-rats-detection)
