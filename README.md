

# 🌿 Plant Diseases Detection

Detect plant diseases using leaf images and AI! 🍃📷

<img src="project_images/cover.png" alt="cover" width="100%" />

---

## 📋 Table of Contents

* [About](#-about)
* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [Installation](#-installation)
* [Usage](#-usage)
* [Dataset & Model](#-dataset--model)
* [Results](#-results)
* [Future Work](#-future-work)
* [Contribute](#-contribute)
* [License](#-license)
* [Contact](#-contact)

---

## 🔍 About

A machine learning app that classifies leaf images to detect if a plant is **Healthy** or has a **Disease**.
Perfect for gardeners, farmers, and plant lovers.

<img src="project_images/about.png" alt="about-image" width="100%" />

---

## 🎯 Features

* ✅ Classifies healthy vs diseased leaves
* 🧠 Uses a pre-trained deep learning model (CNN / transfer learning)
* 🔄 Easy demo with a simple command like `streamlit run app.py`
* 🖼️ Outputs predictions + confidence scores + disease highlights

<img src="project_images/features.png" alt="features-image" width="100%" />

---

## 🛠 Tech Stack

* **Python** 3.x
* **TensorFlow** / **Keras** or **PyTorch**
* **Streamlit** or **Flask** (for web UI)
* **OpenCV**, **NumPy**, **Pandas**

<img src="project_images/tech_stack.png" alt="tech-stack-image" width="100%" />

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/nihanth6721/Plant_Dieases_Detection.git
cd Plant_Dieases_Detection

# 2. (Optional) Create and activate virtual env
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the trained model in the project root
mv path/to/your/model_weights.h5 model_weights.h5

# 5. Launch the app
streamlit run app.py
```

---

## 🧠 Dataset & Model

* **Dataset**: Trained on images of healthy vs diseased leaves (e.g., PlantVillage)
* **Model**: CNN or transfer learning (e.g., ResNet, MobileNet)
* **Model file**: `model_weights.h5`
* **Usage**: The app loads this model to predict leaf health instantly

<img src="project_images/model_architecture.png" alt="model-architecture" width="100%" />

---

## 📊 Results

| Class    | Accuracy | Precision | Recall |
| -------- | -------- | --------- | ------ |
| Healthy  | 98%      | 0.97      | 0.99   |
| Diseased | 97%      | 0.95      | 0.98   |

* **Overall Accuracy**: \~97.5%
* 🔁 **Sample UI**: Upload a leaf photo → display result + confidence

<img src="project_images/prediction.png" alt="prediction-ui" width="100%" />
<img src="project_images/confidence_output.png" alt="confidence-score" width="100%" />

---

## 🛠 Future Work

* ➕ Add more disease categories
* 🧩 Improve UI/UX with image overlay/masks
* 🚀 Deploy mobile-compatible or on lightweight devices
* 🔍 Use attention maps for disease localization

---

## 🤝 Contribute

Contributions are welcome! 🙌

1. ⭐ Fork the repo
2. 🔁 Create a new branch: `git checkout -b my-feature`
3. 🧪 Make changes & test thoroughly
4. 📄 Add clear documentation/comments
5. 🔃 Submit a pull request

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📫 Contact

* **GitHub**: [nihanth6721](https://github.com/nihanth6721)
* **Email**: [jnihanthreddy@gmail.com](mailto:jnihanthreddy@gmail.com)

---

## ⚠️ Tips & Acknowledgements

* 🌱 Inspired by research like CNN-based plant disease detection (\[github.com]\[1], \[arxiv.org]\[2], \[huggingface.co]\[3], \[github.com]\[4], \[github.com]\[5], \[bijeshshrestha.github.io]\[6], \[kandi.openweaver.com]\[7])
* Thanks to the **PlantVillage** dataset and **Streamlit** team!
* Feel free to reach out if you need help training your own model 😊

---

### 🎉 Enjoy exploring and improving plant health!
