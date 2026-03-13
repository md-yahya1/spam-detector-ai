<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,40:1a0a2e,100:2d1b69&height=220&section=header&text=AI%20Spam%20Detector&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=%F0%9F%9B%A1%EF%B8%8F%20Real-time%20NLP%20classifier%20%E2%80%94%20Text%20or%20File%2C%20Spam%20or%20Not.&descSize=16&descAlignY=60&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://spam-detector-ai-i839.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/md-yahya1/spam-detector-ai?style=for-the-badge&color=f59e0b&labelColor=0d1117)](https://github.com/md-yahya1/spam-detector-ai/stargazers)

<br/>

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-spam--detector--ai.onrender.com-7c3aed?style=for-the-badge)](https://spam-detector-ai-i839.onrender.com)

</div>

---

## 🧠 What Is This?

**AI Spam Detector** is a production-deployed ML web app that classifies text messages and uploaded `.txt` email files as **Spam** or **Not Spam** in real time. Built on a Naive Bayes + TF-IDF pipeline and served through a lightweight Flask backend — containerized with Docker and live on Render.

> A complete end-to-end ML project: from raw dataset → trained model → serialized pipeline → web API → cloud deployment.

---

## ✨ Key Features

- 🛡️ **Real-Time Classification** — Paste any text and get an instant Spam / Not Spam verdict
- 📂 **File Upload Support** — Upload `.txt` email files directly for analysis
- 📊 **Model Accuracy Display** — The app shows the model's live accuracy on screen
- 🐳 **Dockerized** — Fully containerized for consistent, reproducible deployments
- ☁️ **Cloud Deployed** — Live on Render, accessible from anywhere

---

## 🏗️ Architecture

<p align="center">
  <img src="images/Architecture.png" width="800"/>
</p>

---

## 📸 Screenshots

<details>
<summary><b>🔴 Spam Detection — Example 1</b></summary>
<br/>
<p align="center">
  <img src="images/SpamCase1.png" width="860"/>
</p>
</details>

<details>
<summary><b>🔴 Spam Detection — Example 2</b></summary>
<br/>
<p align="center">
  <img src="images/SpamCase2.png" width="860"/>
</p>
</details>

<details>
<summary><b>🟢 Not Spam — Example</b></summary>
<br/>
<p align="center">
  <img src="images/NotSpamCase.png" width="860"/>
</p>
</details>

<details>
<summary><b>📂 File Upload — Example</b></summary>
<br/>
<p align="center">
  <img src="images/fileUploadCase.png" width="860"/>
</p>
</details>

---

## 🛠️ Tech Stack

<div align="center">

**ML Pipeline**

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**Web & Backend**

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**DevOps & Deployment**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

</div>

---

## 📁 Project Structure

```bash
spam-detector-ai/
│
├── app.py                  # 🌐 Flask app — routes & prediction logic
├── spam_train.py           # 🧪 Model training script
│
├── spam_model.pkl          # 🤖 Serialized Naive Bayes classifier
├── vectorizer.pkl          # 🔤 Serialized TF-IDF vectorizer
├── accuracy.txt            # 📊 Stored model accuracy (displayed in UI)
│
├── templates/
│   └── index.html          # 🖥️  Frontend UI template
│
├── images/                 # 🖼️  Screenshots & architecture diagram
│
├── requirements.txt        # 📦 Python dependencies
└── Dockerfile              # 🐳 Container configuration
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/md-yahya1/spam-detector-ai.git
cd spam-detector-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python spam_train.py

# 4. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t spam-detector-ai .

# Run
docker run -p 5000:5000 spam-detector-ai
```

---

## 🧪 How the ML Pipeline Works

```
Input Text / .txt File
        ↓
   TF-IDF Vectorizer  ←── vectorizer.pkl
        ↓
  Naive Bayes Classifier  ←── spam_model.pkl
        ↓
  SPAM 🔴  or  NOT SPAM 🟢
```

---

## 🗺️ Roadmap

- [x] Naive Bayes + TF-IDF classification pipeline
- [x] Flask REST backend
- [x] `.txt` file upload support
- [x] Docker containerization
- [x] Cloud deployment on Render
- [ ] Confidence score / probability display
- [ ] Support `.eml` and `.pdf` formats
- [ ] Compare SVM / Logistic Regression accuracy
- [ ] Highlight words that triggered spam detection

---

## 📚 What I Learned

- End-to-end ML pipeline — from raw CSV to a deployed, user-facing product
- Model serialization — saving and loading `pickle` files correctly in production
- Flask integration — connecting an ML model to a web backend cleanly
- Docker — writing Dockerfiles and understanding containerized environments
- Cloud deployment debugging — diagnosing issues between local Docker and Render's runtime

---

## 🙌 Acknowledgements

| Resource | Purpose |
|---|---|
| [SMS Spam Collection – Kaggle](https://www.kaggle.com/) | Training dataset |
| [scikit-learn](https://scikit-learn.org/) | ML pipeline |
| [Flask](https://flask.palletsprojects.com/) | Web framework |
| [Docker](https://www.docker.com/) | Containerization |
| [Render](https://render.com/) | Cloud deployment |

---

<div align="center">

**Built from scratch by [Mohammed Yahya](https://github.com/md-yahya1)**

[![GitHub](https://img.shields.io/badge/GitHub-md--yahya1-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/md-yahya1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohammed-yahya-4b9879326)
[![Live App](https://img.shields.io/badge/%F0%9F%9A%80_Try_it_Live-Render-46E3B7?style=for-the-badge)](https://spam-detector-ai-i839.onrender.com)

<br/>

⭐ **Found this useful? Star the repo — it takes 2 seconds and means a lot!** ⭐

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2d1b69,50:1a0a2e,100:0a0a0a&height=100&section=footer" width="100%"/>

</div>
