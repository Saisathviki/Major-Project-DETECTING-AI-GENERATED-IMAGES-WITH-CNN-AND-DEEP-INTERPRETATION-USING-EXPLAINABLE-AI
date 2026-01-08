

\# Detecting AI-Generated Images with CNN and Deep Interpretation using Explainable AI



\## 📌 Project Overview

With the rapid advancement of generative models such as GANs and diffusion models, AI-generated images are becoming increasingly realistic and difficult to distinguish from real images. This project aims to \*\*detect AI-generated images using Convolutional Neural Networks (CNNs)\*\* and enhance trust and transparency by applying \*\*Explainable AI (XAI)\*\* techniques to interpret the model’s decisions.



The system is implemented as a \*\*Django-based web application\*\*, allowing users to upload images and receive predictions along with visual explanations.



---



\## 🎯 Problem Statement

AI-generated images can be misused for misinformation, deepfakes, identity fraud, and digital manipulation. Traditional detection methods often act as black boxes, offering no explanation for their predictions.  

This project addresses the need for:

\- Accurate detection of AI-generated images

\- Transparent and interpretable model decisions using Explainable AI



---



\## 🧠 Objectives

\- To build a CNN-based model for detecting AI-generated images  

\- To integrate Explainable AI techniques for model interpretability  

\- To develop a user-friendly Django web interface for image upload and analysis  

\- To visualize important regions influencing the model’s decision  



---



\## 🗂 Dataset Description

\- The dataset consists of \*\*real images\*\* and \*\*AI-generated images\*\*

\- Images are preprocessed (resizing, normalization)

\- Data is split into training and testing sets

\- Augmentation techniques are applied to improve generalization



\*(Note: Dataset files are not included in the repository due to size constraints.)\*



---



\## 🏗 System Architecture

1\. User uploads an image via the web interface  

2\. Image preprocessing is applied  

3\. CNN model predicts whether the image is real or AI-generated  

4\. Explainable AI module generates interpretation maps  

5\. Prediction and explanation are displayed to the user  



---



\## 🧩 CNN Model Architecture

\- Convolutional Layers for feature extraction  

\- ReLU activation functions  

\- Max Pooling layers for dimensionality reduction  

\- Fully Connected layers for classification  

\- Softmax / Sigmoid output layer  



The CNN learns visual artifacts and patterns commonly introduced by AI image generation techniques.



---



\## 🔍 Explainable AI (XAI)

To overcome the black-box nature of CNNs, Explainable AI techniques are used to:

\- Highlight image regions influencing the prediction

\- Improve trust and transparency

\- Assist users in understanding model behavior



\### XAI Techniques Used:

\- Gradient-based visualization (e.g., Grad-CAM)

\- Feature importance mapping



---



\## 🖥 Web Application

\- Built using \*\*Django Framework\*\*

\- Allows image upload through a simple UI

\- Displays prediction results and explanation outputs

\- Modular design with separate apps for detection and XAI



---



\## 🛠 Tech Stack

\- \*\*Programming Language:\*\* Python  

\- \*\*Framework:\*\* Django  

\- \*\*Deep Learning:\*\* CNN (TensorFlow / PyTorch)  

\- \*\*Explainable AI:\*\* Grad-CAM / Visualization techniques  

\- \*\*Frontend:\*\* HTML, CSS, Bootstrap  

\- \*\*Version Control:\*\* Git \& GitHub  



---



\## ▶ How to Run the Project



\### 1️⃣ Clone the Repository

```bash

git clone https://github.com/Saisathviki/Major-Project-DETECTING-AI-GENERATED-IMAGES-WITH-CNN-AND-DEEP-INTERPRETATION-USING-EXPLAINABLE-AI.git

cd ai\_detector



\## 📌 Project Overview

With the rapid advancement of generative models such as GANs and diffusion models, AI-generated images are becoming increasingly realistic and difficult to distinguish from real images. This project aims to \*\*detect AI-generated images using Convolutional Neural Networks (CNNs)\*\* and enhance trust and transparency by applying \*\*Explainable AI (XAI)\*\* techniques to interpret the model’s decisions.



The system is implemented as a \*\*Django-based web application\*\*, allowing users to upload images and receive predictions along with visual explanations.



---



\## 🎯 Problem Statement

AI-generated images can be misused for misinformation, deepfakes, identity fraud, and digital manipulation. Traditional detection methods often act as black boxes, offering no explanation for their predictions.  

This project addresses the need for:

\- Accurate detection of AI-generated images

\- Transparent and interpretable model decisions using Explainable AI



---



\## 🧠 Objectives

\- To build a CNN-based model for detecting AI-generated images  

\- To integrate Explainable AI techniques for model interpretability  

\- To develop a user-friendly Django web interface for image upload and analysis  

\- To visualize important regions influencing the model’s decision  



---



\## 🗂 Dataset Description

\- The dataset consists of \*\*real images\*\* and \*\*AI-generated images\*\*

\- Images are preprocessed (resizing, normalization)

\- Data is split into training and testing sets

\- Augmentation techniques are applied to improve generalization



\*(Note: Dataset files are not included in the repository due to size constraints.)\*



---



\## 🏗 System Architecture

1\. User uploads an image via the web interface  

2\. Image preprocessing is applied  

3\. CNN model predicts whether the image is real or AI-generated  

4\. Explainable AI module generates interpretation maps  

5\. Prediction and explanation are displayed to the user  



---



\## 🧩 CNN Model Architecture

\- Convolutional Layers for feature extraction  

\- ReLU activation functions  

\- Max Pooling layers for dimensionality reduction  

\- Fully Connected layers for classification  

\- Softmax / Sigmoid output layer  



The CNN learns visual artifacts and patterns commonly introduced by AI image generation techniques.



---



\## 🔍 Explainable AI (XAI)

To overcome the black-box nature of CNNs, Explainable AI techniques are used to:

\- Highlight image regions influencing the prediction

\- Improve trust and transparency

\- Assist users in understanding model behavior



\### XAI Techniques Used:

\- Gradient-based visualization (e.g., Grad-CAM)

\- Feature importance mapping



---



\## 🖥 Web Application

\- Built using \*\*Django Framework\*\*

\- Allows image upload through a simple UI

\- Displays prediction results and explanation outputs

\- Modular design with separate apps for detection and XAI



---



\## 🛠 Tech Stack

\- \*\*Programming Language:\*\* Python  

\- \*\*Framework:\*\* Django  

\- \*\*Deep Learning:\*\* CNN (TensorFlow / PyTorch)  

\- \*\*Explainable AI:\*\* Grad-CAM / Visualization techniques  

\- \*\*Frontend:\*\* HTML, CSS, Bootstrap  

\- \*\*Version Control:\*\* Git \& GitHub  



---



\## ▶ How to Run the Project



\### 1️⃣ Clone the Repository

```bash

git clone https://github.com/Saisathviki/Major-Project-DETECTING-AI-GENERATED-IMAGES-WITH-CNN-AND-DEEP-INTERPRETATION-USING-EXPLAINABLE-AI.git

cd ai\_detector



\### 2️⃣ Create Virtual Environment

```bash

python -m venv venv

venv\\Scripts\\activate



\### 3️⃣ Install Dependencies

```bash

pip install -r requirements.txt



\### 4️⃣ Run Migrations

```bash

python manage.py migrate



\### 5️⃣ Start Server

```bash
python manage.py runserver



Open browser and go to:

http://127.0.0.1:8000/



📊 Results



The CNN model successfully differentiates AI-generated images from real images



Explainable AI visualizations highlight key regions influencing predictions



Improved transparency and user trust in AI decisions



🚀 Future Enhancements



Support for multiple generative model detection (GANs, Diffusion, etc.)



Integration of Vision Transformers (ViT)



Improved accuracy with larger datasets



Deployment on cloud platforms



Real-time detection API



👩‍💻 Author



Sathviki

B.Tech – Final Year

Major Project



📄 License



This project is developed for academic purposes as part of a final-year major project.







