# Brain Tumor Classification & Explainable AI (XAI) System
### Capstone Project | Bachelor of Science in Computer Engineering
**Author:** Momen Albawarid  
**Institution:** Al Hussein Technical University (HTU)  

---

## 🧬 The Research Journey: From Theory to Tool
This project was developed in two distinct phases: an initial academic research study followed by a full software engineering implementation.

### Phase 1: The Research Phase (CRP)
Before writing a single line of the app code, I conducted a deep dive into medical imaging AI. 
* **The Goal:** To determine which CNN architectures (Xception, ResNet, MobileNet, etc.) could handle the "Domain Shift" between different medical datasets.
* **Key Finding:** I discovered that while heavy data augmentation is standard in AI, it can actually *harm* medical MRI models because flipping or rotating brain scans can distort critical anatomical orientations. 
* **The Breakthrough:** Image enhancement (contrast normalization) was found to be more effective than geometric augmentation for this specific task.
* **Documentation:** You can read the original research paper here: [`research/CRP - Momen Albawarid.pdf`](./research/)

### Phase 2: The Engineering Phase (Capstone)
Using the findings from Phase 1, I transitioned from a research notebook to a production-ready application.
* **Optimization:** Unified the dataset strategy and implemented Class Weighting to reach **93% accuracy** (up from the 77% baseline in the research phase).
* **Interpretability:** Integrated **Score-CAM** to solve the "Black Box" problem, ensuring that the AI provides a visual heatmap of its decision-making process.
* **Architecture:** Developed a multi-threaded desktop GUI using **CustomTkinter** to provide a professional experience for medical practitioners.

---

## 🛠️ Technical Specifications
* **Architecture:** Fine-tuned MobileNetV2.
* **Explainable AI:** Score-Weighted Class Activation Mapping (Score-CAM).
* **Interface:** CustomTkinter (Dark Mode Optimized).
* **Concurrency:** Background threading for non-blocking AI inference.

## 📂 Repository Contents
* `src/`: Core Python implementation.
* `models/`: Trained model weights (`.h5` - 21MB).
* `research/`: Contains the **Research Paper (CRP)** and the **Final Capstone Report**.
* `test_suite.py`: Automated unit tests for data integrity.

## 📊 Results Summary
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Glioma** | 0.94 | 0.96 | 0.95 |
| **Meningioma** | 0.90 | 0.86 | 0.88 |
| **Pituitary** | 0.96 | 0.98 | 0.97 |
| **No Tumor** | 0.92 | 0.95 | 0.93 |

## 💻 Installation & Usage
1. **Clone & Install:**
   ```bash
   git clone [https://github.com/MomenAlbawaridGit/CapstoneProject_MomenAlbawarid.git](https://github.com/MomenAlbawaridGit/CapstoneProject_MomenAlbawarid.git)
   pip install customtkinter tensorflow numpy pillow opencv-python
