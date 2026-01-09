markdown
Copy code
# Neural Machine Translation (English → Arabic)
**Developed by:** Hamza Alawneh & Nabil Al-Halabi

---

## **Project Overview (General Summary)**
This project is a complete End-to-End NLP Pipeline designed to translate text from English to Arabic. We followed a professional AI engineering workflow, starting from raw data cleaning to fine-tuning a Transformer model (MarianMT) and finally deploying a user-friendly interface.

---

## **Requirements Analysis vs. Implementation**  
### *(What we have done)*  
Based on the official project requirements (Task A to Task E), here is what we successfully implemented:

---

## **Task A: Data Preparation (Preprocessing)**  
**Implementation:** `data_prep.py`

### **What we did:**
- Loaded the parallel dataset (`ara_translation.csv`).
- **Cleaning:** Removed empty rows and exact duplicates.
- **Length Filtering:** Implemented rules to remove long sentences for model stability and to prevent VRAM issues.
- **Splitting:** Divided the data into **80% Train**, **10% Validation**, **10% Test** using a fixed random seed.

---

## **Task B: Tokenization**  
**Implementation:** `tokenizer_inspect.py`

### **What we did:**
- Used **Option 1 (Hugging Face Pretrained Tokenizer)** from the MarianMT family.
- **Sub-word Tokenization:** Selected because it handles Arabic’s complex morphology and reduces OOV issues.
- **Evidence:** Script prints vocabulary size + tokenization examples for 3 sentences.

---

## **Task C: Building the Transformer Model**  
**Implementation:** `train.py`

### **What we did:**
- Followed **Track 1 (Fine-tuning)** using `Helsinki-NLP/opus-mt-en-ar`.
- Implemented the training loop using **Seq2SeqTrainer**, with hyperparameters (LR, batch size, epochs) managed via `config.yaml`.
- **Model Saving:** Automatically stores best-performing model + tokenizer.

---

## **Task D: Evaluation & Error Analysis**  
**Implementation:** `evaluate.py` & `error_analysis.py`

### **What we did:**
- **Quantitative Metrics:** Used **BLEU** and **chrF**.
- **Qualitative Analysis:**  
  `error_analysis.py` generates a detailed report detecting:
  - **Under-translation** (missing content)  
  - **Over-translation** (repetition)  
  - **Length Ratio Analysis** (source vs. target)

---

## **Task E: Presentation & Demo**  
**Implementation:** `demo_gradio.py`

### **What we did:**
- Built a **Gradio Web UI** for instant English → Arabic translation using our fine-tuned model.

---

## **Technical Highlights (Our Engineering Decisions)**
- **Modular Design:** Clear separation via multiple scripts + `config.yaml` for production-grade structure.
- **Reproducibility:** Global seed control in `utils.py` ensures identical results for all runs.
- **Arabic-Specific Evaluation:** Selected **chrF** as a primary metric for its robustness with Arabic morphology.

---

## **Team Contribution (Acknowledgment)**
This project was a highly collaborative effort.  
- **Nabil Al-Halabi**: Lead developer — core architecture, training logic, and advanced error analysis.  
- **Hamza Alawneh**: Data engineering, testing, and system integration.

---
