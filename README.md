Neural Machine Translation (English → Arabic)
Developed by: Hamza Alawneh & Nabil Al-Halabi

1. Project Overview (General Summary)
This project is a complete End-to-End NLP Pipeline designed to translate text from English to Arabic. We followed a professional AI engineering workflow, starting from raw data cleaning to fine-tuning a Transformer model (MarianMT) and finally deploying a user-friendly interface.

2. Requirements Analysis vs. Implementation (What we have done)
Based on the official project requirements (Task A to Task E), here is what we successfully implemented:

 Task A: Data Preparation (Preprocessing)
Implementation: Developed in data_prep.py.

What we did: * Loaded the parallel dataset (ara_translation.csv).

Cleaning: Removed empty rows and exact duplicates.

Length Filtering: Implemented a rule to remove sentences that are too long to ensure model stability and prevent memory issues (VRAM).

Splitting: Divided the data into 80% Train, 10% Validation, and 10% Test using a fixed random seed for reproducibility.

 Task B: Tokenization
Implementation: Verified in tokenizer_inspect.py.

What we did: * Used Option 1 (Hugging Face Pretrained Tokenizer) from the MarianMT family.

Sub-word Tokenization: Justified this choice because it handles the complex morphology of the Arabic language and solves the "Out-of-Vocabulary" (OOV) problem.

Evidence: The script outputs the vocabulary size and provides live tokenization examples for 3 sentences.

 Task C: Building the Transformer Model
Implementation: Developed in train.py.

What we did: * Followed Track 1 (Fine-tuning) using the Helsinki-NLP/opus-mt-en-ar model.

Implemented the training loop using Seq2SeqTrainer with specific hyperparameters (Learning rate, Batch size, Epochs) managed via config.yaml.

Model Saving: The system automatically saves the best-performing model and its tokenizer.

 Task D: Evaluation & Error Analysis
Implementation: Developed in evaluate.py and error_analysis.py.

What we did: * Quantitative Metrics: Used BLEU and chrF scores to measure accuracy.

Qualitative Analysis: This is our project's strength. error_analysis.py generates a report identifying:

Under-translation: Missing content.

Over-translation: Repeated phrases.

Length Ratio Analysis: Comparing source vs. target lengths.

 Task E: Presentation & Demo
Implementation: Developed in demo_gradio.py.

What we did: Built a web-based UI using Gradio, where a user can type English text and get an instant Arabic translation using our fine-tuned model.

3. Technical Highlights (Our Engineering Decisions)
Modular Design: We separated the logic into distinct files linked by a central config.yaml. This makes the project production-ready and easy to scale.

Reproducibility: By using utils.py to set the global seed, we ensure that anyone running the code will get the exact same results.

Arabic-Specific Evaluation: We chose chrF as a primary metric because it is more robust for Arabic's rich morphology compared to standard BLEU.

Team Contribution (Acknowledgment)
As we discussed, this project was a highly collaborative effort. Nabil Al-Halabi played a crucial role as the lead developer, handling the core architecture, training logic, and the advanced error analysis reports, while I focused on the data engineering, testing, and integration.
