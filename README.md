[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/jedick/AI4citations)
[![Codecov test coverage](https://codecov.io/gh/jedick/AI4citations/graph/badge.svg)](https://app.codecov.io/gh/jedick/AI4citations)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# AI4citations: Automated Citation Verification

The integrity of scientific literature depends on citations that are supported by the referenced source material. These citations are sometimes inaccurate, contributing to unverified claims. AI4citations provides an easy-to-use solution for automated citation verification that leverages state-of-the-art machine learning models trained on domain-specific datasets.

## 🎯 Use Cases

- **Academic Researchers**: Verify citations in literature reviews and research papers
- **Journal Editors**: Automated fact-checking during peer review process  
- **Students**: Learn proper citation practices and evidence evaluation
- **Science Communicators**: Verify claims in popular science writing
- **Fact-checkers**: Quick verification of scientific claims in media

## 🚀 Quick Start

### Try the App Online
**No installation required!** Use AI4citations directly in your browser:

👉 **[Launch AI4citations on Hugging Face Spaces](https://huggingface.co/spaces/jedick/AI4citations)**

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jedick/AI4citations.git
   cd AI4citations
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (optional, for GPT retrieval)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Launch the application**
   ```bash
   gradio app.py
   ```

5. **Access the app**
   - Open your browser and navigate to the displayed URL (typically `http://127.0.0.1:7860`)
   - Upload a PDF or input text directly to start verifying citations

## 📖 How to Use

1. **Input a claim** (hypothesis) you want to verify
2. **Provide evidence** in one of two ways:
   - Upload a PDF and use automatic evidence retrieval
   - Manually input evidence text
3. **Get predictions** with confidence scores for:
   - **Support**: Evidence supports the claim
   - **Refute**: Evidence contradicts the claim  
   - **NEI** (Not Enough Information): Evidence is insufficient
4. **Provide feedback** to help improve the model

![Screenshot of AI4citations app](./images/AI4citations_screenshot.png)

## 🔗 Related Projects

This app is part of a comprehensive ML engineering ecosystem:

- **🏗️ [MLE Capstone Project](https://github.com/jedick/MLE-capstone-project)** - Complete ML pipeline with baselines, evaluation, and deployment
- **📦 [pyvers Package](https://github.com/jedick/pyvers)** - Python package for training claim verification models
- **🤖 [Fine-tuned Model](https://huggingface.co/jedick/DeBERTa-v3-base-mnli-fever-anli-scifact-citint)** - Production model on Hugging Face

## ⚡ Key Features

### Claim Verification Models
- **Fine-tuned DeBERTa** (default): Trained on SciFact and Citation-Integrity datasets for scientific claim verification
- **Base DeBERTa**: Pre-trained on multiple natural language inference (NLI) datasets
- **Interactive model switching**: Compare results between different models
- **Detailed predictions**: Get instant results with confidence scores

### Evidence Retrieval Methods
Choose from three complementary approaches to extract relevant evidence from PDFs:

- 🔍 **BM25S** (Traditional keyword matching with BM25 ranking)
- 🧠 **DeBERTa** (AI-based question-answering with context extraction)
- 🤖 **OpenAI GPT** (Advanced AI: Large language model with document understanding)

For BM25S and DeBERTa, you can adjust the number of evidence sentences retrieved (top-k sentences).

### User Experience Features
- **Interactive examples**: Pre-loaded examples for each prediction class
- **PDF upload**: Drag-and-drop PDF processing
- **Responsive design**: Works on desktop and mobile devices
- **GPU acceleration**: Optimized for fast inference on Hugging Face Spaces
- **Token usage tracking**: Monitor OpenAI API usage
- **Real-time feedback collection**: Help improve the model with your corrections

**[Click here to see the collected feedback dataset!](https://huggingface.co/datasets/jedick/AI4citations-feedback)**

## 📊 Performance Evaluation

Benchmarked on the SciFact test set with gold evidence as baseline:

| Retrieval Method | Macro F1 | Speed (avg.) | Best Use Case |
|------------------|----------|--------------|---------------|
| **Gold evidence** | 0.834 | - | Baseline (human-selected) |
| **BM25S** | 0.649 | 0.36s | Fast keyword matching |
| **DeBERTa** | 0.610 | 7.00s | Semantic understanding |
| **GPT** | 0.615 | 19.84s | Complex reasoning |

The fine-tuned model achieves a **7 percentage point improvement** over single-dataset baselines through multi-dataset training.

## 🛠️ Technical Architecture

### Core Components
- **Frontend**: Gradio interface with custom styling and Font Awesome icons
- **Backend**: PyTorch Lightning with Hugging Face Transformers
- **PDF Processing**: PyMuPDF (fitz) with text cleaning and normalization
- **Retrieval**: Multiple engines (BM25S, DeBERTa QA, OpenAI GPT)
- **Deployment**: Hugging Face Spaces with GPU acceleration
- **CI Testing**: GitHub Actions workflow for integration and unit tests

### Data Pipeline
1. **PDF Text Extraction**: Multi-page processing with layout preservation
2. **Text Normalization**: Unicode conversion, hyphen removal, sentence tokenization
3. **Evidence Retrieval**: Method-specific processing (keyword, QA, or LLM-based)
4. **Claim Verification**: Transformer-based classification with confidence scores
5. **Feedback Loop**: User corrections saved for continuous improvement

## 📚 Datasets

The model was trained and evaluated on two high-quality datasets for claim verification in biomedical and health sciences:

### SciFact
- **Size**: 1,409 scientific claims verified against 5,183 abstracts
- **Source**: [AllenAI SciFact Dataset](https://github.com/allenai/scifact)

### Citation-Integrity  
- **Size**: 3,063 citation instances from biomedical publications
- **Source**: [Citation-Integrity Dataset](https://github.com/ScienceNLP-Lab/Citation-Integrity/)

Both datasets were normalized with consistent labeling for robust cross-domain performance.

## 🙏 Acknowledgments

This project builds upon exceptional work from the research and open-source communities:

### Core Technologies
- **[Gradio](https://github.com/gradio-app/gradio)**: Web interface framework enabling easy ML app deployment
- **[Hugging Face Transformers](https://huggingface.co/transformers/)**: State-of-the-art transformer models and tokenizers
- **[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)**: Scalable ML training framework

### Models and Datasets
- **[DeBERTa](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)**: Base model pre-trained on multiple NLI datasets by MoritzLaurer
- **[SciFact Dataset](https://github.com/allenai/scifact)**: Scientific claim verification dataset by Wadden et al. (2020)
- **[Citation-Integrity Dataset](https://github.com/ScienceNLP-Lab/Citation-Integrity/)**: Biomedical citation verification by Sarol et al. (2024)

### Retrieval Technologies
- **[BM25S](https://github.com/xhluca/bm25s)**: High-performance BM25 implementation for keyword-based retrieval
- **[PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)**: Robust PDF text extraction and processing
- **[OpenAI GPT](https://platform.openai.com/docs/pricing)**: Advanced language model for complex reasoning tasks

### Development Tools
- **[NLTK](https://www.nltk.org/)**: Natural language processing utilities for tokenization
- **[Unidecode](https://github.com/avian2/unidecode)**: Unicode to ASCII text conversion
- **[Codecov](https://codecov.io/)**: Test coverage reporting and monitoring
- **AI Assistance**: BERT retrieval code developed with assistance from Claude Sonnet 4

### Research Foundations
- **MultiVerS Model**: Longformer-based claim verification by Wadden et al. (2021)
- **Natural Language Inference**: Foundational NLI datasets (MultiNLI, FEVER, ANLI)
- **Domain Adaptation**: Cross-dataset training techniques for improved generalization

For detailed technical information and experimental results, see the [ML Engineering Capstone Project](https://github.com/jedick/MLE-capstone-project) repository and associated [blog posts](https://jedick.github.io/blog/experimenting-with-transformer-models-for-citation-verification/).

---

**💡 Questions or Issues?** Open an issue on GitHub!
