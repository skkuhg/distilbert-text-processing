# DistilBERT Base Cased - Text Processing Model

This repository contains a Jupyter notebook demonstrating the use of DistilBERT, a distilled version of BERT (Bidirectional Encoder Representations from Transformers), for masked language modeling and text embedding generation.

## Overview

DistilBERT is a smaller, faster, and lighter version of BERT that retains 97% of BERT's language understanding while being 60% faster and 40% smaller in size. This project demonstrates both the cased and uncased variants of DistilBERT.

## Features

- **Fill-Mask Pipeline**: Uses DistilBERT to predict masked tokens in sentences
- **Word Embeddings**: Generates contextual word embeddings for text processing
- **GPU Support**: Configured to run on CUDA-enabled GPUs for faster inference
- **Easy Integration**: Simple examples using Hugging Face Transformers library

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- CUDA-compatible GPU (optional, but recommended)

## Installation

Install the required dependencies:

```bash
pip install -U transformers
```

For GPU support, ensure you have PyTorch with CUDA installed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Fill-Mask Task

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="distilbert/distilbert-base-cased")
result = pipe("Hello I'm a [MASK] model.")

for candidate in result:
    print(candidate)
```

### Generating Word Embeddings

```python
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# Access the embeddings
embeddings = output.last_hidden_state
```

### Direct Model Loading

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-cased")
```

## Notebook Contents

The [Distilbert-base-cased.ipynb](Distilbert-base-cased.ipynb) notebook includes:

1. **Installation**: Setting up the Transformers library
2. **Pipeline Usage**: High-level API for fill-mask tasks
3. **Direct Model Loading**: Lower-level API for custom implementations
4. **Embedding Generation**: Creating contextual word embeddings
5. **Token Visualization**: Inspecting tokenization results

## Models Used

- **distilbert-base-cased**: DistilBERT model trained on cased English text
- **distilbert-base-uncased**: DistilBERT model trained on lowercased English text

Model pages:
- [distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased)
- [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)

## Example Output

When running the fill-mask task with "Hello I'm a [MASK] model.", the model predicts:

1. fashion (15.75%)
2. professional (6.04%)
3. role (2.56%)
4. celebrity (1.94%)
5. model (1.73%)

## Use Cases

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Identifying entities in text
- **Question Answering**: Building QA systems
- **Text Embeddings**: Feature extraction for downstream tasks
- **Language Understanding**: Transfer learning for NLP tasks

## Performance

DistilBERT offers an excellent trade-off between performance and efficiency:

- **Speed**: 60% faster than BERT
- **Size**: 40% smaller than BERT
- **Performance**: Retains 97% of BERT's capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If the code snippets do not work, please open an issue on:
- [Model Repository](https://huggingface.co/distilbert/distilbert-base-cased)
- [Hugging Face.js](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face**: For the Transformers library and pre-trained models
- **DistilBERT Authors**: Sanh et al. for the DistilBERT research and implementation

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [DistilBERT Model Card](https://huggingface.co/distilbert/distilbert-base-cased)

## Contact

For questions or feedback, please open an issue in this repository.
