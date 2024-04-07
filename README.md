# MiniGPT: Text Generation with Prompt Injection

This project contains the development and training of a MiniGPT model, a compact version of the Generative Pretrained Transformer designed to operate within resource-constrained environments. The MiniGPT model is capable of generating textual responses based on previously seen text, equivalent to input from the `input.txt` file.

The MiniGPT was trained on two distinct datasets: `TinyShakespeare.txt`, featuring the poetic and archaic language of Shakespeare's works, and `ConservativeVoicesUSA.txt`, a compilation of modern conservative dialogue from Joe Rogan podcasts. These datasets presented unique challenges and learning opportunities for the model.

## Project Structure
- `FinalModelCorrect.ipynb`: Contains the MiniGPT model, which focuses on training a machine learning model for text generation, learning from input text to generate contextually relevant and stylistically similar outputs.
- `Spellchecker.ipynb`: Implements a spell checking utility that identifies misspelled words in a text file and calculates the percentage of correctly spelled words. Methods have all been moved to `FinalModelCorrect.ipynb`

## Getting Started

### Prerequisites

- Python 3.9
- PyTorch
- pyspellchecker
- tiktoken (Note: May encounter compatibility issues with macOS systems.)

### Installation

Install the necessary Python packages by running:

```bash
pip install torch==2.1.0
pip install pyspellchecker==0.8.1
pip install tiktoken==0.6.0
```
Python version 3.9 must be running. It is recommended to make a virtual enviornment for this project to avoid negatively interacting python packages.

## Prompt Injection for Model Text Continuation

This is in `FinalModelCorrect.ipynb`.

To initiate text generation and guide the model in continuing the text based on a given prompt,

```python
encoded_text = encoding.encode("Test")
```

This line encodes the prompt "" using the model's encoding method, preparing it for the text generation process. By changing the prompt, we allowing the model to generate text that continues from the prompt that is provided.

## Hardware Requirements and Model Parameters

This model has been developed with consideration for computational resource constraints. Training a model of this nature, one that exceeds 100M parameters, requires computational resources, ideally a high-capacity GPU. Our experiments leveraged GPUs such as NVIDIA RTX A6000 and NVIDIA H100, indicating the need for significant computational power.

**Note**: Users with less powerful hardware should be cautious of potential memory limitations and may need to adjust model parameters or utilize external compute resources.

## Choosing the Tokenizer

The `FinalModelCorrect.ipynb` script allows users to select the tokenizer best suited for their specific text processing needs. Supported tokenization methods include character, Byte-Pair Encoding (BPE), and word, which can be easily chosen within the code. This feature enables users to tailor the model's preprocessing step to optimize performance based on the characteristics of the input data.

### Mac Compatibility

The `tiktoken` package, used for Byte-Pair Encoding (BPE) tokenization, has known compatibility issues with macOS systems. An alternative option is to choose character-wise tokenization. Users attempting to run this project on a Mac may need to seek alternative tokenization libraries or run the code within a Docker container or virtual machine that emulates a compatible environment. 

## Results

#### Key Observations:

- **Adaptability**: The MiniGPT demonstrated an ability to adapt to the vastly different linguistic styles of the datasets. It was able to mimic the poetic structure and Middle English vocabulary from TinyShakespeare, as well as capture the conversational tone and opinions discussed in the ConservativeVoicesUSA dataset.
- **Tokenization Impact**: The choice of Byte-Pair Encoding (BPE) for tokenization significantly influenced the model's performance. BPE allowed for a more nuanced understanding of the text structure, leading to better generation quality, albeit at the cost of increased model complexity.
- **Resource Constraints**: The experiments highlighted the balance between model performance and computational resource demands. Training models with parameters nearing 100M required substantial GPU capabilities, emphasizing the importance of resource availability in achieving desired outcomes.

### Results and Performance Metrics
The training and validation losses of our final model are shown in the image below. The generated losses were based on a model trained on the ConservativeVoicesUSA dataset.

![TrainValLoss](https://github.com/AnjaKroon/MiniGPT/assets/154330044/c3e8eb8d-7db9-4b88-833a-a4a2bccc43cb)

#### Performance Metrics:

- **Spelling Accuracy**: Achieved a spelling accuracy of 80%. This metric was particularly notable given the diverse and complex language structures the model was exposed to.
- **Model Complexity**: The model configurations tested varied in complexity, with the chosen setup involving over 300 million parameters. This scale facilitated the model's learning capabilities but also necessitated the use of powerful GPUs for training.
- **Text Generation Quality**: Qualitative assessments of generated text highlighted the model's proficiency in creating content that was stylistically and thematically aligned with the input datasets. Generated text samples exhibited clear influences from the training data, including the emulation of Shakespearean language and the reflection of conservative viewpoints from the ConservativeVoicesUSA dataset.

