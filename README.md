![Afbeelding van WhatsApp op 2024-04-07 om 21 08 55_4ef44b13](https://github.com/AnjaKroon/MiniGPT/assets/154330044/09c6a973-39cc-4cbc-9479-4db5f5ade04e)
# MiniGPT: Text Generation on Constrained Resources

This project contains the development and training of a MiniGPT model, a compact version of the Generative Pretrained Transformer designed to operate within resource-constrained environments. The MiniGPT model is capable of generating textual responses based on previously seen text, equivalent to input from the `input.txt` file.

## Project Structure

- `Spellchecker.py`: Implements a spell checking utility that identifies misspelled words in a text file and calculates the percentage of correctly spelled words.
- `FinalModelCorrect.py`: Contains the MiniGPT model, which focuses on training a machine learning model for text generation, learning from input text to generate contextually relevant and stylistically similar outputs.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- pyspellchecker
- tiktoken (Note: May encounter compatibility issues with macOS systems.)

### Installation

Install the necessary Python packages by running:

```bash
pip install torch pyspellchecker
# Note: usage of tiktoken may fail on macOS, consider alternatives if necessary.
pip install tiktoken
```

## Usage

### Spellchecker.py

To use the spell checker, simply run the script. Ensure you have an `input.txt` and a 'Generated.txt' file in the same directory.

```bash
python Spellchecker.py
```

### FinalModelCorrect.py

To train the MiniGPT model and generate text, ensure you have an `input.txt` file in the same directory.

```bash
python FinalModelCorrect.py
```

The script will train the model using the specified dataset and then generate text based on learned patterns.

## Prompt Injection for Model Text Continuation

To initiate text generation and guide the model in continuing the text based on a given prompt, the following line is used:

```python
encoded_text = encoding.encode("Test")
```

This line encodes the prompt "" using the model's encoding method, preparing it for the text generation process. By changing the prompt, we allowing the model to generate text that continues from the prompt that is provided.

## Hardware Requirements and Model Parameters

This model has been developed with consideration for computational resource constraints. Training a model of this nature, one that exceeds 100M parameters, requires computational resources, ideally a high-capacity GPU. Our experiments leveraged GPUs such as NVIDIA RTX A6000 and NVIDIA H100, indicating the need for significant computational power.

**Note**: Users with less powerful hardware should be cautious of potential memory limitations and may need to adjust model parameters or utilize external compute resources.

### Mac Compatibility

The `tiktoken` package, used for Byte-Pair Encoding (BPE) tokenization, has known compatibility issues with macOS systems. Users attempting to run this project on a Mac may need to seek alternative tokenization libraries or run the code within a Docker container or virtual machine that emulates a compatible environment.

### Observations and Findings from the Experiments

Our experiments with the MiniGPT model yielded insightful observations regarding text generation capabilities under constrained computational resources. The MiniGPT was trained on two distinct datasets: TinyShakespeare, featuring the poetic and archaic language of Shakespeare's works, and ConservativeVoicesUSA, a compilation of modern conservative dialogue. These datasets presented unique challenges and learning opportunities for the model.

#### Key Observations:

- **Adaptability**: The MiniGPT demonstrated a remarkable ability to adapt to the vastly different linguistic styles of the datasets. It was able to mimic the poetic structure and Middle English vocabulary from TinyShakespeare, as well as capture the conversational tone and contemporary issues discussed in the ConservativeVoicesUSA dataset.
- **Tokenization Impact**: The choice of Byte-Pair Encoding (BPE) for tokenization significantly influenced the model's performance. BPE allowed for a more nuanced understanding of the text structure, leading to better generation quality, albeit at the cost of increased model complexity.
- **Resource Constraints**: The experiments highlighted the balance between model performance and computational resource demands. Training models with parameters nearing 100M required substantial GPU capabilities, emphasizing the importance of resource availability in achieving desired outcomes.

### Results and Performance Metrics
The training and validation losses of our final model are shown in the image below. The generated losses were based on a model trained on the ConservativeVoicesUSA dataset.

![TrainValLoss](https://github.com/AnjaKroon/MiniGPT/assets/154330044/c3e8eb8d-7db9-4b88-833a-a4a2bccc43cb)


Step	Train Loss	Validation Loss
0	    13,3283	    13,3301
500	    5,1164	    5,5664
1000	4,6390	    5,0758
1500	4,2554	    4,7383
2000	3,9126	    4,3932
2500	3,5102	    3,9965
3000	3,1091	    3,6826
3500	2,7662	    3,2322
4000	2,3922	    2,8583
4500	2,1043	    2,5301
4999	1,8565	    2,2026


#### Performance Metrics:

- **Spelling Accuracy**: Achieved a spelling accuracy of 78.34%. This metric was particularly notable given the diverse and complex language structures the model was exposed to.
- **Model Complexity**: The model configurations tested varied in complexity, with the chosen setup involving over 300 million parameters. This scale facilitated the model's learning capabilities but also necessitated the use of powerful GPUs for training.
- **Text Generation Quality**: Qualitative assessments of generated text highlighted the model's proficiency in creating content that was stylistically and thematically aligned with the input datasets. Generated text samples exhibited clear influences from the training data, including the emulation of Shakespearean language and the reflection of conservative viewpoints from the ConservativeVoicesUSA dataset.

#### Insights:

The experiments underscore the feasibility of deploying smaller-scale GPT models for specialized text generation tasks, balancing performance with computational resource constraints. The findings also illuminate the critical role of tokenization in model performance and the inherent trade-offs between model complexity and resource availability.

## Choosing the Tokenizer

The `FinalModelCorrect.py` script is designed with flexibility in mind, allowing users to select the tokenizer best suited for their specific text processing needs. Supported tokenization methods include character, Byte-Pair Encoding (BPE), and word, which can be easily chosen within the code. This feature enables users to tailor the model's preprocessing step to optimize performance based on the characteristics of the input data.
