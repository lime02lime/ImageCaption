# Image Captioning with Transformer Decoder and CLIP

This project implements an image captioning model using a Transformer-based decoder architecture and the CLIP model for image encoding. The notebook `model.ipynb` contains the code for preprocessing, model definition, training, and evaluation.

## Overview

The notebook is structured as follows:
1. **Data Loading and Preprocessing**: 
   - The Flickr30k dataset is loaded and split into training and testing sets.
   - Images and captions are preprocessed using the CLIP processor to prepare them for the model.

2. **Model Architecture**:
   - The model uses the CLIP vision encoder to extract image embeddings.
   - A Transformer-based decoder autoregressively generates captions from the image embeddings and tokenized text inputs.
   - The decoder includes:
     - Multi-head self-attention layers.
     - Feedforward layers.
     - Positional embeddings for sequence modeling.

3. **Training**:
   - The model is trained using teacher forcing, where the ground truth captions are used as input during training.
   - The loss is computed using cross-entropy, and metrics are logged using `wandb`.

4. **Evaluation**:
   - The model is evaluated on the test set, and the average loss is reported.
   - Captions are generated for sample images to visually inspect the model's performance.

5. **Visualization**:
   - The notebook includes code to display images alongside their generated captions.

## Model Architecture

The `ImageCaptionModel` consists of the following components:
1. **CLIP Vision Encoder**:
   - Pretrained CLIP model's vision encoder is used to extract image embeddings.
   - The encoder's parameters are frozen during training.

2. **Transformer Decoder**:
   - A stack of Transformer decoder blocks is used to generate captions.
   - Each block includes:
     - Multi-head self-attention for modeling dependencies within the sequence.
     - Feedforward layers for feature transformation.
     - Layer normalization and residual connections for stability.

3. **Text Embedding and Projection**:
   - The CLIP text embedding layer is used to embed tokenized captions.
   - A projection layer aligns the text embedding dimensions with the decoder's input dimensions.

4. **Positional Embeddings**:
   - Learnable positional embeddings are added to the input sequence to encode positional information.

5. **Output Layer**:
   - A linear layer projects the decoder's output to the vocabulary size for token prediction.

## Training and Evaluation

- **Training**:
  - The model is trained for 16 epochs using the AdamW optimizer.
  - Metrics such as batch loss and epoch loss are logged to `wandb`.

- **Evaluation**:
  - The model is evaluated on the test set, and captions are generated for sample images.
  - The generated captions are compared with the ground truth to assess performance.

## Usage

1. **Preprocessing**:
   - Run the preprocessing cells to prepare the dataset.

2. **Training**:
   - Execute the training cells to train the model.

3. **Evaluation**:
   - Use the evaluation cells to generate captions for test images.

4. **Visualization**:
   - Visualize the generated captions alongside the corresponding images.

## Dependencies

- Python
- PyTorch
- Transformers
- Datasets
- Matplotlib
- WandB

## Example Output

The model generates captions for images, such as:

| Image | Generated Caption |
|-------|-------------------|
| ![Image](example1.png) | "A man riding a bicycle on a city street." |
| ![Image](example2.png) | "A group of people sitting at a table outdoors." |

## Future Work

- Fine-tune the CLIP encoder for improved performance.
- Experiment with larger datasets and more complex architectures.
- Evaluate the model using BLEU or CIDEr scores for quantitative analysis.
