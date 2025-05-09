import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import wandb  # Add this import
from tqdm import tqdm
import os


# Initialize the processor (handles both images and text)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def collate_fn(batch):
    images = [item['image'] for item in batch]
    captions = [item['caption'][0] if isinstance(item['caption'], list) else item['caption'] for item in batch]
    
    # Batch process both text and image
    inputs = processor(
        text=captions,
        images=images,
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs['pixel_values']
    }


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(input_dim, emb_dim)
        self.k_proj = nn.Linear(input_dim, emb_dim)
        self.v_proj = nn.Linear(input_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, input_dim)  # or emb_dim depending on your design

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.emb_dim)
        return self.out_proj(out)
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, emb_dim)
        )
    def forward(self, x):
        return self.MLP(x)
    


class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(emb_dim, emb_dim, num_heads)
        self.ff = FeedForward(emb_dim, ff_hidden_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.norm1(x), mask=mask)   # Self-attention with residual
        x = x + self.ff(self.norm2(x))                     # Feedforward with residual
        return x
    
class ImageCaptionModel(nn.Module):
    def __init__(self, num_decoder_layers, decoder_layer, emb_dim, num_heads, ff_hidden_dim, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(clip_model_name) # importing the CLIP model
        self.image_encoder = self.clip_model.vision_model # extracting the image encoder part of the CLIP model
        self.processor = CLIPProcessor.from_pretrained(clip_model_name) # extracting the image preprocessor from the CLIP model

        # Freeze CLIP image encoder parameters
        for img_param in self.image_encoder.parameters():
            img_param.requires_grad = False

        # DOUBLE CHECK THIS 
        self.text_embedding = self.clip_model.text_model.embeddings.token_embedding  # Text embedding layer
        self.text_proj = nn.Linear(512, 768)  # Project text to match image embedding dimension
        # Freeze CLIP text embedding parameters
        for text_param in self.text_embedding.parameters():
            text_param.requires_grad = False

        # Load tokenizer for the CLIP model (including BOS/EOS tokens)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # REVIEW THE DIMENSIONS HERE
        self.pos_embed = nn.Parameter(torch.randn(1, 71, emb_dim))  # 71 is max total sequence length (image + text)

        # Update decoder layers to use input_dim=emb_dim (1048)
        self.decoder = nn.ModuleList([
            decoder_layer(emb_dim, num_heads, ff_hidden_dim) for _ in range(num_decoder_layers)
        ])

        # Learnable output projection (to vocab size)
        self.output_layer = nn.Linear(emb_dim, self.tokenizer.vocab_size)

        # Add a projection layer to match the decoder's expected input dimension
        self.combined_proj = nn.Linear(768, emb_dim)  # Project combined tensor to emb_dim (1048)

    def generate_causal_mask(self, attention_mask, image_len, device):
        """
        attention_mask: (B, T_text), with 1s for real tokens, 0s for padding/post-EOS
        Returns:
            mask: (B, 1, T_total, T_total)
        """
        B, T_text = attention_mask.size()
        T_total = image_len + T_text

        # Base causal mask for text
        causal_mask = torch.tril(torch.ones(T_text, T_text, device=device))  # (T_text, T_text)

        # Apply attention mask: only allow attention to non-masked tokens
        attn_mask = attention_mask.unsqueeze(1).float()  # (B, 1, T_text)
        text_mask = causal_mask.unsqueeze(0) * attn_mask  # (B, T_text, T_text)

        # Expand to full attention mask
        full_mask = torch.ones(B, T_total, T_total, device=device)

        # Image-image and image-text remain fully visible
        full_mask[:, image_len:, image_len:] = text_mask

        return full_mask.unsqueeze(1)  # (B, 1, T_total, T_total)


    def decode(self, images, text_input_ids, attention_mask):
        # Encode image
        image_outputs = self.image_encoder(pixel_values=images).last_hidden_state  # (B, I, D)
        B, I, D = image_outputs.shape

        # Embed text
        text_embeds = self.text_embedding(text_input_ids)  # (B, T, D)
        text_embeds = self.text_proj(text_embeds)  # (B, T, D)

        # Concat [image | text] tokens
        combined = torch.cat([image_outputs, text_embeds], dim=1)  # (B, I+T, D)

        # Project combined tensor to match decoder's expected input dimension
        combined = self.combined_proj(combined)  # (B, I+T, emb_dim)

        # Generate mask using attention mask
        mask = self.generate_causal_mask(attention_mask=attention_mask, image_len=I, device=images.device)

        pos = self.pos_embed[:, :combined.size(1), :]
        x = combined + pos
        
        for layer in self.decoder:
            x = layer(x, mask=mask)

        # Project only the text part
        output_logits = self.output_layer(x[:, I:, :])  # (B, T, vocab_size)
        return output_logits
    

    @torch.no_grad()
    def generate(self, image, max_length=18, temperature=0.7):
        """
        image: preprocessed image tensor of shape (1, 3, H, W)
        Returns: list of generated token IDs
        """
        self.eval()

        generated = [self.tokenizer.bos_token_id]  # Start with BOS token

        for _ in range(max_length):
            # Create tensor from current tokens
            input_ids = torch.tensor([generated], device=image.device)
            attention_mask = torch.ones_like(input_ids, device=image.device)  # Mask for current tokens

            # Use decode function
            logits = self.decode(image, input_ids, attention_mask)  # (1, T, vocab_size)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature scaling
            scaled_logits = next_token_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:  # Use EOS token for stopping condition
                break

        return generated[1:]  # Drop BOS token


def show_image(image_tensor, ax):
    """
    Helper to display a single image.
    """
    img = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    ax.imshow(img)
    ax.axis('off')


def train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=5):
    """
    Train the ImageCaptionModel and log metrics to wandb.
    """
    # Initialize wandb
    wandb.init(project="image-captioning", name="training-run", entity="emilengdahl", config={
        "num_epochs": num_epochs,
        "batch_size": train_dataloader.batch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    }, reinit=True)

    print(f"[INFO] Starting training on {device}")
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs} begins")
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for i, batch in enumerate(progress_bar):
            # Move batch data to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model.decode(pixel_values, input_ids[:, :-1], attention_mask[:, :-1])  # Teacher forcing
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # Log metrics to wandb
            wandb.log({"batch_loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"[INFO] Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Log epoch loss to wandb
        wandb.log({"epoch": epoch + 1, "epoch_loss": avg_epoch_loss})

        # Save the most recent model weights to wandb, overwriting the previous one
        model_path = "latest_model.pth"
        torch.save(model.state_dict(), model_path)
        artifact = wandb.Artifact("image_caption_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()


@torch.no_grad()
def evaluate_model(model, test_dataloader, criterion, device):
    """
    Evaluate the ImageCaptionModel.
    """
    print(f"\n[INFO] Starting evaluation on {device}")
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    progress_bar = tqdm(test_dataloader, desc="Evaluating")
    
    for i, batch in enumerate(progress_bar):
        # Move batch data to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)

        if i == 0:
            print(f"[DEBUG] Evaluation input_ids shape: {input_ids.shape}")
        
        # Forward pass
        logits = model.decode(pixel_values, input_ids[:, :-1], attention_mask[:, :-1])
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"[INFO] Evaluation completed. Average Loss: {avg_loss:.4f}")
    return avg_loss


# Example usage
def main():
    # Prompt user for mode of operation
    mode = input("Enter mode (train/finetune/eval/example): ").strip().lower()
    if mode not in ["train", "finetune", "eval", "example"]:
        print("[ERROR] Invalid mode. Please enter 'train', 'finetune', 'eval', or 'example'.")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Check if preprocessed datasets exist
    if os.path.exists("train_dataset") and os.path.exists("test_dataset"):
        print("[INFO] Loading preprocessed datasets from disk...")
        train_dataset = load_from_disk("train_dataset")
        test_dataset = load_from_disk("test_dataset")
    else:
        print("[INFO] Preprocessing datasets...")
        # Load the dataset
        dataset = load_dataset("nlphuji/flickr30k", split="test[:25000]")

        # Split into train and test sets
        train_test = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test['train']
        test_dataset = train_test['test']

        # Apply the preprocessing function to the train and test datasets
        #train_dataset = train_dataset.map(preprocess_function, batched=True)
        #test_dataset = test_dataset.map(preprocess_function, batched=True)

        # Save the processed datasets to disk
        train_dataset.save_to_disk("train_dataset")
        test_dataset.save_to_disk("test_dataset")

    # Create the DataLoader for both train and test datasets
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Initialize model
    model = ImageCaptionModel(
        num_decoder_layers=6,
        decoder_layer=TransformerDecoderBlock,
        emb_dim=1048,
        num_heads=8,
        ff_hidden_dim=1048
    )

    if mode == "finetune":
        print("[INFO] Loading model weights from latest_model.pth for fine-tuning...")
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    if mode == "train":
        # Train the model from scratch
        train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=25)
    elif mode == "finetune":
        # Fine-tune the model
        train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=10)
    elif mode == "eval":
        # Evaluate the model
        print("[INFO] Loading model weights from latest_model.pth for evaluation...")
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))
        evaluate_model(model, test_dataloader, criterion, device)
    elif mode == "example":
        # Load the model weights for generating captions
        print("[INFO] Loading model weights from latest_model.pth for generating captions...")
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))
        # proceeding to creating sample images

    # Generate captions for example images
    model.eval()
    model.to(device)

    # Load a batch from the test dataloader
    batch = next(iter(test_dataloader))
    images = batch['pixel_values'][6:12]  # Select 6 images
    images = images.to(device)

    # Generate captions and display images
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(6):
        img = images[i].unsqueeze(0)  # Shape: (1, 3, 224, 224)

        # Generate caption
        with torch.no_grad():
            token_ids = model.generate(img)
            caption = model.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Show image and caption
        ax = axs[i // 3, i % 3]
        show_image(images[i], ax)
        ax.set_title(caption, fontsize=10)

    plt.tight_layout()

    # Save the final example images
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "example_images.png")
    plt.savefig(output_path)
    print(f"[INFO] Example images saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
