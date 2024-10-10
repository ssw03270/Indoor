# accelerate launch --num_processes 1  scripts/models/diffusion_velocity_field/train_unet2d_condition.py --train_data_dir "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/train" --val_data_dir "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/val"

import os
import pickle

from dataclasses import dataclass
from datasets import Dataset

import numpy as np
import torch

from diffusers import DDPMScheduler
from PIL import Image
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path

import argparse  # Import argparse for command-line argument parsing

def load_local_pkl_files(file_dir):
    data = []
    # Load all pkl files in the directory.
    for file_name in os.listdir(file_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(file_dir, file_name)
            with open(file_path, 'rb') as f:
                data_item = pickle.load(f)
                data.append(data_item)  # Add each pkl file's data to the list
    return data

# Convert local dataset to a Dataset object
def load_custom_dataset(data_dir):
    data = load_local_pkl_files(data_dir)
    
    for item in data:
        # Ensure 'gt_velocity_field' is a float tensor
        if not isinstance(item['gt_velocity_field'], torch.Tensor):
            item['gt_velocity_field'] = torch.tensor(item['gt_velocity_field'], dtype=torch.float32)
        # Ensure 'image_condition' is a float tensor and normalize it
        if not isinstance(item['image_condition'], torch.Tensor):
            item['image_condition'] = torch.tensor(item['image_condition'], dtype=torch.float32)
        # Normalize 'image_condition' to [0, 1] (assuming original range is [0, 255])
        item['image_condition'] = item['image_condition'] / 255.0
        # Ensure 'image_condition' shape is (C, H, W)
        if item['image_condition'].shape[0] != 3:
            item['image_condition'] = item['image_condition'].permute(2, 0, 1)
        # Ensure 'gt_velocity_field' shape is (C, H, W)
        if item['gt_velocity_field'].ndim == 2:
            item['gt_velocity_field'] = item['gt_velocity_field'].unsqueeze(0)  # Add channel dimension if missing
    dataset = Dataset.from_list(data)
    return dataset

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm_64'  # the model name locally and on the HF Hub

    push_to_hub = False  # Set to False if not pushing to Hugging Face Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the script
    seed = 0

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a diffusion model.')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--val_data_dir', type=str, required=True, help='Path to the validation data directory.')
    args = parser.parse_args()

    config = TrainingConfig()

    # Local dataset load path for training
    train_data_dir = args.train_data_dir

    # Load training dataset
    train_dataset = load_custom_dataset(train_data_dir)
    train_dataset.set_format(type='torch')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

    # Load validation dataset
    val_data_dir = args.val_data_dir
    val_dataset = load_custom_dataset(val_data_dir)
    val_dataset.set_format(type='torch')

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

    from transformers import CLIPTextModel, AutoTokenizer

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    from diffusers import UNet2DConditionModel

    # Update in_channels to 5 (2 channels for 'gt_velocity_field' + 3 channels for 'image_condition')
    model = UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=5,  # Updated from 2 to 5
        out_channels=2,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(160, 320, 640, 640),  # the number of output channels for each downsampling block
        down_block_types=( 
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D", 
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",  
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D"
        ),
        norm_num_groups=32,  # normalization groups set to 32
        cross_attention_dim=512  # set this to match the text encoder's hidden size
    )

    noise_scheduler_1000 = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler_50 = DDPMScheduler(num_train_timesteps=50)


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    from diffusers.optimization import get_cosine_schedule_with_warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
   
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    text_encoder = accelerator.prepare(text_encoder)
    text_encoder.requires_grad_(False)

    # Start training
    train_loop(
        config, model, noise_scheduler_1000, noise_scheduler_50, optimizer,
        train_dataloader, val_dataloader, lr_scheduler, tokenizer, text_encoder, accelerator
    )

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def evaluate(config, epoch, model, tokenizer, text_encoder, accelerator, val_dataloader,
             noise_scheduler_1000, noise_scheduler_50):
    model.eval()
    device = accelerator.device

    test_dir = os.path.join(config.output_dir, "samples", f"epoch_{epoch:03d}")
    os.makedirs(test_dir, exist_ok=True)

    idx = 0  # Initialize sample index

    total_samples = len(val_dataloader.dataset)
    progress_bar = tqdm(total=total_samples, desc="Evaluating", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch_size = batch['gt_velocity_field'].size(0)
            scene_ids = batch.get('scene_id', [f"scene_{i}" for i in range(idx, idx + batch_size)])
            clean_images_batch = batch['gt_velocity_field'].to(device)
            image_conditions_batch = batch['image_condition'].to(device)
            text_conditions = batch['text_condition']

            for i in range(batch_size):
                scene_id = scene_ids[i] if isinstance(scene_ids, list) else scene_ids[i].item()
                save_dir = os.path.join(test_dir, f"{scene_id}")
                os.makedirs(save_dir, exist_ok=True)

                clean_image = clean_images_batch[i:i+1]
                image_condition = image_conditions_batch[i:i+1]
                text_condition = [text_conditions[i]]

                # Prepare the text conditioning
                input_ids = tokenizer(
                    text_condition,
                    padding="longest",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(input_ids.input_ids.to(device)).last_hidden_state

                # Starting from random noise
                generator = torch.manual_seed(config.seed)
                sample = torch.randn(
                    (1, model.config.in_channels - 3, config.image_size, config.image_size),
                    generator=generator,
                ).to(device)

                # Select the appropriate noise scheduler
                if idx < 10:
                    noise_scheduler = noise_scheduler_1000
                else:
                    noise_scheduler = noise_scheduler_50

                noise_scheduler.set_timesteps(noise_scheduler.num_train_timesteps)
                timesteps = noise_scheduler.timesteps

                # Sampling process
                for t in timesteps:
                    combined_input = torch.cat([sample, image_condition], dim=1)

                    noise_pred = model(
                        combined_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False
                    )[0]

                    sample = noise_scheduler.step(noise_pred, t, sample).prev_sample

                # Convert sample to image
                generated_sample = sample.clamp(0, 1)
                generated_image = generated_sample[:, 0:1, :, :]
                generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()
                generated_image = Image.fromarray((generated_image.squeeze() * 255).astype(np.uint8), mode='L')

                # Apply rotation and flipping to generated_image
                generated_image = generated_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

                # Process ground truth image
                gt_image = clean_image[:, 0:1, :, :].cpu().permute(0, 2, 3, 1).numpy()
                gt_image = Image.fromarray((gt_image.squeeze() * 255).astype(np.uint8), mode='L')
                gt_image = gt_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)

                # Convert image_condition to PIL image
                img_condition = image_condition.cpu().numpy()
                img_condition = img_condition.transpose(0, 2, 3, 1)
                img_condition = Image.fromarray((img_condition.squeeze() * 255).astype(np.uint8))

                # Save the images and condition immediately
                generated_image.save(os.path.join(save_dir, f"generated_{idx:04d}.png"))
                gt_image.save(os.path.join(save_dir, f"ground_truth_{idx:04d}.png"))
                img_condition.save(os.path.join(save_dir, f"image_condition_{idx:04d}.png"))
                with open(os.path.join(save_dir, f"text_condition_{idx:04d}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text_condition[0])

                idx += 1
                progress_bar.update(1)

    progress_bar.close()

def train_loop(config, model, noise_scheduler_1000, noise_scheduler_50, optimizer,
               train_dataloader, val_dataloader, lr_scheduler, tokenizer, text_encoder, accelerator):
    from huggingface_hub import HfFolder, Repository, whoami
    from pathlib import Path

    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        if accelerator.is_main_process:
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        else:
            progress_bar = None

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['gt_velocity_field'].to(accelerator.device)
            image_condition = batch['image_condition'].to(accelerator.device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler_1000.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler_1000.add_noise(clean_images, noise, timesteps)

            # Concatenate 'noisy_images' and 'image_condition' along the channel dimension
            combined_input = torch.cat([noisy_images, image_condition], dim=1)  # shape (bs, 5, H, W)

            text_condition = batch['text_condition']
            input_ids = tokenizer(
                text_condition,
                padding="longest",
                truncation=True,  # Truncate input sequences to max length
                max_length=tokenizer.model_max_length,  # Explicitly set max length
                return_tensors="pt",
            )
            encoder_hidden_states = text_encoder(input_ids["input_ids"].to(accelerator.device)).last_hidden_state

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(combined_input, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.is_main_process:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            global_step += 1

        if accelerator.is_main_process:
            progress_bar.close()

        # After each epoch, optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(
                    config, epoch, accelerator.unwrap_model(model), tokenizer, text_encoder, accelerator,
                    val_dataloader, noise_scheduler_1000, noise_scheduler_50
                )
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    # Save the model
                    model.save_pretrained(config.output_dir)
                    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
