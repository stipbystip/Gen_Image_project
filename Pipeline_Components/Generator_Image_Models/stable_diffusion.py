from safetensors import safe_open
from PIL import Image
import torch
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler, EulerDiscreteScheduler
from Interfaces_Pipeline_Components.IGeneratorImage import ImageGenerator


class StableDiffusion(ImageGenerator):
    def __init__(self):
        self.model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(self.model_name,
                                                 subfolder="vae",
                                                 use_safetensors=True,
                                                 torch_dtype=torch.float32)

        self.unet = UNet2DConditionModel.from_pretrained(self.model_name,
                                                         subfolder='unet',
                                                         use_safetensors=True,
                                                         torch_dtype=torch.float32)

        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.model_name,
                                                                 subfolder="scheduler",
                                                                 beta_start=0.00085,
                                                                 beta_end=0.012,
                                                                 prediction_type='epsilon'
                                                                 )

    def apply_lora(self, lora_path):
        with safe_open(lora_path, framework="pt") as f:
            lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        self.unet.load_state_dict(lora_weights, strict=False)

    def decode_images(self, latents):
        latents = 1 / 0.13025 * latents  # Было 1 / 0.18215 0.13025
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float32):
                image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image

    def generate(self, embedding_data, lora_path=None):
        if lora_path:
            self.apply_lora(lora_path)
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_data_tensor = torch.tensor(embedding_data, device=torch_device)
        self.vae.to(torch_device, memory_format=torch.channels_last)
        self.unet.to(torch_device, memory_format=torch.channels_last)

        height, width = 512, 512
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8), device=torch_device, dtype=torch.float32
        ) * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(400)
        guidance_scale = 7.5

        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=embedding_data_tensor,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        result_images = self.decode_images(latents)
        return result_images
