"""
This code generates the desired number NUM_IMAGES of images using a SD, SD + LoRA, SD3 or SD3 + LoRA model.
Please change the CONFIGURATION variables accordingly.
The script will store the generated images at colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}_gen/.

"""

############################################## IMPORTS ###############################################################################
import os
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
import random

device = 'cuda' if torch.cuda.is_available else 'cpu'

SEED = 42

####################################### CONFIGURATION ##############################################################

TRIAL_NUMBER = 20                  # number corresponding to the DB model chosen (see gridsearch_DB.py)
                                  
DISEASE = 'esantema-maculo-papuloso'
FLAG = 'SD_LORA'
BLACK_OR_BROWN = 'brown'
DISEASE_UNIQUE_ID = 'blckskn6' if BLACK_OR_BROWN == 'black' else 'brwnskn6'
PROMPT = f"Very black {DISEASE_UNIQUE_ID} human skin" if BLACK_OR_BROWN == "black" else f"Brown {DISEASE_UNIQUE_ID} human skin."
NEGATIVE_PROMPT = "eye, white, violet, blue, pale skin, fair skin, light skin, white skin"
GUIDANCE_SCALE = 7.5
NUM_IMAGES = 600

####################################################################################################################

if FLAG == "SD_LORA":
   from diffusers import DiffusionPipeline
   from diffusers import DPMSolverMultistepScheduler
   pipe = StableDiffusionPipeline.from_pretrained(f"runwayml/stable-diffusion-v1-5", safety_checker = None)
   pipe.load_lora_weights(f"colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}/dreambooth-model{TRIAL_NUMBER}")
   pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

elif FLAG == "SD3_LORA":
   from diffusers import StableDiffusion3Pipeline
   pipe = StableDiffusion3Pipeline.from_pretrained(f"stabilityai/stable-diffusion-3-medium-diffusers", safety_checker=None)
   pipe.load_lora_weights(f"colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}/dreambooth-model{TRIAL_NUMBER}")

elif FLAG == "SD":
   # Create a StableDiffusionControlNetPipeline using the pre-trained models
   pipe = StableDiffusionPipeline.from_pretrained(
          f"colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}/dreambooth-model{TRIAL_NUMBER}",
          safety_checker = None,
          torch_dtype=torch.float16
          )

   # Use UniPCMultistepScheduler with the pipeline
   pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

elif FLAG == "SD3":
   from diffusers import StableDiffusion3Pipeline
   pipe = StableDiffusion3Pipeline.from_pretrained(
          f"colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}/dreambooth-model{TRIAL_NUMBER}",
          safety_checker=None)

pipe.to(device)
generator = torch.Generator(device=device).manual_seed(SEED)
random.seed(SEED)

image_number_start = 0

num_range = NUM_IMAGES // 40

with torch.autocast(device):
    # GENERATE!
    for i in range(num_range + 1):
          NUM_IMAGES_PIPE = 40
          images = pipe(PROMPT, num_inference_steps=50, height=512, width=512, guidance_scale=GUIDANCE_SCALE,\
                    num_images_per_prompt=NUM_IMAGES_PIPE, negative_prompt=NEGATIVE_PROMPT, \
                    generator=generator).images

          generator=torch.Generator(device=device).manual_seed(SEED+i)

          for image_number, image in enumerate(images):
              image.resize((256,256))
              bl_br = 'bl' if BLACK_OR_BROWN=='black' else 'br'
              image_n = image_number_start + image_number
              image.save(f"colab/DB/{DISEASE}/{FLAG}/{BLACK_OR_BROWN}_gen/{image_n:05}_{TRIAL_NUMBER}_{bl_br}.png")

          image_number_start += i*NUM_IMAGES_PIPE