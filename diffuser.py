from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# from PIL import Image
import torch


class diffuse_generator:
    def __init__(self, repo_id: str='stabilityai/stable-diffusion-2-base'):
        self.repo_id = repo_id
        pipe = DiffusionPipeline.from_pretrained(self.repo_id, torch_dtype=torch.float16, revision="fp16")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")


if __name__ == '__main__':
    gen = diffuse_generator()

    # prompt = "golden retriever"
    prompt = 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea'
    images = gen.pipe(prompt, num_inference_steps=25).images

    for i in range(len(images)):
        # image = images[i].resize((8, 8))
        image = images[i]
        image.save(f"golden_{i}.png")
