import argparse
import torch
import os
import numpy as np
from decord import VideoReader, cpu

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.utils import disable_torch_init


def load_video(video_path, max_frames_num):
        vr = VideoReader(video_path, num_threads=4)
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]

        uniform_sampled_frames = np.linspace(0, len(vr) - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        return spare_frames
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)    

    qs = args.question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # read and process video
    video_path = args.video_path
    video = load_video(video_path, max_frames_num=args.max_frames)
    video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
    videos = [video_tensor]
                        
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(dim=0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images =videos,
            do_sample = True if args.temperature > 0 else False,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"User input: {args.question}\n")
    print(outputs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="Flying-Lynx/prerelease-slowfast-videomllm-7B-frame64-s1t4")
    parser.add_argument("--model-path", type=str, default="Flying-Lynx/prerelease-slowfast-videomllm-7B-frame96-s1t6")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--video_path", type=str, default="assets/catinterrupt.mp4")
    parser.add_argument("--question", type=str, default="Please describe this video in detail.")
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
