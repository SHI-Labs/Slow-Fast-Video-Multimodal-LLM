import gradio as gr
import os
import torch
import numpy as np
from decord import VideoReader, cpu

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria

from PIL import Image
import argparse

from transformers import TextIteratorStreamer
from threading import Thread

# os.environ['GRADIO_TEMP_DIR'] = './gradio_tmp'
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--server_name", default="0.0.0.0", type=str)
argparser.add_argument("--port", default="6324", type=str)
argparser.add_argument("--model-path", default="shi-labs/slowfast-video-mllm-qwen2-7b-convnext-576-frame96-s1t6", type=str)
argparser.add_argument("--model-base", type=str, default=None)
argparser.add_argument("--num-gpus", type=int, default=1)
argparser.add_argument("--conv-mode", type=str, default="qwen_1_5",)
argparser.add_argument("--temperature", type=float, default=0.2)
argparser.add_argument("--max-new-tokens", type=int, default=512)
argparser.add_argument("--num_frames", type=int, default=96)
argparser.add_argument("--load-8bit", action="store_true")
argparser.add_argument("--load-4bit", action="store_true")
argparser.add_argument("--debug", action="store_true")

args = argparser.parse_args()
model_path = args.model_path
conv_mode = args.conv_mode
max_num_frames=args.num_frames
filt_invalid="cut"
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, use_flash_attn=True)
our_chatbot = None

def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, num_threads=4)

    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, len(vr) - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames
  
def upvote_last_response(state):
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state):
    return ("",) + (disable_btn,) * 3


def flag_last_response(state):
    return ("",) + (disable_btn,) * 3

def clear_history():
    state =conv_templates[conv_mode].copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def add_text(state, video_input, textbox, image_process_mode):
    # print("add-text")
    if state is None:
        state = conv_templates[conv_mode].copy()

    if video_input is not None:
        textbox = DEFAULT_IMAGE_TOKEN + '\n' + textbox
        # image = Image.open(imagebox).convert('RGB')

    if video_input is not None:
        textbox = (textbox, video_input, image_process_mode)
        state.video_input = video_input
        # textbox = (textbox, video_input)

    state.append_message(state.roles[0], textbox)
    state.append_message(state.roles[1], None)

    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
    # yield (state, None, "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def delete_text(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def regenerate(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

# @spaces.GPU
def generate(state, video_input, textbox, image_process_mode, temperature, top_p, repetition_penalty, max_output_tokens):
    prompt = state.get_prompt()

    ori_prompt = prompt
    num_image_tokens = 0

    video = load_video(state.video_input, max_frames_num=max_num_frames)
    video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
    videos = [video_tensor]

    max_context_length = getattr(model.config, 'max_position_embeddings', 8192)
    max_new_tokens = max_output_tokens
    do_sample = True if temperature > 0.001 else False
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.to(device='cuda', non_blocking=True).unsqueeze(dim=0)    

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        # yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
        return
    
    # stop_str = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2
    stop_str = state.sep
    
    with torch.inference_mode():
        thread = Thread(target=model.generate, 
                        kwargs=dict(
                            inputs=input_ids,
                            images=videos,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            max_new_tokens=max_new_tokens,
                            streamer=streamer,
                            use_cache=True)
                        )
    
        thread.start()
        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            state.messages[-1][-1] = generated_text
            yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
    
        yield (state, state.to_gradio_chatbot(), "", None) + (enable_btn,) * 5
    
        torch.cuda.empty_cache()

txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)


title_markdown = ("""
# Slow-Fast Architecture for Video Multi-Modal Large Language Models
[[Code](https://github.com/SHI-Labs/Slow-Fast-Video-Multimodal-LLM)] | [[Model](https://huggingface.co/collections/shi-labs/slow-fast-video-mllm-67ef347a28772734c15a78b5)] | 📚 [[Arxiv](https://arxiv.org/abs/2504.01328)]
""")

attention_markdown = ("""
### ⚠️Attention⚠️
+ Please upload a video before starting a conversation. Pure text input is not supported in this demo.
+ To switch to a new video, make sure to **clear** the chat history first.
+ This is a simple demo showcasing the video understanding capabilities of our model. Some known or unknown bugs may still exist.
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the. Please contact us if you find any potential violation.
""")

block_css = """
#buttons button { 
    min-width: min(120px,100%);
}
"""

textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
with gr.Blocks(title="Slow-Fast Video MLLM", theme=gr.themes.Default(), css=block_css) as demo:
    state = gr.State()

    gr.Markdown(title_markdown)

    with gr.Row():
        with gr.Column(scale=3):
            # video_input = gr.Image(label="Input Image", type="filepath")
            video_input = gr.Video(label="Input Video")
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(examples=[
                [f"{cur_dir}/assets/cuda-introduction.mp4", "Tell me the function of the given CUDA programming example."],
                [f"{cur_dir}/assets/catinterrupt.mp4", "Provide a detailed description of this video."],
                [f"{cur_dir}/assets/marinated_salmon.mp4", "Please list all the steps to make this dish."],
                [f"{cur_dir}/assets/hero.mp4", "Why is the little girl begging on the street?"],
            ], inputs=[video_input, textbox], cache_examples=False)

            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=2048, value=1024, step=64, interactive=True, label="Max output tokens",)
                repetition_penalty = gr.Slider(minimum=1.0, maximum=1.2, value=1.1, step=0.05, interactive=True, label="Repetition penalty")

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Slow-Fast MLLM",
                height=650,
                # layout="panel",
            )
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary")
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

    gr.Markdown(attention_markdown)
    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)
    url_params = gr.JSON(visible=False)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        downvote_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        flag_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn]
    )

    clear_btn.click(
        clear_history,
        None,
        [state, chatbot, textbox, video_input] + btn_list,
        queue=False
    )

    regenerate_btn.click(
        delete_text,
        [state, image_process_mode],
        [state, chatbot, textbox, video_input] + btn_list,
    ).then(
        generate,
        [state, video_input, textbox, image_process_mode, temperature, top_p, repetition_penalty, max_output_tokens],
        [state, chatbot, textbox, video_input] + btn_list,
    )
    textbox.submit(
        add_text,
        [state, video_input, textbox, image_process_mode],
        [state, chatbot, textbox, video_input] + btn_list,
    ).then(
        generate,
        [state, video_input, textbox, image_process_mode, temperature, top_p, repetition_penalty, max_output_tokens],
        [state, chatbot, textbox, video_input] + btn_list,
    )

    submit_btn.click(
        add_text,
        [state, video_input, textbox, image_process_mode],
        [state, chatbot, textbox, video_input] + btn_list,
    ).then(
        generate,
        [state, video_input, textbox, image_process_mode, temperature, top_p, repetition_penalty, max_output_tokens],
        [state, chatbot, textbox, video_input] + btn_list,
    )

# demo.launch(share=True)
demo.queue(
    status_update_rate=10,
    api_open=False
).launch()