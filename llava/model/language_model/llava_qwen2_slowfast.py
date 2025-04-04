#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

import transformers
# from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.hybrid_decoder_layer import Qwen2DecoderLayer, Qwen2HybridDecoderLayer

class LlavaQwenSlowFastConfig(Qwen2Config):
    model_type = "llava_qwen_slow_fast"


class LlavaQwenSlowFastModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenSlowFastConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenSlowFastModel, self).__init__(config)
        
        # initialize the cross-attention layers
        self.slow_branch_is_initialized = False
        
        if hasattr(config, "cross_attn_every_n_layers"):
            self.initialize_slow_branch_modules(config)
    
    def initialize_slow_branch_modules(self, args):
        if self.slow_branch_is_initialized:
            return 
        # number of decoder layers
        num_layers = len(self.layers) 
        
        cross_attn_every_n_layers = args.cross_attn_every_n_layers
        cross_attn_gating_type = args.cross_attn_gating_type
        cross_attn_implementation = args.cross_attn_implementation
        cross_attn_max_layer_depth = getattr(args, "cross_attn_max_layer_depth", num_layers)
        cross_attn_min_layer_depth = getattr(args, "cross_attn_min_layer_depth", 0)
        if cross_attn_max_layer_depth is None:
            cross_attn_max_layer_depth = num_layers
        if cross_attn_min_layer_depth is None:
            cross_attn_min_layer_depth = 0
        
        self.config.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.config.cross_attn_implementation = cross_attn_implementation
        self.config.cross_attn_gating_type = cross_attn_gating_type
        self.config.cross_attn_max_layer_depth = cross_attn_max_layer_depth
        self.config.cross_attn_min_layer_depth = cross_attn_min_layer_depth
        
        # set pooling operations
        tile_image_input = getattr(args, "tile_image_input", True) # tile all the image input into a video sequence
        min_fast_frames = getattr(args, "min_fast_frames", 1)   # force to sample at least `min_fast_frames` frames for fast visual tokens
        if min_fast_frames is None:
            min_fast_frames = 1
        
        self.config.tile_image_input = tile_image_input
        self.config.min_fast_frames = min_fast_frames
        
        # generate layer index for the hybrid layer
        hybrid_layer_idx = []
        for i in range(cross_attn_min_layer_depth, cross_attn_max_layer_depth, cross_attn_every_n_layers):
            hybrid_layer_idx.append(i)
        
        # substitute the original decoder layer with hybrid layer
        initialize_kv_from_lm = getattr(args, "initialize_cross_attn_kv_from_lm", False) # whether use LLM's pretrained kv projection to initialize the kv projection weight of cross-attn
        for idx in range(len(self.layers)):
            if idx in hybrid_layer_idx:
                original_decoder_layer = self.layers[idx]
                hybrid_decoder_layer = Qwen2HybridDecoderLayer(self.config, layer_idx=idx, is_hyper_enabled=True, cross_attn_gating_type=cross_attn_gating_type, cross_attn_implementation=cross_attn_implementation)
                _, unexpected_keys = hybrid_decoder_layer.load_state_dict(original_decoder_layer.state_dict(), strict=False) # cause problem when using deepspeed zero3
                if initialize_kv_from_lm and hasattr(hybrid_decoder_layer.self_attn, "cross_attn_kv_proj"):
                    kv_weight = torch.cat([original_decoder_layer.self_attn.k_proj.weight,
                                           original_decoder_layer.self_attn.v_proj.weight], dim=0)
                    kv_bias = torch.cat([original_decoder_layer.self_attn.k_proj.bias,
                                           original_decoder_layer.self_attn.v_proj.bias], dim=0)
                    new_state_dict = OrderedDict()
                    new_state_dict['weight'] = kv_weight
                    new_state_dict['bias'] = kv_bias
                    hybrid_decoder_layer.self_attn.cross_attn_kv_proj.load_state_dict(new_state_dict)
                assert len(unexpected_keys) == 0
                self.layers[idx] = hybrid_decoder_layer
        
        # fast token config
        self.config.fast_token_spatial_stride = args.fast_token_spatial_stride
        self.config.fast_token_temporal_stride = args.fast_token_temporal_stride
        self.config.fast_token_temporal_sampling_stride = args.fast_token_temporal_sampling_stride
        
        self.slow_branch_is_initialized = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                # logger.warning_once(
                #     "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                #     "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                #     "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                # )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if not isinstance(decoder_layer, Qwen2HybridDecoderLayer):
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class LlavaQwenSlowFastForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenSlowFastConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_slow_fast"
        config.rope_scaling = None

        self.model = LlavaQwenSlowFastModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Qwen2HybridDecoderLayer):
            module.gradient_checkpointing = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def sample_fast_frames(self,
                      total_frames, 
                      stride,
                      min_frame_number):
    
        all_indices_list = list(range(total_frames))
        
        if total_frames < min_frame_number:
            return all_indices_list
        
        sampled_frames = max(total_frames // stride, min_frame_number)
        stride = total_frames / sampled_frames
        
        fast_indices = [min(int(i * stride), total_frames-1) for i in range(sampled_frames)]
    
        return fast_indices

    def split_slow_fast_tokens(self, 
                               visual_tokens, 
                               temporal_sampling_stride=1,
                               spatial_stride=1,
                               temporal_stride=1):
        # TODO: Min: this function is very messy and can be simplified.
        if isinstance(visual_tokens, torch.Tensor):
            # for all image inputs, only perform spatial pooling 
            b, n, c = visual_tokens.shape
            h = w = int(n**0.5)
            fast_visual_tokens = nn.functional.avg_pool2d(visual_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2),
                                                          kernel_size=spatial_stride, 
                                                          stride=spatial_stride).flatten(2,3).transpose(1,2)            
            return fast_visual_tokens, visual_tokens
        else:
            fast_visual_tokens = []
            for sample_ in visual_tokens:
                t, n, c = sample_.shape
                if t > 1: # is a video
                    T_downsampling_rate = temporal_sampling_stride * temporal_stride
                    
                    if t % T_downsampling_rate != 0:
                        padding_size = (T_downsampling_rate - t % T_downsampling_rate) % T_downsampling_rate
                        # Pad on the first dimension (sequence length) with zeros
                        sample_ = nn.functional.pad(sample_, (0, 0, 0, 0, 0, padding_size))  # (dim_pad_left, dim_pad_right, T_pad_left, T_pad_right)
                    
                    # 1. temporal direct sampling
                    if temporal_sampling_stride > 1:
                        fast_token_indices = self.sample_fast_frames(total_frames=t,
                                                    stride=temporal_sampling_stride,
                                                    min_frame_number=self.config.min_fast_frames)
                    else:
                        fast_token_indices = list(range(t))
                    
                    sample_ = torch.stack([sample_[idx] for idx in fast_token_indices], dim=0)     
                    b, n, c = sample_.shape
                    h = w = int(n**0.5)     
                    sample_ = sample_.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    
                    # 2. temporal average pooling
                    if temporal_stride > 1:
                        if (sample_.shape[0] // temporal_stride) >= self.config.min_fast_frames:
                            sample_ = nn.functional.avg_pool3d(sample_.transpose(0, 1), kernel_size=(temporal_stride, 1, 1)).transpose(0, 1)
                        else:
                            h_, w_ = sample_.shape[-2:]
                            output_frames_num = min(sample_.shape[0], self.config.min_fast_frames)
                            sample_ = nn.functional.adaptive_avg_pool3d(sample_.transpose(0, 1), output_size=(output_frames_num, h_, w_)).transpose(0, 1)
                        
                    # 3. spatial pooling
                    if spatial_stride > 1:
                        sample_ = nn.functional.avg_pool2d(sample_,
                                                        kernel_size=spatial_stride, 
                                                        stride=spatial_stride)
                    sample_ = sample_.flatten(2,3).transpose(1,2)
                
                else:
                    if spatial_stride > 1:
                        h = w = int(n**0.5)     
                        sample_ = sample_.reshape(t, h, w, c).permute(0, 3, 1, 2)
                        sample_ = nn.functional.avg_pool2d(sample_,
                                                        kernel_size=spatial_stride, 
                                                        stride=spatial_stride)
                        sample_ = sample_.flatten(2,3).transpose(1,2)
                        
                fast_visual_tokens.append(sample_.flatten(0, 1).contiguous())
            slow_visual_tokens = [_.flatten(0, 1).contiguous() for _ in visual_tokens]
            
            return fast_visual_tokens, slow_visual_tokens

                
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # clear the visual tokens if current one is a pure text sample
            if images is None and input_ids.shape[1] > 1:
                for layer in self.get_decoder().layers:
                        if hasattr(layer, "clear_vis_x"):
                            layer.clear_vis_x()
            
            token_types = torch.ones_like(input_ids, dtype=input_ids.dtype, device=input_ids.device)
            for layer in self.get_decoder().layers:
                if hasattr(layer, "condition_vis_x"):
                    layer.media_locations = token_types
            
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # handle image input
        images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images] # list [ [T, C, H, W], ]
        feature_split_size = [x.shape[0] for x in images]
        all_features, feature_split_size = self.encode_images(torch.cat(images, dim=0), feature_split_size)

        raw_image_features = torch.split(all_features, feature_split_size, dim=0)
        image_features = []
        for sample_feat in raw_image_features:   # initial spatial pooling for all video tokens
            if sample_feat.shape[0] > 1 and self.config.mm_video_pooling_stride > 1:
                b, n, c = sample_feat.shape
                h = w = int(n**0.5)                    
                sample_feat = nn.functional.avg_pool2d(sample_feat.reshape(b, h, w, c).permute(0, 3, 1, 2),
                                                        kernel_size=self.config.mm_video_pooling_stride, 
                                                        stride=self.config.mm_video_pooling_stride).flatten(2,3).transpose(1,2)         
            image_features.append(sample_feat.contiguous())
        del raw_image_features, all_features
        
        ## generate fast and slow tokens
        image_features, slow_image_features = self.split_slow_fast_tokens(image_features,
                                                                        temporal_sampling_stride=self.config.fast_token_temporal_sampling_stride,
                                                                        spatial_stride=self.config.fast_token_spatial_stride,
                                                                        temporal_stride=self.config.fast_token_temporal_stride)
        
        ## set cross-attention states
        if isinstance(slow_image_features, (list, tuple)):
            padded_tensors = torch.nn.utils.rnn.pad_sequence(slow_image_features, batch_first=True)
            cross_attn_mask = torch.ones(padded_tensors.shape[:-1], dtype=torch.bool, device=padded_tensors.device)
            for i, tensor in enumerate(slow_image_features):
                cross_attn_mask[i, len(tensor):] = False  # Mark padded elements as False  
            slow_image_features = padded_tensors
        else:
            cross_attn_mask = torch.ones(slow_image_features.shape[:-1], dtype=torch.bool, device=slow_image_features.device)
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        new_token_types = [] 
        # NOTE: Min: we need to record the type of tokens so that we can split the tokens in the hybrid decoder layer
        # Token type 1: user's input and system tokens, 2: response text tokens, 3: visual tokens, 4: invalid tokens (padding)
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                
                cur_token_type = torch.full((cur_input_ids.shape[0],), 2, dtype=cur_input_ids[-1].dtype, device=cur_input_ids[-1].device) 
                cur_token_type[labels[batch_idx] == IGNORE_INDEX] = 1 # token with ignore tokens are considered as user input 
                new_token_types.append(cur_token_type)
                
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_token_type_noim = []
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                
                cur_token = torch.full((cur_labels_noim[-1].shape[0],), 2, dtype=cur_input_ids_noim[-1].dtype, device=cur_input_ids_noim[-1].device) 
                cur_token[cur_labels[image_token_indices[i]+1:image_token_indices[i+1]] == IGNORE_INDEX] = 1 # ingored tokens are considered as user input 
                cur_token_type_noim.append(cur_token) 
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_token_type = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_token_type.append(cur_token_type_noim[i])
                
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_token_type.append(torch.full((cur_image_features.shape[0],), 3, device=cur_labels.device, dtype=cur_labels.dtype)) # insert image token type

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_token_type = torch.cat(cur_new_token_type) ##

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_token_types.append(cur_new_token_type) ##
            
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_token_types = [x[:tokenizer_model_max_length] for x in new_token_types]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_token_types_padded = torch.full((batch_size, max_len), 4, dtype=new_labels[0].dtype, device=new_labels[0].device) ## 4 is invalid token type (padding)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_token_types_padded[i, -cur_len:] = new_token_types[i] ## 
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_token_types_padded[i, :cur_len] = new_token_types[i]

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # token type
        token_types = new_token_types_padded
        # send token type to cross-attn layers
        if _input_ids is not None and _input_ids.shape[-1] == 1:
            pass
        else:
            if slow_image_features is not None:
                for layer in self.get_decoder().layers:
                    if hasattr(layer, "condition_vis_x"):
                        layer.condition_vis_x(slow_image_features, 
                                              cross_attn_mask,
                                              token_type=token_types)
            else:
                for layer in self.get_decoder().layers:
                    if hasattr(layer, "clear_vis_x"):
                        layer.clear_vis_x() 

        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


AutoConfig.register("llava_qwen_slow_fast", LlavaQwenSlowFastConfig)
AutoModelForCausalLM.register(LlavaQwenSlowFastConfig, LlavaQwenSlowFastForCausalLM)