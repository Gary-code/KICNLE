import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,CausalLMOutputWithCrossAttentions,)
from transformers.modeling_utils import Conv1D
from typing import Tuple
from utils.eval_utils import top_filtering
from transformers import GPT2Tokenizer
import copy


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)



class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)


        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # self.transformer_1 = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        tokenizer_path = '/home/cike/Reasoning/KB-VCR/nlxgpt/pretrained_model/pretrain_tokenizer_0'
        # tokenizer_path = '/home/cike/Reasoning/KB-VCR/nlxgpt/pretrained_model/VQA-X/nle_gpt2_tokenizer_0'
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer

        # self.transformer_1.load_state_dict(self.transformer.state_dict())
        # self.transformer = copy.deepcopy(self.transformer)


        # tokenization process
        self.o_segment_id, self.k_segment_id, self.q_segment_id, self.a_segment_id, self.e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<object>',
             '<knowledge>',
             '<question>',
             '<answer>',
             '<explanation>'])
            
        SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<object>', '<knowledge>', '<question>', '<answer>', '<explanation>']
        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.because_token = self.tokenizer.convert_tokens_to_ids('Ġbecause')
        self.blank_token = self.tokenizer.convert_tokens_to_ids('Ġ')
        self.temperature = 1

        self.init_weights()

        # seq_len
        self.explanation_max_len = 20
        self.answer_max_len = 3
        self.question_max_len = 10
        self.knowledge_max_len = 0

        self.iteration_num = 2

        self.loss_fct = CrossEntropyLoss()
    

    def eval_pred(self, lm_logits, top_k=0, top_p=0.9):
        logits = lm_logits[0, -1, :] / self.temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.topk(probs, 1)[1]

        return prev

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    
    def forward(
        self,
        input_ids_init_ans, 
        labels_init_ans, 
        segment_init_ans_ids, 
        input_ids_to_ex, 
        labels_to_ex, 
        segment_to_ex_ids, 
        input_ids_to_ans, 
        labels_to_ans, 
        segment_to_ans_ids,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Q, K -> A
        transformer_outputs_init_ans = self.transformer(
            input_ids_init_ans,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=segment_init_ans_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_init_ans = transformer_outputs_init_ans[0]

        lm_logits_init_ans = self.lm_head(hidden_states_init_ans)

        # Shift so that tokens < n predict n
        shift_logits_init_ans = lm_logits_init_ans[..., :-1, :].contiguous()
        shift_labels_init_ans = labels_init_ans[..., 1:].contiguous()
        # Flatten the tokens
        loss_init_ans = self.loss_fct(shift_logits_init_ans.view(-1, shift_logits_init_ans.size(-1)), shift_labels_init_ans.view(-1))

        loss_ex = 0
        loss_ans = 0
        for i in range(self.iteration_num):
            # Q, K, A -> E
            transformer_outputs_to_ex = self.transformer(
            input_ids_to_ex,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=segment_to_ex_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )

            hidden_states_to_ex = transformer_outputs_to_ex[0]

            lm_logits_to_ex = self.lm_head(hidden_states_to_ex)

            # Shift so that tokens < n predict n
            shift_logits_to_ex = lm_logits_to_ex[..., :-1, :].contiguous()
            shift_labels_to_ex = labels_to_ex[..., 1:].contiguous()
            # Flatten the tokens
            loss_ex += self.loss_fct(shift_logits_to_ex.view(-1, shift_logits_to_ex.size(-1)), shift_labels_to_ex.view(-1))

            # Q, K, E -> A
            transformer_outputs_to_ans = self.transformer(
            input_ids_to_ans,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=segment_to_ans_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )

            hidden_states_to_ans = transformer_outputs_to_ans[0]

            lm_logits_to_ans = self.lm_head(hidden_states_to_ans)

                # Shift so that tokens < n predict n
            shift_logits_to_ans = lm_logits_to_ans[..., :-1, :].contiguous()
            shift_labels_to_ans = labels_to_ans[..., 1:].contiguous()
            # Flatten the tokens
            loss_ans += self.loss_fct(shift_logits_to_ans.view(-1, shift_logits_to_ans.size(-1)), shift_labels_to_ans.view(-1))

        loss = 0.3 * loss_init_ans + 0.7 * (loss_ex + loss_ans) / self.iteration_num
        # loss = 0.3 * loss_init_ans + 0.5 * loss_ex / self.iteration_num + 0.2 * loss_ans / self.iteration_num

        # loss_ex = self.loss_fct(shift_logits_to_ex.view(-1, shift_logits_to_ex.size(-1)), shift_labels_to_ex.view(-1))
        # loss_ans = self.loss_fct(shift_logits_to_ans.view(-1, shift_logits_to_ans.size(-1)), shift_labels_to_ans.view(-1))
        # loss = 0.2 * loss_init_ans + 0.7 * loss_ex + 0.1 * loss_ans



        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            # cross_attentions=transformer_outputs.cross_attentions,
        )


    def _sample(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Eval (batch size must be 1)
        init_prompt_len = input_ids.shape[1]  # q+k"

        # Step 1: V, Q, K -> A
        answer_fir = []  # predict answer part input ids
        answer_init = self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is"))
        answer_init = torch.tensor(answer_init, dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
        answer_segment_init = torch.tensor([self.a_segment_id] * answer_init.shape[1], dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
        segment_ids_init_answer = torch.cat((token_type_ids, answer_segment_init), dim=-1).long()
        input_ids_answer_init = torch.cat((input_ids, answer_init), dim=-1).long()
        for step in range(self.answer_max_len + 1):
            if step == self.answer_max_len:
                break
        
            transformer_outputs_initial_answer = self.transformer(
                input_ids_answer_init,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=segment_ids_init_answer,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states_init_ans = transformer_outputs_initial_answer[0]  # bs, seq_len, hidden_dim

            lm_logits_init_ans = self.lm_head(hidden_states_init_ans)  # bs, seq_len, vocab_size

            prev_ans = self.eval_pred(lm_logits_init_ans)

            if prev_ans.item() in self.special_tokens_ids:
                break
            new_segment = torch.LongTensor([self.special_tokens_ids[-2]]).to(input_ids.device).unsqueeze(dim=0)  # answer segment
            segment_ids_init_answer = torch.cat((segment_ids_init_answer, new_segment), dim=1)
            input_ids_answer_init = torch.cat((input_ids_answer_init, prev_ans.unsqueeze(0)), dim=1)
            answer_fir.append(prev_ans.item())
        
        # obtain the answer
        answer_iter_ids = torch.tensor(answer_fir).to(input_ids.device).unsqueeze(dim=0)
        for i in range(self.iteration_num):
            # Step 2: V, Q, K, A -> E
            explanation = []
            ex_head_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because")
            explanation_iter = self.tokenizer.convert_tokens_to_ids(ex_head_token)
            explanation_iter = torch.tensor(explanation_iter, dtype=torch.long).unsqueeze(dim=0).to(input_ids.device)
            ex_seg_ids = torch.tensor([self.e_segment_id] * explanation_iter.shape[1], dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
            input_ids_to_explanation = torch.cat((input_ids, answer_init[:, 1:], answer_iter_ids, explanation_iter), dim=-1).long()
            a_seg_ids = torch.tensor([self.a_segment_id] * answer_iter_ids.shape[1], dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
            segment_ids_to_explanation = torch.cat((token_type_ids, answer_segment_init[:, 1:], a_seg_ids, ex_seg_ids), dim=-1).long()

            for step in range(self.explanation_max_len + 1):
                
                if step == self.explanation_max_len:
                    break

                transformer_outputs_to_explanation = self.transformer(
                    input_ids_to_explanation,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids_to_explanation,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                lm_logits_ex = self.lm_head(transformer_outputs_to_explanation[0])

                prev_ex = self.eval_pred(lm_logits_ex)

                if prev_ex.item() in self.special_tokens_ids:
                    break
                new_segment = torch.LongTensor([self.special_tokens_ids[-1]]).to(input_ids.device).unsqueeze(dim=0)  # explanation segment
                segment_ids_to_explanation = torch.cat((segment_ids_to_explanation, new_segment), dim=1)
                input_ids_to_explanation = torch.cat((input_ids_to_explanation, prev_ex.unsqueeze(0)), dim=1)
                explanation.append(prev_ex.item())
            
            explanation_iter_ids = torch.tensor(explanation).to(input_ids.device).unsqueeze(dim=0)
            shift_logits_ex = lm_logits_ex[..., :-1, :].contiguous()  # bs, seq_len(q_len+k_len+5+a_len), vocab_size
            seq_with_ex = shift_logits_ex.argmax(dim=-1)  # bs, seq_len
            explanation_iter_ids = seq_with_ex[:, input_ids_to_explanation.shape[1]:].long()

            # Step 3: V, Q, K, E -> A
            answer = []
            ans_head_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" so the answer is")
            answer_iter = self.tokenizer.convert_tokens_to_ids(ans_head_token)
            answer_iter = torch.tensor(answer_iter, dtype=torch.long).unsqueeze(dim=0).to(input_ids.device)
            ans_seg_ids = torch.tensor([self.a_segment_id] * answer_iter.shape[1], dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
            input_ids_to_answer = torch.cat((input_ids, explanation_iter[:, 1:], explanation_iter_ids, answer_iter), dim=-1).long()
            e_seg_ids = torch.tensor([self.e_segment_id] * explanation_iter_ids.shape[1], dtype=torch.long).to(input_ids.device).unsqueeze(dim=0)
            segment_ids_to_answer = torch.cat((token_type_ids, explanation_iter[:, 1:], e_seg_ids, ans_seg_ids), dim=-1).long()
            
            for step in range(self.answer_max_len + 1):

                if step == self.answer_max_len:
                    break

                transformer_outputs_to_answer = self.transformer(
                input_ids_to_answer,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=segment_ids_to_answer,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )

                lm_logits_ans = self.lm_head(transformer_outputs_to_answer[0])

                prev_ans = self.eval_pred(lm_logits_ans)

                if prev_ans.item() in self.special_tokens_ids:
                    break
                new_segment = torch.LongTensor([self.special_tokens_ids[-2]]).to(input_ids.device).unsqueeze(dim=0)  # answer segment
                segment_ids_to_answer = torch.cat((segment_ids_to_answer, new_segment), dim=1)
                input_ids_to_answer = torch.cat((input_ids_to_answer, prev_ans.unsqueeze(0)), dim=1)
                answer.append(prev_ans.item())

                
        # directly return the a+e ids
        final_output = []
        final_output.extend(answer)
        final_output.append(self.because_token)
        # final_output.append(self.blank_token)
        final_output.extend(explanation)
        return final_output


    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )