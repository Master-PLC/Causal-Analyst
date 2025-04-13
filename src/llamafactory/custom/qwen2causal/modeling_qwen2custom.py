from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import Qwen2Config, Qwen2Model, Qwen2PreTrainedModel
from transformers.utils import ModelOutput

from ..daggnn import DAGGNN


class Qwen2CustomConfig(Qwen2Config):

    model_type = "qwen2custom"

    def __init__(
        self, vocab_size=151936, hidden_size=4096, intermediate_size=22016, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, hidden_act="silu", max_position_embeddings=32768, initializer_range=0.02, rms_norm_eps=1e-6,
        use_cache=True, tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None, use_sliding_window=False,
        sliding_window=4096, max_window_layers=28, attention_dropout=0.0, num_classes=5, num_feature_nodes=38, training_stage=1,
        forbidden_edges_filepath=None, positive_required_edges_filepath=None, negative_required_edges_filepath=None,
        feature_names=None, feature_subset=None, alignment_type='additive', x_dim=1, z_dim=1, graph_threshold=0.3, tau_A=0.0, 
        lambda_A=0.0, c_A=1.0, use_A_connect_loss=False, use_A_positiver_loss=False, encoder_hidden_size=128, 
        decoder_hidden_size=128, encoder_dropout=0.0, decoder_dropout=0.0, recursive_dag_search=False, cls_lambda=1.0,
        align_lambda=1.0, elbo_lambda=1.0, rec_lambda=1.0, **kwargs,
    ):
        self.num_classes = num_classes
        self.num_feature_nodes = num_feature_nodes

        assert training_stage in [1, 2, 3]
        self.training_stage = training_stage

        # Forbidden Prior Knowledge
        self.forbidden_edges_filepath = forbidden_edges_filepath
        # Required Prior Knowledge
        self.positive_required_edges_filepath = positive_required_edges_filepath
        self.negative_required_edges_filepath = negative_required_edges_filepath

        self.feature_names = feature_names
        self.feature_subset = feature_subset

        self.alignment_type = alignment_type
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.graph_threshold = graph_threshold
        self.tau_A = tau_A
        self.lambda_A = lambda_A
        self.c_A = c_A

        self.use_A_connect_loss = use_A_connect_loss
        self.use_A_positiver_loss = use_A_positiver_loss

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        self.recursive_dag_search = recursive_dag_search

        self.cls_lambda = cls_lambda
        self.align_lambda = align_lambda
        self.elbo_lambda = elbo_lambda
        self.rec_lambda = rec_lambda

        super().__init__(
            vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings, initializer_range=initializer_range, rms_norm_eps=rms_norm_eps,
            use_cache=use_cache, tie_word_embeddings=tie_word_embeddings, rope_theta=rope_theta, rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window, sliding_window=sliding_window, max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            **kwargs,
        )


@dataclass
class CausalAnalystOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    classes: Optional[torch.FloatTensor] = None
    losses: Optional[Dict[str, float]] = None


class Qwen2Custom(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.num_feature_nodes = config.num_feature_nodes

        self.cls_lambda = config.cls_lambda
        self.align_lambda = config.align_lambda
        self.elbo_lambda = config.elbo_lambda

        self.model = Qwen2Model(config)
        self.head_cls = nn.Linear(config.hidden_size, self.num_classes, bias=False)

        self.head_feat = nn.Linear(config.hidden_size, self.num_feature_nodes, bias=False)
        self.head_align = nn.Linear(1, self.config.x_dim, bias=False) if self.config.x_dim > 1 else nn.Identity()

        self.graph_learner = DAGGNN(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_causal_graph(self):
        try:
            return self.graph_learner.auto_get_dag()
        except AttributeError:
            return self.graph_learner.modules_to_save.default.auto_get_dag()
        except Exception as e:
            raise RuntimeError(f"Failed to get causal graph: {e}")

    def refine_graph(self):
        try:
            return self.graph_learner.graph_refinement()
        except AttributeError:
            return self.graph_learner.modules_to_save.default.graph_refinement()
        except Exception as e:
            raise RuntimeError(f"Failed to refine graph: {e}")

    def forward(
        self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
        classes: Optional[torch.FloatTensor] = None, features: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalAnalystOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # [B, L, D]

        batch_size = input_ids.shape[0]

        # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        if self.config.padding_side == "right":
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(hidden_states.device)
        else:
            sequence_lengths = -1

        if self.config.training_stage == 1:
            logits = self.head_cls(hidden_states)  # [B, L, C]
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [B, C]

            classes = classes.to(logits.device).float()
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(pooled_logits, classes)
            losses = None

        elif self.config.training_stage == 2:
            latent_feature_score = self.head_feat(hidden_states)  # [B, L, F]
            pooled_latent_feature_score = latent_feature_score[torch.arange(batch_size, device=latent_feature_score.device), sequence_lengths]  # [B, F]

            prompt_feature_score = features[:, self.config.feature_subset] if self.config.feature_subset else features
            if self.training:
                prompt_feature_score = torch.concat([prompt_feature_score, classes], dim=-1)  # [B, F]
            else:
                with torch.no_grad():
                    logits = self.head_cls(hidden_states)  # [B, L, C]
                    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [B, C]
                    pooled_logits = torch.sigmoid(pooled_logits)
                prompt_feature_score = torch.concat([prompt_feature_score, pooled_logits], dim=-1)  # [B, F]

            # align the prompt feature score with the latent feature score
            if self.config.alignment_type == "additive":
                prompt_feature = prompt_feature_score + pooled_latent_feature_score
            elif self.config.alignment_type == "multiplicative":
                prompt_feature = prompt_feature_score * pooled_latent_feature_score
            elif self.config.alignment_type == "attentive":
                score = prompt_feature_score * pooled_latent_feature_score
                score = F.softmax(score, dim=-1)
                prompt_feature = pooled_latent_feature_score * score + prompt_feature_score

            prompt_feature = self.head_align(prompt_feature.unsqueeze(-1))  # [B, F, X]
            graph_losses = self.graph_learner(prompt_feature)

            loss_align = F.mse_loss(prompt_feature_score, pooled_latent_feature_score)
            graph_losses["loss_align"] = loss_align.item()
            graph_losses["loss_cls"] = 0

            loss = graph_losses.pop("loss") + self.align_lambda * loss_align
            pooled_logits = None
            classes = None
            losses = torch.tensor([list(graph_losses.values())], device=loss.device)

        elif self.config.training_stage == 3:  # for evaluation
            logits = self.head_cls(hidden_states)  # [B, L, C]
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [B, C]

            classes = classes.to(logits.device).float()
            loss_fct = BCEWithLogitsLoss()

            loss_cls = loss_fct(pooled_logits, classes)

            latent_feature_score = self.head_feat(hidden_states)  # [B, L, F]
            pooled_latent_feature_score = latent_feature_score[torch.arange(batch_size, device=latent_feature_score.device), sequence_lengths]  # [B, F]

            prompt_feature_score = features[:, self.config.feature_subset] if self.config.feature_subset else features
            if self.training:
                prompt_feature_score = torch.concat([prompt_feature_score, classes], dim=-1)  # [B, F]
            else:
                pooled_logits_ = torch.sigmoid(pooled_logits)
                prompt_feature_score = torch.concat([prompt_feature_score, pooled_logits_], dim=-1)  # [B, F]

            # align the prompt feature score with the latent feature score
            if self.config.alignment_type == "additive":
                prompt_feature = prompt_feature_score + pooled_latent_feature_score
            elif self.config.alignment_type == "multiplicative":
                prompt_feature = prompt_feature_score * pooled_latent_feature_score
            elif self.config.alignment_type == "attentive":
                score = prompt_feature_score * pooled_latent_feature_score
                score = F.softmax(score, dim=-1)
                prompt_feature = pooled_latent_feature_score * score + prompt_feature_score

            prompt_feature = self.head_align(prompt_feature.unsqueeze(-1))  # [B, F, X]
            graph_losses = self.graph_learner(prompt_feature)

            loss_align = F.mse_loss(prompt_feature_score, pooled_latent_feature_score)
            graph_losses["loss_align"] = loss_align.item()
            graph_losses["loss_cls"] = loss_cls.item()

            loss = graph_losses.pop("loss") + self.align_lambda * loss_align + self.cls_lambda * loss_cls
            losses = torch.tensor([list(graph_losses.values())], device=loss.device)

        return CausalAnalystOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            classes=classes,
            losses=losses,
        )
