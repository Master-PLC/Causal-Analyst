import os
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoProcessor, \
    AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ...custom.qwen2causal import Qwen2Custom, Qwen2CustomConfig
from ...extras import logging
from ...extras.misc import count_parameters
from ...model.adapter import init_adapter
from ...model.loader import TokenizerModule, _get_init_kwargs
from ...model.model_utils.attention import print_attn_implementation
from ...model.model_utils.checkpointing import prepare_model_for_training
from ...model.model_utils.embedding import resize_embedding_layer
from ...model.model_utils.liger_kernel import apply_liger_kernel
from ...model.model_utils.misc import register_autoclass
from ...model.model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from ...model.model_utils.moe import add_z3_leaf_module
from ...model.model_utils.unsloth import load_unsloth_pretrained_model
from ...model.model_utils.valuehead import load_valuehead_params, prepare_valuehead_model
from ...model.model_utils.visual import autocast_projector_dtype
from ...model.patcher import patch_config, patch_processor, patch_tokenizer, patch_valuehead_model

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments, CustomArguments

from types import MethodType

logger = logging.get_logger(__name__)


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def load_tokenizer(model_args: "ModelArguments", custom_args: "CustomArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args, custom_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.debug(f"Processor was not found: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments", custom_args: "CustomArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = Qwen2CustomConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    return wrap_config(config, custom_args)


def wrap_config(config: "PretrainedConfig", custom_args: "CustomArguments") -> "PretrainedConfig":
    r"""
    Wraps config with custom arguments.
    """
    if custom_args is not None:
        for key, value in custom_args.to_dict().items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    custom_args: "CustomArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args, custom_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():
                load_class = AutoModelForSeq2SeqLM
            elif type(config) == Qwen2CustomConfig:
                load_class = Qwen2Custom
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.padding_side = tokenizer.padding_side

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # if custom_args.extra_modules_to_train is not None and is_trainable:
    #     for module_name in custom_args.extra_modules_to_train:
    #         for name, param in model.named_parameters():
    #             if module_name in name:
    #                 if 'adj_A' in name and custom_args.training_stage == 2 and custom_args.first_train:
    #                     param.data = torch.zeros_like(param.data)
    #                 param.requires_grad = True
    #                 param.data = param.data.to(torch.float32)

    if finetuning_args.additional_target is not None and is_trainable and custom_args.training_stage in [2, 3] and custom_args.first_train:
        for name, param in model.named_parameters():
            if 'adj_A' in name:
                param.data = torch.zeros_like(param.data, dtype=param.data.dtype)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
