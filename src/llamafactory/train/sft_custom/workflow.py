import os
import pickle as pkl
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.convert_matrix import from_numpy_array

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ..callbacks import ModelPostProcessorCallback
from .loader import load_model, load_tokenizer
from .metric import ComputeAllMetrics, ComputeClassificationMetrics, ComputeGraphMetrics
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import CustomArguments, DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft_custom(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    custom_args: "CustomArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args, custom_args)
    tokenizer = tokenizer_module["tokenizer"]
    if model_args.flash_attn == 'fa2':
        tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    model = load_model(tokenizer, model_args, finetuning_args, custom_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if custom_args.training_stage == 1:
        metric_module["compute_metrics"] = ComputeClassificationMetrics()
    elif custom_args.training_stage == 2:
        metric_module["compute_metrics"] = ComputeGraphMetrics()
    else:
        metric_module["compute_metrics"] = ComputeAllMetrics()

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    if custom_args.training_stage in [2, 3]:
        callbacks.append(ModelPostProcessorCallback)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        try:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        except Exception as e:
            logger.warning_rank0_once("Training failed, trying again without resume_from_checkpoint.")
            train_result = trainer.train(resume_from_checkpoint=False)

        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if custom_args.training_stage == 1:
                keys += [
                    "eval_causal_test_loss", "eval_causal_test_accuracy", 
                    "eval_causal_test_precision", "eval_causal_test_recall", "eval_causal_test_f1"
                ]
            elif custom_args.training_stage == 2:
                keys += [
                    "eval_causal_test_loss_nll", "eval_causal_test_loss_kl", "eval_causal_test_loss_elbo",
                    "eval_causal_test_loss_sparse", "eval_causal_test_loss_lagr", "eval_causal_test_loss_mse",
                    # "eval_causal_test_loss_connect", "eval_causal_test_loss_positive", 
                    "eval_causal_test_loss_graph", "eval_causal_test_loss_align",
                ]
            elif custom_args.training_stage == 3:
                keys += [
                    "eval_causal_test_loss", "eval_causal_test_accuracy", 
                    "eval_causal_test_precision", "eval_causal_test_recall", "eval_causal_test_f1",
                    "eval_causal_test_loss_nll", "eval_causal_test_loss_kl", "eval_causal_test_loss_elbo",
                    "eval_causal_test_loss_sparse", "eval_causal_test_loss_lagr", "eval_causal_test_loss_mse",
                    # "eval_causal_test_loss_connect", "eval_causal_test_loss_positive", 
                    "eval_causal_test_loss_graph", "eval_causal_test_loss_align",
                ]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero() and custom_args.training_stage in [2, 3]:
            graphs = model.get_causal_graph()
            with open(os.path.join(training_args.output_dir, "graph.pkl"), "wb") as f:
                pkl.dump(graphs, f)

            # Create binary adjacency matrix using config threshold
            # matG1 = np.matrix(graph)
            # final_df = pd.DataFrame(matG1, index=custom_args.feature_names, columns=custom_args.feature_names)

            # Save final binary adjacency matrix
            # final_df.to_csv(os.path.join(training_args.output_dir, "final_adjacency_matrix.csv"), index=True)

            # # Draw the DAG
            # final_DAG = from_numpy_array(final_df.to_numpy(), create_using=nx.DiGraph)
            # final_DAG = nx.relabel_nodes(
            #     final_DAG, dict(zip(list(range(custom_args.num_feature_nodes)), custom_args.feature_names))
            # )
            # final_DAG.remove_nodes_from(list(nx.isolates(final_DAG)))

            # nx.draw(
            #     final_DAG,
            #     node_color="lightcoral",
            #     node_size=75,
            #     font_size=3,
            #     width=0.5,
            #     arrowsize=4,
            #     with_labels=True,
            #     pos=nx.spring_layout(final_DAG),
            # )
            # plt.draw()
            # plt.savefig(os.path.expanduser(os.path.join(training_args.output_dir, "DAG_plot.png")), format="PNG", dpi=500)
            # plt.close()

    # Predict
    if training_args.do_predict and custom_args.training_stage in [1, 3]:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        test_dataset = dataset_module["eval_dataset"]['causal_test']
        predict_results = trainer.predict(test_dataset, metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(test_dataset, predict_results)
