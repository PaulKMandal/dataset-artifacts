import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_dataset_nli_hypothesis_only, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, CustomTrainer, CustomQuestionAnsweringTrainer, ResidualTrainer, compute_accuracy
import os
import json
from functools import partial

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # Existing arguments
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    # Existing flags
    argp.add_argument('--save_only_final_model', action='store_true',
                      help='When set, only the final model will be saved, not intermediate checkpoints.')
    argp.add_argument('--save_dynamics', action='store_true',
                      help='When set, the training dynamics will be saved to training_dynamics.jsonl in the output directory.')

    # New flags for residual fitting
    argp.add_argument('--unlearn_residual', action='store_true',
                      help='When set, trains the main model with residual fitting to unlearn dataset bias.')
    argp.add_argument('--train_biased_model', action='store_true',
                      help='When set, trains a biased hypothesis-only model before training the main model.')
    argp.add_argument('--biased_model', type=str, default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune as the biased model.
        This should either be a HuggingFace model ID or a path to a saved model checkpoint.""")
    argp.add_argument('--biased_model_task', type=str, choices=['nli'], default='nli',
                      help='Task for the biased model. Currently only "nli" is supported.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Adjust TrainingArguments if save_only_final_model is set
    if args.save_only_final_model:
        training_args.save_strategy = 'no'  # Do not save checkpoints during training
    else:
        # Default behavior: save checkpoints at each epoch
        training_args.save_strategy = 'epoch'

    # Dataset selection
    if args.dataset and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)

    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        # Use functools.partial to fix tokenizer and max_length
        prepare_train_dataset = partial(prepare_dataset_nli, tokenizer=tokenizer, max_seq_length=args.max_length)
        prepare_eval_dataset = partial(prepare_dataset_nli, tokenizer=tokenizer, max_seq_length=args.max_length)
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    # Initialize variables for main model
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None

    # Initialize variables for biased model
    biased_model = None
    biased_tokenizer = None
    biased_train_dataset_featurized = None

    if args.train_biased_model:
        # Load biased model and tokenizer
        biased_model = AutoModelForSequenceClassification.from_pretrained(args.biased_model, **task_kwargs)
        biased_tokenizer = AutoTokenizer.from_pretrained(args.biased_model, use_fast=True)

        # Prepare biased training dataset (hypothesis only)
        if args.task == 'nli':
            prepare_biased_dataset_nli = partial(prepare_dataset_nli_hypothesis_only, tokenizer=biased_tokenizer, max_seq_length=args.max_length)
            biased_train_dataset = dataset['train']
            if args.max_train_samples:
                biased_train_dataset = biased_train_dataset.select(range(args.max_train_samples))
            biased_train_dataset = biased_train_dataset.map(lambda ex, idx: {'idx': idx}, with_indices=True)
            biased_train_dataset_featurized = biased_train_dataset.map(
                prepare_biased_dataset_nli,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=biased_train_dataset.column_names
            )
        else:
            raise ValueError('Biased model training is only supported for NLI tasks.')

    if not args.train_biased_model:
        # Prepare main training dataset
        if training_args.do_train:
            train_dataset = dataset['train']
            if args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
            # Add index to each example
            train_dataset = train_dataset.map(lambda ex, idx: {'idx': idx}, with_indices=True)
            train_dataset_featurized = train_dataset.map(
                prepare_train_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=train_dataset.column_names
            )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    compute_metrics = None
    if args.task == 'qa':
        trainer_class = CustomQuestionAnsweringTrainer
        metric = evaluate.load('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        if args.unlearn_residual:
            trainer_class = ResidualTrainer  # Custom trainer for residual fitting
            compute_metrics = compute_accuracy
        else:
            trainer_class = CustomTrainer
            compute_metrics = compute_accuracy
    else:
        raise ValueError('Unsupported task.')

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Determine label names based on task
    if args.task == 'qa':
        label_names = ['start_positions', 'end_positions', 'idx']
    else:
        label_names = ['labels', 'idx']

    # Initialize the main model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model, **task_kwargs)

    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # If training a biased model
    if args.train_biased_model:
        trainer_kwargs = dict(
            model=biased_model,
            args=training_args,
            train_dataset=biased_train_dataset_featurized if training_args.do_train else None,
            eval_dataset=eval_dataset_featurized if training_args.do_eval else None,
            tokenizer=biased_tokenizer,
            compute_metrics=compute_metrics_and_store_predictions if training_args.do_eval else None,
            output_dir=os.path.join(training_args.output_dir, 'biased_model')
        )

        trainer = trainer_class(**trainer_kwargs)
        trainer.label_names = label_names

        if args.task == 'qa':
            trainer.eval_examples = eval_dataset

        # Train and/or evaluate the biased model
        if training_args.do_train:
            trainer.train()
            trainer.save_model()

        if training_args.do_eval:
            results = trainer.evaluate()

            print('Biased Model Evaluation results:')
            print(results)

            os.makedirs(trainer.args.output_dir, exist_ok=True)

            with open(os.path.join(trainer.args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
                json.dump(results, f)

            with open(os.path.join(trainer.args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                if args.task == 'qa':
                    predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                    for example in eval_dataset:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_answer'] = predictions_by_id.get(example['id'], '')
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
                else:
                    for i, example in enumerate(eval_dataset):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')

    # Prepare for residual fitting training or normal main model training
    if not args.train_biased_model:
        # Determine if residual fitting is enabled
        if args.unlearn_residual:
            # Load the biased model
            if not args.biased_model:
                raise ValueError("Biased model path must be provided when using residual unlearning.")
            biased_model = AutoModelForSequenceClassification.from_pretrained(args.biased_model, **task_kwargs)
            biased_model.eval()
            for param in biased_model.parameters():
                param.requires_grad = False

        # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized if training_args.do_train else None,
            eval_dataset=eval_dataset_featurized if training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions if training_args.do_eval else None,
            output_dir=training_args.output_dir
        )

        # Pass additional arguments based on whether residual fitting is enabled
        if args.unlearn_residual:
            trainer_kwargs['biased_model'] = biased_model  # Pass the biased model to the ResidualTrainer

        trainer = trainer_class(**trainer_kwargs)
        trainer.label_names = label_names

        # For QA, pass eval_examples to the trainer
        if args.task == 'qa':
            trainer.eval_examples = eval_dataset

        # Train and/or evaluate the main model
        if training_args.do_train:
            trainer.train()
            trainer.save_model()

        if training_args.do_eval:
            results = trainer.evaluate()

            print('Main Model Evaluation results:')
            print(results)

            os.makedirs(training_args.output_dir, exist_ok=True)

            with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
                json.dump(results, f)

            with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                if args.task == 'qa':
                    predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                    for example in eval_dataset:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_answer'] = predictions_by_id.get(example['id'], '')
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
                else:
                    for i, example in enumerate(eval_dataset):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')


if __name__ == "__main__":
    main()
