from functools import partial
from pathlib import Path
import nltk
import pickle as pk
import torch
from context_enforcement.models.context_enforcer import compute_context_boundary
from context_enforcement.trainers.train_bart import model_init as model_init_form1
from context_enforcement.trainers.train_bart_crossed_form1 import model_init as model_init_form2
from context_enforcement.data.common import create_text_tokenizer, SmartCollator
from context_enforcement.trainers.common import get_dataset_specified_tasks,create_inference_args
from pytorch_lightning import seed_everything

import sys
import os
seed_everything(1376)

from torch.utils.data import DataLoader,SequentialSampler
import tqdm
from context_enforcement.data.common import write_to_file
import evaluate
metrics = evaluate.combine(['bleu','meteor',"rouge"])
def generate(tokenizer,generator,test_data_loader,beam_size=10):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_summaries =[]
    for batch in test_data_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        
        seq_len = b_input_ids.shape[1]

        context_boundary = compute_context_boundary(seq_len,
                                                    context_max_len=110)
        
        boundary_mask =  batch.get("boundary",)
        bb=generator.generate(input_ids=b_input_ids,
                attention_mask=b_input_mask,
                num_beams=beam_size,
                do_sample=False,
                num_return_sequences=1,
                max_length=140)
        sentences = tokenizer.batch_decode(bb,clean_up_tokenization_spaces=True,skip_special_tokens=True)
        output_summaries+=sentences
    return output_summaries

def run_inference(tokenizer,checkpoint_folder, model_gen_func,test_data_loader,inference_config,targets):
    generator = model_gen_func()
    checkpoint_path = os.path.join(checkpoint_folder,"pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        return 
    
    state_dict = torch.load(checkpoint_path)
    generator.load_state_dict(state_dict,strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_name = [s for s in checkpoint_folder.split('/') if len(s)>0 and 'checkpoint' in s][-1]
    
    inference_path = f"outputs/{inference_config.run_id}/{checkpoint_name}/"
    os.makedirs(inference_path,exist_ok=True)
    
    print(f"Performing inference using {checkpoint_name}")
    
    if inference_config.is_baseline:
        print("Using baseline model")
        rbase_output = output_summary = generate(tokenizer,generator,test_data_loader)
        write_to_file(rbase_output[:len(targets)], 
                f"{inference_path}/best_base_final")
        
        scores = metrics.compute(predictions=rbase_output,references=targets)
        print(scores)
        
        pk.dump(scores
                ,open(f"{inference_path}/best_results.pk",'wb'))
        return
        
    
    context_lens = [100, 200, 300, 400, 500, 650]
    outputs = {}
    results = {}
    for cl in context_lens:
        print(f'Generating for the context length: {cl}')
        generator.model.context_max_len = cl
        generator.model.encoder.context_max_len = cl
        rbase_output = output_summary = generate(tokenizer,generator,test_data_loader)
        outputs[cl] = rbase_output
        
        write_to_file(rbase_output[:len(targets)], 
                f"{inference_path}/best_base_final_{cl}")
        
        scores = metrics.compute(predictions=rbase_output,references=targets)
        print(scores)
        
        results[cl]= scores
    pk.dump(results,open(f"{inference_path}/best_results.pk",'wb'))
        


if __name__ == "__main__":
    parser = create_inference_args()

    arguments = parser.parse_args()
    params = vars(arguments)
    # trained-model-path
    trained_model_path = os.path.join(arguments.trained_model_path,arguments.run_id)
    train_args_path = os.path.join(trained_model_path, "train_args.ap")
    configs =  pk.load(open(train_args_path ,'rb'))
    tokenizer = create_text_tokenizer(configs.model_base)
    is_baseline = configs.is_baseline
    context_max_len = configs.context_max_len
    context_max_len_list = configs.context_max_len_list
    context_max_len_list = (
        ast.literal_eval(context_max_len_list[0])
        if context_max_len_list
        else [context_max_len_list]
    )

    if type(context_max_len_list) is not list:
        context_max_len_list = list(context_max_len_list)

    context_sampling_bounds = (0.15, 0.45)
    # configs.context_sampling_bounds
    task_dataset_gen = get_dataset_specified_tasks(configs.task_type)

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if task_dataset_gen is not None:
        raw_dataset = task_dataset_gen(
            tokenizer=tokenizer,
        )
        train_dataset = raw_dataset["train"]
        eval_dataset = raw_dataset["validation"]
        test_dataset = raw_dataset["test"]
    
    test_data_loader = DataLoader(test_dataset,batch_size=12,
                                sampler= SequentialSampler(test_dataset),
                                collate_fn= SmartCollator(
                pad_token_id=train_dataset.tokenizer.pad_token_id,
                max_len=configs.max_seq_len,
                context_max_len=context_max_len,
                context_sampling_bounds=context_sampling_bounds,
            
            ))
    
    is_form2 = arguments.is_form2
    model_init = model_init_form2 if is_form2 else model_init_form1
    
    model_builder = model_init(
        vocab_size=len(train_dataset.tokenizer),
        model_base=configs.model_base,
        is_baseline=is_baseline,
        context_max_len=context_max_len,
        context_sampling_bounds=context_sampling_bounds,
        context_max_len_list=context_max_len_list,
        
        
    )
    checkpoints=[s.as_posix() for s in Path(trained_model_path).rglob("*checkpoint-*")]
    print(checkpoints)
    
    targets = [tokenizer.decode(c.labels,clean_up_tokenization_spaces=True,skip_special_tokens=True) for c in test_dataset]
    
    for checkpoint in checkpoints:
        run_inference(tokenizer,checkpoint,model_builder,test_data_loader ,configs,targets)
