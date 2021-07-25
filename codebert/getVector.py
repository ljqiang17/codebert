import argparse
import glob
import logging
import os
import random
from sklearn import preprocessing
import time


import numpy as np
from numpy import *
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          RobertaModel)

from utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a.cpu())
    vector_b = np.mat(vector_b.cpu())
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def load_and_cache_examples(task, tokenizer, dir):
    # data_dir = "../data/codesearch/test/java"
    ttype = 'test'
    data_dir = dir
    test_file = "batch_0.txt"
    # test_file = "batch_test.txt"
    model_type = 'roberta'
    # max_seq_length = 200
    max_seq_length = 500
    local_rank = -1
    model_name_or_path = 'microsoft/codebert-base'

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    '''
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    '''
    file_name = test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    '''
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, test_file)
        
    except:
    '''
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    '''
    if ttype == 'train':
        examples = processor.get_train_examples(args.data_dir, args.train_file)
    elif ttype == 'dev':
        examples = processor.get_dev_examples(args.data_dir, args.dev_file)
    elif ttype == 'test':
        examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    '''
    examples, instances = processor.get_test_examples(data_dir, test_file)

    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 1,
                                            pad_on_left=bool(model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0)
    if local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default="", type=str)
    parser.add_argument("--bug_id", default="", type=str)
    args = parser.parse_args()
    # basePath = "/home/tank01/Desktop/Simfix/result/" + args.project_name + "/" + args.bug_id
    basePath = "/data/Simfix/result/" + args.project_name + "/" + args.bug_id

    vecList = []

    local_rank = -1
    task_name = "codesearch"
    output_dir = basePath

    # load the model
    print("----------loading the model----------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    model_type = 'roberta'
    print("--------load model finished----------\n")

    # load the data
    print("----------loading the data----------")
    eval_task_names = (task_name,)
    eval_outputs_dirs = (output_dir,)
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, instances = load_and_cache_examples(eval_task, tokenizer,output_dir)

        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        per_gpu_eval_batch_size = 32
        n_gpu = torch.cuda.device_count()
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        print("----------load data finished--------\n")


        # use codeBERT
        print("----------running the model----------\n")
        for batch in tqdm(eval_dataloader, desc="Running"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,
                          }
                        # XLM don't use segment_ids
                         # 'labels': batch[3]}

                outputs = model(**inputs)
                # outputs2 = model(candidate_code)

                # get_similarity(outputs, outputs2)

            #print("----------showing the result---------")
            #print(outputs)
            # print(outputs[0][0].shape)
            # print(outputs[1].shape)
            vecOfCode = outputs[1]
            # buggy = vecOfCode[0]
            # vecList = []
            # cosList = []
            for vec in vecOfCode:
                vecList.append(vec)
            """
            for i in range(1, len(vecList)):
                cos = cos_sim(buggy, vecList[i])
                cosList.append(cos)
            """

    # Read time from file
    # total_time = read_from_file()

    cosList = []
    buggy = vecList[0]


    for i in range(1, len(vecList)):
        cos = cos_sim(buggy, vecList[i])
        cosList.append(cos)




    simPath = basePath + "/similarity.txt"
    print("-----------writing the sim----------")
    with open(simPath, 'w') as fw:
        for eachCos in cosList:
            fw.write(str(eachCos) + '\n')


            #print(buggy)
            # for vec in vecOfCode:
            # print(vec)






main()