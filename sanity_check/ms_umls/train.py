import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_util import UMLSDataset, fixed_length_dataloader
from model import UMLSPretrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import torch
from torch import nn
import time
import os
import numpy as np
import argparse
import time
import pathlib
import itertools
#import ipdb
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
from tensorboardX import SummaryWriter


def train(args, model, umls_dataloader, umls_dataset):
    writer = SummaryWriter(comment='umls')

    t_total = args.max_steps

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    for name, param in model.bert.named_parameters():
        if name.startswith("encoder.layer"):
            if name.startswith("encoder.layer.11"):
                pass
            else:
                print(name)
                param.requires_grad = False

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(args.warmup_steps)
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    if args.schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    if args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    print("***** Running training *****")
    print("  Total Steps =", t_total)
    print("  Steps needs to be trained=", t_total - args.shift)
    print("  Instantaneous batch size per GPU =", args.umls_batch_size)
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) =",
        args.umls_batch_size
        * args.gradient_accumulation_steps,
    )
    print("  Gradient Accumulation steps =", args.gradient_accumulation_steps)

    model.zero_grad()

    for i in range(args.shift):
        scheduler.step()
    global_step = args.shift

    while True:
        model.train()

        batch_loss = 0.
        # batch_iterator = tqdm(zip(itertools.cycle(tree_dataloader), umls_dataloader), desc="Iteration", ascii=True)
        batch_iterator = tqdm(umls_dataloader, desc="Iteration", ascii=True)
        for umls_batch in batch_iterator:
            if umls_batch is not None:
                input_ids_0 = umls_batch[0].to(args.device)
                input_ids_1 = umls_batch[1].to(args.device)
                input_ids_2 = umls_batch[2].to(args.device)
                cui_label_0 = umls_batch[3].to(args.device)
                cui_label_1 = umls_batch[4].to(args.device)
                cui_label_2 = umls_batch[5].to(args.device)
                sty_label_0 = umls_batch[6].to(args.device)
                sty_label_1 = umls_batch[7].to(args.device)
                sty_label_2 = umls_batch[8].to(args.device)
                # use umls_batch[9] for re, use umls_batch[10] for rel
                if args.use_re:
                    re_label = umls_batch[9].to(args.device)
                else:
                    re_label = umls_batch[10].to(args.device)
                # for item in umls_batch:
                #     print(item.shape)

                loss = \
                    model(input_ids_0, input_ids_1, input_ids_2,
                                       cui_label_0, cui_label_1, cui_label_2,
                                       sty_label_0, sty_label_1, sty_label_2,
                                       re_label)
                batch_loss = float(loss.item())

                # tensorboardX
                writer.add_scalar(
                    'rel_count', umls_dataloader.batch_sampler.rel_sampler_count, global_step=global_step)
                writer.add_scalar('batch_loss', batch_loss,
                                  global_step=global_step)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                batch_iterator.set_description("Rel_count: %s, Loss: %0.4f" %
                                               (umls_dataloader.batch_sampler.rel_sampler_count, batch_loss))

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            global_step += 1
            if global_step % args.save_step == 0 and global_step > 0:
                save_path = os.path.join(
                    args.output_dir, f'model_{global_step}.pth')
                torch.save(model, save_path)

            if args.max_steps > 0 and global_step > args.max_steps:
                return None


    return None


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    #args.output_dir = args.output_dir + "_" + str(int(time.time()))

    # dataloader
    if args.lang == "eng":
        lang = ["ENG"]
    if args.lang == "all":
        lang = None
        assert args.model_name_or_path.find("bio") == -1, "Should use multi-language model"
    umls_dataset = UMLSDataset(
        umls_folder=args.umls_dir, model_name_or_path=args.model_name_or_path, lang=lang, json_save_path=args.output_dir, eval_data_path=args.eval_data_path)
    umls_dataloader = fixed_length_dataloader(
        umls_dataset, fixed_length=args.umls_batch_size, num_workers=args.num_workers)

    print('-------')
    print(len(umls_dataset))
    print('-------')

    if args.use_re:
        rel_label_count = len(umls_dataset.re2id)
    else:
        rel_label_count = len(umls_dataset.rel2id)


    model_load = False
    # if os.path.exists(args.output_dir):
    #     save_list = []
    #     for f in os.listdir(args.output_dir):
    #         if f[0:5] == "model" and f[-4:] == ".pth":
    #             save_list.append(int(f[6:-4]))
    #     if len(save_list) > 0:
    #         args.shift = max(save_list)
    #         if os.path.exists(os.path.join(args.output_dir, 'last_model.pth')):
    #             model = torch.load(os.path.join(
    #                 args.output_dir, 'last_model.pth')).to(args.device)
    #             model_load = True
    #         else:
    #             model = torch.load(os.path.join(
    #                 args.output_dir, f'model_{max(save_list)}.pth')).to(args.device)
    #             model_load = True
    if not model_load:
        os.makedirs(args.output_dir, exist_ok=True)
        model = UMLSPretrainedModel(device=args.device, model_name_or_path=args.model_name_or_path).to(args.device)
        if args.coder_path[-4:] == ".pth":
            coder = torch.load(args.coder_path, map_location=args.device).to(args.device)
        else:
            coder = AutoModel.from_pretrained(args.coder_path).to(args.device)
        model.bert = coder
        args.shift = 0
        model_load = True


    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, umls_dataloader, umls_dataset)
        torch.save(model, os.path.join(args.output_dir, 'last_model.pth'))

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umls_dir",
        default="../umls",
        type=str,
        help="UMLS dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="../biobert_v1.1",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_step",
        default=25000,
        type=int,
        help="Save step",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to run training.")
    parser.add_argument(
        "--umls_batch_size", default=256, type=int, help="Batch size per GPU/CPU for umls training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=1000000,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=10000,
                        help="Linear warmup over warmup_steps or a float.")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    parser.add_argument("--seed", type=int, default=72,
                        help="random seed for initialization")
    parser.add_argument("--schedule", type=str, default="linear",
                        choices=["linear", "cosine", "constant"], help="Schedule.")
    parser.add_argument("--trans_margin", type=float, default=1.0,
                        help="Margin of TransE.")
    parser.add_argument("--use_re", default=False, type=bool,
                        help="Whether to use re or rel.")
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Num workers for data loader, only 0 can be used for Windows")
    parser.add_argument("--lang", default='eng', type=str, choices=["eng", "all"],
                        help="language range, eng or all")
    parser.add_argument("--sty_weight", type=float, default=0.0,
                        help="Weight of sty.")
    parser.add_argument("--re_weight", type=float, default=1.0,
                        help="Weight of re.")

    parser.add_argument("--umls_neg_loss", action="store_true",
                        help="include negative loss for umls")
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--coder_path", type=str, default=None)

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()