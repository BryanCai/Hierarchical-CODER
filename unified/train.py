import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_util import UMLSDataset, TreeDataset, fixed_length_dataloader
from sampler_util import my_collate_fn
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
from tensorboardX import SummaryWriter


def train(args, model, umls_dataloader, tree_dataloader=None):
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

    if args.fine_tune:
        for name, param in model.bert.named_parameters():
            if name.startswith("encoder.layer"):
                if name.startswith("encoder.layer.11"):
                    pass
                else:
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

    model.zero_grad()
    global_step=0

    while True:
        model.train()

        if not args.fine_tune:
            batch_loss = 0.
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
                    loss =  model.get_umls_loss(input_ids_0, input_ids_1, input_ids_2,
                                                cui_label_0, cui_label_1, cui_label_2,
                                                sty_label_0, sty_label_1, sty_label_2)
                    batch_loss = float(loss.item())

                    writer.add_scalar('batch_loss', batch_loss, global_step=global_step)
                    batch_iterator.set_description("Loss: {:.4f}".format(batch_loss))

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()


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

        else:
            batch_loss = 0.

            batch_iterator = tqdm(zip(itertools.cycle(tree_dataloader), umls_dataloader), desc="Iteration", ascii=True)
            for tree_batch, umls_batch in batch_iterator:
                if tree_batch is not None:
                    anchor_ids        = tree_batch[0].to(args.device)
                    neg_samples_ids   = tree_batch[1].to(args.device)
                    neg_samples_dists = tree_batch[2].to(args.device)
                    loss = model.get_tree_loss(anchor_ids, neg_samples_ids, neg_samples_dists)
                    tree_loss = float(loss.item())

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

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

                    loss =  model.get_umls_loss(input_ids_0, input_ids_1, input_ids_2,
                                                cui_label_0, cui_label_1, cui_label_2,
                                                sty_label_0, sty_label_1, sty_label_2)
                    batch_loss = float(loss.item())

                    writer.add_scalar('batch_loss', batch_loss, global_step=global_step)
                    batch_iterator.set_description("Loss: {:.4f}".format(batch_loss))

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()



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


    if args.fine_tune:
        tree_dataset = TreeDataset(tree_dir=args.tree_dir,
            model_name_or_path=args.model_name_or_path,
            eval_data_path=args.eval_data_path)

        tree_dataloader = fixed_length_dataloader(
            tree_dataset, fixed_length=args.tree_batch_size, num_workers=args.num_workers)
    else:
        tree_dataloader=None


    print('-------')
    print(len(tree_dataset))
    print(len(umls_dataset))
    print('-------')



    os.makedirs(args.output_dir, exist_ok=True)
    model = UMLSPretrainedModel(base_model=args.model_name_or_path,
                                clogit_alpha=args.clogit_alpha,
                                sim_dim=args.sim_dim).to(args.device)


    if args.fine_tune:
        if args.coder_path[-4:] == ".pth":
            coder = torch.load(args.coder_path, map_location=args.device).to(args.device)
        else:
            coder = AutoModel.from_pretrained(args.coder_path).to(args.device)
        model.bert = coder


    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    train(args, model, umls_dataloader, tree_dataloader)
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
        "--tree_dir",
        type=str,
        help="Path to tree directory",
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
    parser.add_argument(
        "--umls_batch_size", default=256, type=int, help="Batch size per GPU/CPU for umls training.",
    )
    parser.add_argument(
        "--tree_batch_size", default=512, type=int, help="Batch size per GPU/CPU for tree training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-6,
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
    parser.add_argument("--num_workers", default=1, type=int,
                        help="Num workers for data loader, only 0 can be used for Windows")
    parser.add_argument("--lang", default='eng', type=str, choices=["eng", "all"],
                        help="language range, eng or all")


    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--coder_path", type=str, default=None)

    parser.add_argument("--clogit_alpha", type=float, default=2,
                        help="alpha for clogit loss.")

    parser.add_argument("--sim_dim", type=int, default=-1,
                        help="dimension to use for similarity")


    parser.add_argument("--fine_tune", action="store_true",
                        help="freeze all but last layer")


    args = parser.parse_args()


    print(args.fine_tune)
    run(args)


if __name__ == "__main__":
    main()
