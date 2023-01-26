import traceback
import signal
import lap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import baselines
import bipartite_matching
from globals import *
import globals
from corpus_utils import *
from transformer import Encoder as SelfAttentiveEncoder

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
world_size = 1
if "WORLD_SIZE" in os.environ:
    world_size = int(os.environ["WORLD_SIZE"])
    distributed = world_size > 1
    ngpus_per_node = torch.cuda.device_count()
rank = -1
if 'SLURM_PROCID' in os.environ: # for slurm scheduler
    rank = int(os.environ['SLURM_PROCID'])
    gpu = rank % torch.cuda.device_count()
dist.init_process_group("nccl", init_method='env://',
                            world_size=world_size, rank=rank)


script_path = os.path.dirname(os.path.realpath(__file__))+'/'
np.random.seed(globals.RANDOM_SEED)
torch.manual_seed(globals.RANDOM_SEED)

def handler(signum, frame):
    raise Exception("Timeout:Could not solve LAP in time")

signal.signal(signal.SIGALRM, handler)


class Similarity(nn.Module):
    def __init__(self,
                 d_model, deep):
        super(Similarity, self).__init__()

        out_bi = d_model if deep else 1
        self.bi = nn.Bilinear(d_model, d_model, out_bi)
        self.proj = None
        if deep:
            self.proj = nn.Sequential(nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, svecs, pvecs):
        if self.proj is None:
            return self.bi(svecs, pvecs)
        v = self.bi(svecs, pvecs)
        return self.proj(v)

class TransfoEnc(nn.Module):
    def __init__(self, 
                n_src_vocab, len_max_seq, d_word_vec,
                n_layers, n_head, d_k, d_v,
                d_model, d_inner, dropout=0.1, scratchbert=False, mathbert=False):

        super(TransfoEnc, self).__init__()

        # self.encoder = SelfAttentiveEncoder(n_src_vocab, len_max_seq, d_word_vec,
        #                                     n_layers, n_head, d_k, d_v,
        #                                     d_model, d_inner, dropout=dropout)
        assert not (scratchbert and mathbert)
        if scratchbert:
            self.encoder = torch.nn.parallel.DataParallel(BertModel.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/train_from_scratch/model_files"))
        elif mathbert:
            self.encoder = torch.nn.parallel.DataParallel(BertModel.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/baseline/MathBERT-custom"))
        else:
            raise Exception("No encoder specified")
        self.sim = Similarity(d_model, deep=False)

    def forward(self, input):
        """
        Args:
            input: list of torch.long tensors
        Returns:
            res: tensor of size (batch, out_dim)
        """
        input = [i[:self.encoder.max_length] for i in input]

        padded_input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
        pos = torch.arange(1, len(padded_input[0]) + 1).repeat(len(padded_input), 1).to(padded_input.device)

        output = self.encoder(padded_input, pos)

        # output is a tuple of length 1 or 2 (2 if return attention vectors)
        res = output[0]

        #res = res[:,0,:]
        return torch.max(res, dim=1)[0]




def get_voc(corpus):
    # TODO: fix for splitting math / font
    voc = defaultdict(int)
    voc["<UNK>"] += 2
    voc["<START>"] += 2
    voc["<END>"] += 2

    for pair in corpus:
        statement = pair.statement.get_statement_tokenized(BOTH)
        for token in statement:
            voc[token] += 1
        proof = pair.proof.get_statement_tokenized(BOTH)
        for token in proof:
            voc[token] += 1

    voc = {k: v for k,v in voc.items() if v > 1}

    """
    for pair in corpus:
        statement = pair.statement.get_statement(BOTH)
        # todo: use sklearn tokenizer
        for token in statement.split():
            for char in token:
                voc[char] += 1
        proof = pair.proof.get_statement(BOTH)
        for token in proof.split():
            for char in token:
                voc[char] += 1
    """
    return voc


def prepare_toks(statement, device, input_type, char2i):
    
    text = statement.get_statement_tokenized(input_type)
    
    idxes = [char2i["<START>"]]
    for token in text:
        if token in char2i:
            idxes.append(char2i[token])
        else:
            idxes.append(char2i["<UNK>"])
    idxes.append(char2i["<END>"])
    return torch.tensor(idxes, dtype=torch.long).to(device)

def prepare_chars(statement, device, input_type, char2i):
    ## TODO: broken (ignore spaces)
    text = statement.get_statement(input_type)
    
    idxes = [char2i["<START>"]]
    for char in text:
        if char in char2i:
            idxes.append(char2i[char])
        else:
            idxes.append(char2i["<UNK>"])
    idxes.append(char2i["<END>"])

    return torch.tensor(idxes, dtype=torch.long).to(device)


def predict_eval(device, encoder, statements, proofs, global_decoding=False):
    # statements, proofs = corpus
    statement_vecs, proof_vecs = encode_documents(device, encoder, statements, proofs)

    # check whether same behaviour at train in compute_and print ranks and in evaluate_dense, turns out no :(
    # bc forgot customized sim_fun (bilinear form)
    #compute_and_print_ranks(encoder.sim, statement_vecs, proof_vecs, "_________________ranks")
    #  (Now fixed, leaving comments just in case)
    if global_decoding:
        return evaluate_global(statement_vecs, proof_vecs, encoder.sim)
    else:
        return evaluate_dense(statement_vecs, proof_vecs, encoder.sim, verbose=True)

def evaluate_global(statement_vecs, proof_vecs, sim_fun):
    # return same as evaluate_dense: mrr, acc, ranks
    #n_examples = len(statement_vecs)
    n_examples = statement_vecs.size(0)

    pruned_bipartite_graph = []
    logger = logging.getLogger()
    for i in range(n_examples):
        if i % 1000 == 0:
            logger.info("Global evaluation, computing similarity matrix: {} / {}".format(i, n_examples))
        # torch version
        row = statement_vecs[i]

        similarities = sim_fun(row.repeat(n_examples, 1), proof_vecs)
        similarities = similarities.cpu().numpy().flatten()

        ranks, sims = baselines.extract_best_ranking(similarities)

        pruned_bipartite_graph.append((i, ranks, sims))

    logger.info("Global evaluation: constructing data ")
    n, cc, ii, kk = bipartite_matching.construct_input_for_lapmod(pruned_bipartite_graph)

    logger.info("Global evaluation: bipartite matching n={} cc={} ii={} kk={}".format(n, cc.shape, ii.shape, kk.shape))
    cost, x, y = lap.lapmod(n, cc, ii, kk)

    logger.info("Global evaluation: bipartite matching:done")

    accuracy = np.sum(x == np.arange(n_examples))

    acc_check = 0
    for i, j in enumerate(x):
        if j == i:
            acc_check += 1
    assert(acc_check == accuracy)

    return accuracy / n_examples, accuracy / n_examples, None



def do_one_batch_step_local(similarities, targets_binary, targets_softmax, method, N, k):

    #probs = torch.sigmoid(similarities)
    #print(probs)
    if method == "sigmoid":
        loss = F.binary_cross_entropy(torch.sigmoid(similarities + np.log(N/k)), targets_binary, reduction="mean")
    elif method == "softmax":
        targets_softmax = torch.arange(similarities.shape[0]).to(device)
        loss = F.cross_entropy(similarities, targets_softmax, reduction='mean')
    else:
        assert(False)

    loss.backward()
    return loss.float()

def do_one_batch_step_global(similarities, cost_augmentation, device, batch_size, batch_accuracies):

    similarities = similarities + cost_augmentation
    numpy_sims = similarities.detach().cpu().numpy()

    #numpy_sims[:] = numpy_sims + (np.ones(numpy_sims.shape) - np.eye(numpy_sims.shape[0]))

    # minus numpy_sims because the algo is min cost (and we need max profit))
    score, assignmentx, assignmenty = lap.lapjv(- numpy_sims)

    accu = (assignmentx == np.arange(batch_size))
    batch_accuracies.append(np.sum(accu) / batch_size)

    if np.alltrue(accu):
        return 0

    gold_score = similarities.diag()
    pred_score = torch.gather(similarities, 1, torch.LongTensor(assignmentx).to(device).view(-1, 1)).view(-1)

    # add hamming cost between pred and gold assignments
    #cost = batch_size - np.sum(hamming)
    loss = torch.sum(pred_score - gold_score) / batch_size

    loss.backward()
    return loss.float()

def compute_similarities_batch(batch_statement, batch_proof, encoder, optimizer, batch_size, MAX_SIZE=8000):

    batch_statement = {k: torch.stack(v, dim=1).to(device) for k, v in batch_statement.items()}
    batch_proof = {k: torch.stack(v, dim=1).to(device) for k, v in batch_proof.items()}

    encoded_statements = encoder.encoder(**batch_statement).pooler_output
    encoded_proofs = encoder.encoder(**batch_proof).pooler_output

    batch_size = min(batch_size, encoded_statements.shape[0])
    repeated_statements = encoded_statements.repeat(1, batch_size).view(batch_size ** 2, -1)
    repeated_proofs = encoded_proofs.repeat(batch_size, 1)

    similarities = encoder.sim.forward(repeated_statements, repeated_proofs).view(batch_size, batch_size)

    return similarities





def export_voc(model, i2char):
    with open("{}/vocabulary".format(model), "w") as f:
        for c in i2char:
            if c == " ":
                c = "<SPACE>"
            if type(c) == tuple:
                c = "\t".join(c)
            f.write("{}\n".format(c))

def load_voc(model):
    i2char = []
    with open("{}/vocabulary".format(model)) as f:
        for line in f:
            c = line.strip("\n")
            if c == "<SPACE>":
                c = " "
            sc = c.split("\t")
            if len(sc) > 1:
                assert(len(sc) in{2, 3})
                c = tuple(c.split("\t"))
            i2char.append(c)
    return i2char, {k:i for i, k in enumerate(i2char)}





#def adjust_lr(optimizer, epoch):
#    lr = init_lr * (0.1 ** (epoch // 20))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr

def train_one_epoch(device, encoder, optimizer, train_statements, train_proofs, dev_statements, dev_proofs, dev_anno_statements, dev_anno_proofs, args, epoch_num):
    #(device, encoder, optimizer, train, batch_size, margin=1.0):
    """
        Generic function to train one epoch, global or local loss.
    """
    logger = logging.getLogger()

    statements = train_statements
    proofs = train_proofs
    N_examples = len(statements)

    local_epoch_loss = 0
    global_epoch_loss = 0
    batch_accuracies = []

    only_loc = args.G is None or (epoch_num < 40 and args.input == "text")
    only_glob = args.L is None
    both = (not only_loc) and (not only_glob)

    loc_obj = only_loc or both
    loc_blob = only_glob or both

    # benedict number_batch
    if args.L is not None:
        #args.negative_examples = args.G
        batch_size = args.N
    else:
        batch_size = args.G

    number_batches = (N_examples // batch_size)

    if args.G is not None:
        cost_augmented = (torch.ones(args.G, args.G) - torch.eye(args.G)).to(device)

    if args.N is not None:
        targets_local = torch.eye(args.N).to(device)
        targets_softmax = torch.arange(args.N).to(device)


    n_iterations = number_batches
    if both:
        n_iterations = 2 * number_batches
    batch = 0

    for batch_statement, batch_proof in tqdm(zip(statements, proofs), total=len(statements), leave=False, dynamic_ncols=True):
        batch += 1
        optimizer.zero_grad()

        next_is_local_step = only_loc or (both and batch % 2 == 1)

        if next_is_local_step:
            next_batch_size = args.N
        else:
            next_batch_size = args.G
        
        # 1. sample batch_size pairs
        sample_ids = [j for j in np.random.choice(N_examples, next_batch_size, replace=False)]

        try:
            similarities = compute_similarities_batch(batch_statement, batch_proof, encoder, optimizer, next_batch_size)

            if next_is_local_step:
                local_epoch_loss += do_one_batch_step_local(similarities,
                                                            targets_local,
                                                            targets_softmax,
                                                            args.L,
                                                            N_examples,
                                                            args.N)
            else:
                assert(only_glob or (both and batch % 2 == 0))
                global_epoch_loss += do_one_batch_step_global(similarities,
                                                              cost_augmented,
                                                              device,
                                                              args.G,
                                                              batch_accuracies)

            optimizer.step()

        except RuntimeError as e:
            logger.critical(str(e))
            logger.warning(traceback.format_exc())

            encoder.cpu()

            torch.save(encoder, "__crashed_model")
            logger.critical("Dumped model in __crashed_model  aborting")
            exit(1)

        if (not only_loc) and batch%2 == 0 and len(batch_accuracies) % 10 == 0:
            bat_ac = sum(batch_accuracies) / len(batch_accuracies)
            bat_ac = round(bat_ac * 100, 1)
            last_10 = batch_accuracies[-10:]
            last_10 = sum(last_10) / 10
            last_10 = round(last_10 * 100, 1)

            logger.info("train batch {}/{} accuracy={} last 10={} ".format(
                        len(batch_accuracies),
                        number_batches,
                        bat_ac,
                        last_10))

    normalized_global_loss = global_epoch_loss / number_batches
    normalized_local_loss = local_epoch_loss / number_batches
    batch_acc = 0
    if len(batch_accuracies) > 0:
        batch_acc = sum(batch_accuracies) / len(batch_accuracies)
    
    if only_loc:
        normalized_global_loss = normalized_local_loss

    # implement dev_loss
    dev_loss = 0
    dev_statements = dev_statements
    dev_proofs = dev_proofs
    dev_examples = len(dev_statements)
    dev_batches = (dev_examples // batch_size)
    with torch.no_grad():
        for batch_statement, batch_proof in zip(dev_statements, dev_proofs):
            # 1. sample batch_size pairs
            sample_ids = [j for j in np.random.choice(dev_examples, next_batch_size, replace=False)]

            # try:
            similarities = compute_similarities_batch(batch_statement, batch_proof, encoder, optimizer, next_batch_size)
            dev_loss += F.cross_entropy(similarities, torch.arange(similarities.shape[0]).to(device), reduction='mean')

    normalized_dev_loss = dev_loss / dev_batches

    return normalized_global_loss, normalized_local_loss, batch_acc, normalized_dev_loss


def do_training(args, device):

    logger = logging.getLogger()

    # train, metatrain, train_cats, train_primary_cats = baselines.initialize_corpus(args.train, "train", args.subset)
    # dev, metadev, dev_cats, dev_primary_cats = baselines.initialize_corpus(args.dev, "dev", args.subset)
    #
    # # vocabulary map
    # vocabulary = get_voc(train)
    # i2char = ["<PAD>"] + sorted(vocabulary, key = lambda x : vocabulary[x], reverse=True)
    # char2i = {k:i for i, k in enumerate(i2char)}
    #
    # export_voc(args.model, i2char)
    #
    # i2c, c2i = load_voc(args.model)
    # assert(c2i == char2i)
    # assert(i2c == i2char)
    #
    # # Encoder: Batch of int tensors -> Batch of fixed size representations
    #
    scratchbert = args.model.lower().startswith("scratchbert")
    mathbert = args.model.lower().startswith("mathbert")
    if scratchbert:
        tokenizer = AutoTokenizer.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/train_from_scratch/model_files", do_lower_case=True)
    elif mathbert:
        tokenizer = AutoTokenizer.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/baseline/MathBERT-custom", do_lower_case=True)
    def preprocess_theory(examples):
        return tokenizer(examples["theory"], padding='max_length', truncation=True, max_length=args.max_length)

    def preprocess_proof(examples):
        return tokenizer(examples["proof"], padding='max_length', truncation=True, max_length=args.max_length)

    train_dataset = pd.read_csv(args.train, sep=',')
    dev_dataset = pd.read_csv(args.dev, sep=',')
    dev_anno_dataset = pd.read_csv(args.anno_dev, sep=',')
    batch_size = args.N
    train_set = Dataset.from_pandas(train_dataset)
    dev_set = Dataset.from_pandas(dev_dataset)
    dev_anno_set = Dataset.from_pandas(dev_anno_dataset)
    train_statements = train_set.map(preprocess_theory, batched=True).remove_columns(['Unnamed: 0', 'theory', 'proof', 'meta'])
    train_proofs = train_set.map(preprocess_proof, batched=True).remove_columns(['Unnamed: 0', 'proof', 'theory', 'meta'])
    dev_statements = dev_set.map(preprocess_theory, batched=True).remove_columns(['Unnamed: 0', 'theory', 'proof', 'meta'])
    dev_proofs = dev_set.map(preprocess_proof, batched=True).remove_columns(['Unnamed: 0', 'proof', 'theory', 'meta'])
    dev_anno_statements = dev_anno_set.map(preprocess_theory, batched=True).remove_columns(['Unnamed: 0', 'theory', 'proof', 'meta'])
    dev_anno_proofs = dev_anno_set.map(preprocess_proof, batched=True).remove_columns(['Unnamed: 0', 'proof', 'theory', 'meta'])

    train_statements_loader = DataLoader(train_statements, batch_size=args.N)
    train_proofs_loader = DataLoader(train_proofs, batch_size=args.N)
    dev_statements_loader = DataLoader(dev_statements, batch_size=args.N)
    dev_proofs_loader = DataLoader(dev_proofs, batch_size=args.N)
    dev_anno_statements_loader = DataLoader(dev_anno_statements, batch_size=args.N)
    dev_anno_proofs_loader = DataLoader(dev_anno_proofs, batch_size=args.N)
    small_train = train_set.shuffle(seed=42).select(range(10000))
    small_train_statements = small_train.map(preprocess_theory, batched=True).remove_columns(['Unnamed: 0', 'theory', 'proof', 'meta'])
    small_train_statements_loader = DataLoader(small_train_statements, batch_size=args.N)
    small_train_proofs = small_train.map(preprocess_proof, batched=True).remove_columns(
        ['Unnamed: 0', 'theory', 'proof', 'meta'])
    small_train_proofs_loader = DataLoader(small_train_proofs, batch_size=args.N)
    encoder = TransfoEnc(n_src_vocab=len(tokenizer.get_vocab()), len_max_seq=args.max_length, d_word_vec=args.W,
                         n_layers=args.d, n_head=args.n_heads, d_k=args.dk, d_v=args.dk,
                         d_model=768, d_inner=args.W, dropout=0.1, scratchbert=scratchbert, mathbert=mathbert)
    # encoder = torch.load('/work/sc066/sc066/waylon/mathbert/tasks/theory-proof-matching/maximin_ori/bert_both/maximin_both')
    encoder.to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(encoder.parameters(), lr=args.l, momentum=0.1, weight_decay=0)
    elif args.optimizer == "asgd":
        optimizer = optim.ASGD(encoder.parameters(), lr=args.l, t0=10000, weight_decay=0)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(encoder.parameters(), lr=args.l)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(encoder.parameters(), lr=args.l)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adadelta(encoder.parameters(), lr=args.l)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    # Turn texts to tensors and transfer them to GPU
    # train_tensors = ([prepare_toks(s.statement, device, args.input, char2i) for s in train],
    #                  [prepare_toks(s.proof, device, args.input, char2i) for s in train])
    #
    # dev_tensors = ([prepare_toks(s.statement, device, args.input, char2i) for s in dev],
    #                [prepare_toks(s.proof, device, args.input, char2i) for s in dev])

    # n_dev = len(dev_tensors[0])
    # sample_train_tensors = (train_tensors[0][:n_dev], train_tensors[1][:n_dev])


    stat_file = open("{}/log_learning".format(args.model), "w", buffering=1)
    stat_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    stat_file.write(stat_line.format("Epoch", "tloss", "tmrr", "tacc", "dmrr", "dacc", "annodmrr", "annodacc", "batch_acc", "loc_loss", "lr"))

    global_model = args.G is not None

    best_acc = 0
    for e in tqdm(range(1, args.i + 1), leave=True, desc='train', dynamic_ncols=True):
        encoder.train()

        for param in encoder.parameters():
            param.requires_grad = True
        tloss, loc_loss, batch_acc, devloss = train_one_epoch(device, encoder, optimizer, train_statements_loader, train_proofs_loader, dev_statements_loader, dev_proofs_loader, dev_anno_statements_loader, dev_anno_proofs_loader, args, e)

        # if args.input == "both":
        #     if not 'anno' in args.train:
        #         with SummaryWriter('./runs_maximin_both/lines/train') as writer:
        #             writer.add_scalar('train_loss', tloss, e)
        #         with SummaryWriter('./runs_maximin_both/lines/val') as writer:
        #             writer.add_scalar('val_loss', devloss, e)
        #         with SummaryWriter('./runs_maximin_both/histo/') as writer:
        #             for name, layer in encoder.named_parameters():
        #                 writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), e)
        #     else:
        #         with SummaryWriter('./runs_maximin_anno_both/lines/train') as writer:
        #             writer.add_scalar('train_loss', tloss, e)
        #         with SummaryWriter('./runs_maximin_anno_both/lines/val') as writer:
        #             writer.add_scalar('val_loss', devloss, e)
        #         with SummaryWriter('./runs_maximin_anno_both/histo/') as writer:
        #             for name, layer in encoder.named_parameters():
        #                 writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), e)
        # else:
        #     if not 'anno' in args.train:
        #         with SummaryWriter('./runs_maximin_'+args.input+'_only/lines/train') as writer:
        #             writer.add_scalar('train_loss', tloss, e)
        #         with SummaryWriter('./runs_maximin_'+args.input+'_only/lines/val') as writer:
        #             writer.add_scalar('val_loss', devloss, e)
        #         with SummaryWriter('./runs_maximin_'+args.input+'_only/histo/') as writer:
        #             for name, layer in encoder.named_parameters():
        #                 writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), e)
        #     else:
        #         with SummaryWriter('./runs_maximin_anno_'+args.input+'_only/lines/train') as writer:
        #             writer.add_scalar('train_loss', tloss, e)
        #         with SummaryWriter('./runs_maximin_anno_'+args.input+'_only/lines/val') as writer:
        #             writer.add_scalar('val_loss', devloss, e)
        #         with SummaryWriter('./runs_maximin_anno_'+args.input+'_only/histo/') as writer:
        #             for name, layer in encoder.named_parameters():
        #                 writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), e)

        for param in encoder.parameters():
            param.requires_grad = False

        encoder.eval()
        scheduler.step()
    
        if e % 5 != 0:
            continue

        with torch.no_grad():
            # TODO: use multiprocessing  or signal to set up a timeout
            # https://stackoverflow.com/questions/492519/timeout-on-a-function-call

            # (Decoding time limit during first epochs)
            # timer = 1 + (60*e if e > 40 else 0)
            # signal.alarm(timer)
            # try:
            logger.info("Epoch done, evaluating on sample of train")
            train_mrr, train_acc, train_ranks = predict_eval(device, encoder, small_train_statements_loader, small_train_proofs_loader, global_model)
            # except Exception as exception:
            #     logger.warning(exception)
            #     train_mrr, train_acc = -1, -1

            # signal.alarm(timer)
            # try:
            logger.info("Evaluation on train done, evaluating on dev")
            dev_mrr, dev_acc, dev_ranks = predict_eval(device, encoder, dev_statements, dev_proofs, global_model)
            logger.info("Done")
            # except Exception as exception:
            #     logger.warning(exception)
            #     dev_mrr, dev_acc = -1, -1

            try:
                logger.info("Evaluation on train done, evaluating on dev")
                dev_anno_mrr, dev_anno_acc, dev_anno_ranks = predict_eval(device, encoder, dev_anno_statements, dev_anno_proofs, global_model)
                logger.info("Done")
            except Exception as exception:
                logger.warning(exception)
                dev_anno_mrr, dev_anno_acc = -1, -1
            signal.alarm(0)

        if dev_acc > best_acc:
            best_acc = dev_acc
            encoder.cpu()
            torch.save(encoder, "{}/{}".format(args.model, args.model))

            encoder.to(device=device)
        

        summary = "Epoch {} l={:.4f} tmrr={:.2f} tacc={:.2f} dmrr={:.2f} dacc={:.2f} annodmrr={:.2f} annodacc={:.2f} bat acc={:.1f} loc loss={:.4f} lr={:.4f}"
        print(summary.format(e, 
                             tloss,
                             train_mrr*100, train_acc*100,
                             dev_mrr*100, dev_acc*100,
                             dev_anno_mrr*100, dev_anno_acc*100,
                             batch_acc*100, loc_loss,
                             optimizer.param_groups[0]['lr']), flush=True)
        #summary = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
        stat_file.write(stat_line.format(e,
                                         tloss,
                                         train_mrr*100, train_acc*100,
                                         dev_mrr*100, dev_acc*100,
                                         dev_anno_mrr*100, dev_anno_acc*100,
                                         batch_acc*100, loc_loss, optimizer.param_groups[0]['lr']))


def encode_documents(device, encoder, statements, proofs):
    statement_vecs = []
    for batch in statements:
        try:
            batch = {k: torch.stack(v, dim=1).to(device) for k, v in batch.items()}
        except TypeError:
            batch = {k: torch.tensor([v]).to(device) for k, v in batch.items()}
        statement_vecs.append(encoder.encoder(**batch).pooler_output)

    # statement_vecs = np.concatenate(statement_vecs)
    statement_vecs = torch.cat(statement_vecs)
    proof_vecs = []
    for batch in proofs:
        try:
            batch = {k: torch.stack(v, dim=1).to(device) for k, v in batch.items()}
        except TypeError:
            batch = {k: torch.tensor([v]).to(device) for k, v in batch.items()}
        proof_vecs.append(encoder.encoder(**batch).pooler_output)

    # print("================================================\n")
    # print(statement_vecs)
    # print(proof_vecs)
    # proof_vecs = np.concatenate(proof_vecs)
    proof_vecs = torch.cat(proof_vecs)
    # print(proof_vecs.shape)
    return statement_vecs, proof_vecs


def compute_and_print_ranks(sim_fun, statement_vecs, proof_vecs, filename):
    n_examples = statement_vecs.shape[0]
    
    #statement_vecs = statement_vecs.cpu().numpy()
    #proof_vecs = proof_vecs.cpu().numpy()

    logger = logging.getLogger()
    with open(filename, "w") as f:
        for i in range(n_examples):
            if i % 1000 == 0:
                logger.info("Evaluation: {:.2f} %".format(i * 100 / statement_vecs.shape[0]))

            # torch version
            row = statement_vecs[i]

            similarities = sim_fun(row.repeat(n_examples, 1), proof_vecs)
            similarities = similarities.cpu().numpy().flatten()

            ranks, sims = baselines.extract_best_ranking(similarities)

            f.write("{}\t{}\t{}\n".format(i,
                                         " ".join(map(str, ranks)),
                                         " ".join(map(lambda x: str(round(x, 4)), sims))))

def evaluate_dense(statement_vecs, proof_vecs, sim_fun, verbose=False):
    """Match statements to proofs and evaluate with Mean Reciprocal Rank
        Same as evaluate, with dense vectiors instead of sparse vectors
    """
    #n_examples = len(statement_vecs)
    n_examples = statement_vecs.size(0)

    mrr = 0
    acc = 0
    ranks = []
    for i in range(n_examples):
        if i % 100 == 0:
            logging.getLogger().info("Evaluation: {:.2f} %".format(i * 100 / statement_vecs.shape[0]))


        # torch version
        row = statement_vecs[i]
        similarities = sim_fun(row.repeat(n_examples, 1), proof_vecs)
        similarities = similarities.cpu().numpy().flatten()

        """# numpy version
        row = statement_vecs[i, :]
        
        similarities = proof_vecs.dot(row.T)
        """

        gold_score = similarities[i]
        above_score = similarities >= gold_score
        assert(len(above_score.shape) == 1)
        rank = above_score.sum()
        ranks.append(rank)

        #argmax = np.argmax(similarities)
        # rank should be the rank of the gold proof
        if rank == 1:
            acc += 1

        """# Sanity check: 
        best_ranks, best_sims = extract_best_ranking(similarities.flatten(), k=10)
        if i in best_ranks:
            rank_alt = best_ranks.index(i) + 1
            print("Rank, rank alt", rank, " ", rank_alt)
        """

#        print(i, best_ranks, rank)
#        print("    ", " ".join(map(lambda x: str(round(x, 2)), similarities)))
#        print("    ", " ".join(map(lambda x: str(round(x, 2)), best_sims)))

        mrr += 1 / rank
    
    return mrr / n_examples, acc / n_examples, ranks

def do_eval(args, device):
    logger = logging.getLogger()
    if args.model.lower().startswith("scratch"):
        logger.info("Loading scratchbert model")
        tokenizer = AutoTokenizer.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/train_from_scratch/model_files", do_lower_case=True)
    elif args.model.lower().startswith("mathbert"):
        logger.info("Loading mathbert tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/baseline/MathBERT-custom", do_lower_case=True)
    elif args.model.lower().startswith("bert"):
        logger.info("Loading bert tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(script_path.replace("theory-proof-matching/maximin_ori/", "") + "pretrain/baseline/bert-base-uncased", do_lower_case=True)

    def preprocess_theory(examples):
        return tokenizer(examples["theory"], padding='max_length', truncation=True, max_length=args.max_length)

    def preprocess_proof(examples):
        return tokenizer(examples["proof"], padding='max_length', truncation=True, max_length=args.max_length)

    dev_dataset = pd.read_csv(
        args.test_prefix,
        sep=',')
    dev_set = Dataset.from_pandas(dev_dataset)
    dev_statements = dev_set.map(preprocess_theory, batched=True).remove_columns(
        ['Unnamed: 0', 'theory', 'proof', 'meta'])
    dev_proofs = dev_set.map(preprocess_proof, batched=True).remove_columns(['Unnamed: 0', 'proof', 'theory', 'meta'])
    dev_statements_loader = DataLoader(dev_statements, batch_size=20)
    dev_proofs_loader = DataLoader(dev_proofs, batch_size=20)

    logger.info("Loading pytorch model...")
    encoder = torch.load(args.model, map_location=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    logger.info("Encoding documents...")
    statement_vecs, proof_vecs = encode_documents(device, encoder, dev_statements_loader, dev_proofs_loader)
    dev_mrr, dev_acc, dev_ranks = predict_eval(device, encoder, dev_statements, dev_proofs, args.glob)
    print("{:.2f}\t{:.2f}".format(dev_mrr*100, dev_acc*100))
    # np.savetxt("{}/statement_vecs".format(globals.rightnow), statement_vecs.cpu().numpy())
    # np.savetxt("{}/proof_vecs".format(globals.rightnow), proof_vecs.cpu().numpy())
    #
    # logger.info("Computing and printing ranks...")
    # compute_and_print_ranks(encoder.sim, statement_vecs, proof_vecs, "{}/ranks".format(globals.rightnow))
    # compute_and_print_ranks(encoder.sim, statement_vecs, proof_vecs, args.output)

def main(args, device):
    """
        TODO: doc
    """
    #train, dev, test, metadata, all_category_counts, primary_category_counts = baselines.initialize(args)

    logger = logging.getLogger()
    if args.mode == "train":
        logger.info("Training mode")
        do_training(args, device)
    
    elif args.mode == "eval":
        logger.info("Evaluation mode")
        do_eval(args, device)



if __name__ == "__main__":
    import argparse
    #import mkl


    usage = main.__doc__
    #parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(dest="mode", description="Execution modes", help='train: training, eval: test')
    subparsers.required = True

    train_parser = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser = subparsers.add_parser("eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # train corpora
    train_parser.add_argument("model", help="Pytorch model")
    train_parser.add_argument("train", help="Training corpus")
    train_parser.add_argument("dev",   help="Dev corpus")
    train_parser.add_argument("anno_dev",  help="Anno dev corpus")
    # general options
    train_parser.add_argument("--gpu", type=int, default=0,
                              help="Use GPU if available")
    train_parser.add_argument("--threads", type=int, default=8,
                              help="Number of threads for torch cpu")
    train_parser.add_argument("--subset", "-S", type=int, default=None,
                              help="Use only X first training examples")
    train_parser.add_argument("--verbose", "-v", type=int, default=10,
                              help="Logger verbosity, higher is quieter")

    # training options
    train_parser.add_argument("-i", default=30, type=int, help="Number of iterations")
    train_parser.add_argument("-l", default=0.001, type=float, help="Learning rate")

    train_parser.add_argument("--input", type=str, default=BOTH, choices=[BOTH, TEXT, FORMULAE],
                              help="Type of input")

    train_parser.add_argument("-L", default=None, choices=["sigmoid", "softmax", None], 
                              help="Loss for local cross entropy. sigmoid: binary cross entropy")
    train_parser.add_argument("-N", type=int, default=5,
                              help="Number of negative examples")
    train_parser.add_argument("-G", type=int, default=None,
                              help="Activate global training with int:batch size")


    #train_parser.add_argument("--margin", type=float, default=1.0, help="Margin to enforce for each assignment")
    #train_parser.add_argument("--pretrained", default=None, help="Load pretrained local model")


    train_parser.add_argument("-W", type=int, default=100, help="Encoder dim")
    train_parser.add_argument("-d", type=int, default=2, help="Encoder depth")
    train_parser.add_argument("--dk", type=int, default=128, help="Encoder query dim")
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--max-length", type=int, default=200, help="Max length for self attention sequence (to avoid out of memory errors)")

    #train_parser.add_argument("--char-embedding-dim", "-c", type=int, default=50)
#    train_parser.add_argument("--kernel-size", "-k", type=int, default=3)
#    train_parser.add_argument("--intermediate-pooling", "-p", type=int, default=None)

    train_parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "asgd", "adagrad", "adadelta"], 
                            help="Optimizer. pytorch lr defaults: adam=0.001 sgd=required asgd=0.01 adagrad=0.01 adadelta=1.0")

    train_parser.add_argument("--seed", type=int, default=100, help="Random seed")

    # test corpus
    eval_parser.add_argument("model", help="Pytorch model")
    eval_parser.add_argument("test_prefix", help="Test corpus: give the prefix only (without _statements, _proofs)")
    eval_parser.add_argument("output", help="Where to output ranks")

    # test options
    eval_parser.add_argument("--gpu", type=int, default=None, help="Use GPU <int> if available")
    eval_parser.add_argument("--verbose", "-v", type=int, default=10, help="Logger verbosity, higher is quieter")
    eval_parser.add_argument("--threads", type=int, default=8,
                                  help="Number of threads for torch cpu")
    eval_parser.add_argument("--max_length", type=int, default=100, help="Max length of tokenizer")
    eval_parser.add_argument("--glob", type=bool, default=False, help="Global decoding or not")
    eval_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    eval_parser.add_argument("--input", type=str, default="both", help="Input type")

    args = parser.parse_args()

    if args.seed is not None:
        globals.RANDOM_SEED = args.seed
    if args.mode == "eval":
        if args.glob is False or args.glob == 'False':
            args.glob = False
        elif args.glob is True or args.glob == 'True':
            args.glob = True
        else:
            raise ValueError("--glob must be either True or False")
    if args.seed is not None:
        globals.RANDOM_SEED = args.seed
    if args.mode == "train":
        if args.input == "math":
            args.train = args.train.replace(".csv", "_math_only.csv")
            args.dev = args.dev.replace(".csv", "_math_only.csv")
            args.anno_dev = args.anno_dev.replace(".csv", "_math_only.csv")
        if args.input == "text":
            args.train = args.train.replace(".csv", "_text_only.csv")
            args.dev = args.dev.replace(".csv", "_text_only.csv")
            args.anno_dev = args.anno_dev.replace(".csv", "_text_only.csv")
    elif args.mode == "eval":
        if args.input == "math":
            args.test_prefix = args.test_prefix.replace(".csv", "_math_only.csv")
        if args.input == "text":
            args.test_prefix = args.test_prefix.replace(".csv", "_text_only.csv")
    np.random.seed(globals.RANDOM_SEED)
    torch.manual_seed(globals.RANDOM_SEED)

    print("Arguments:")
    for k, v in vars(args).items():
        print(k, v)


    #mkl.set_num_threads(args.threads)
    torch.set_num_threads(args.threads)

    logger = logging.getLogger()
    logger.info("Setting logging level to {}".format(args.verbose))
    logger.setLevel(args.verbose)
    logger.info("Must be silent if {} < {}".format(logger.level, args.verbose))


    if args.mode == "train":
        # TODO: update these ones
        globals.rightnow = "{}/neural_{}_{}_c_w{}_d{}_N{}_".format(
                args.model,
                args.input,
                args.model.replace("/", "_"),
                #args.char_embedding_dim,
                args.W,
                args.d,
                args.L) + globals.rightnow

        logging.info("Logs in {}".format(globals.rightnow))
        os.makedirs(globals.rightnow, exist_ok = True)

        os.makedirs(args.model, exist_ok = True)
        with open("{}/meta".format(args.model), "w") as f:
            f.write("{}\n".format(globals.rightnow))
            f.write("{}\n".format(" ".join(sys.argv)))
            f.write("input-type:{}\n".format(args.input))
    else:
        globals.rightnow = "./neural_eval_{}_".format(
                            args.model.replace("/", "_")) + globals.rightnow
        logging.info("Logs in {}".format(globals.rightnow))
        # os.makedirs(globals.rightnow, exist_ok = True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    use_cuda = torch.cuda.is_available()
    if use_cuda and args.gpu is not None:
        logging.info("Using gpu {}".format(args.gpu))
        device = torch.device("cuda")
    else:
        logging.info("Using cpu")
        device = torch.device("cpu")

    main(args, device)


