
from collections import defaultdict
import traceback
import signal
import lap
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.stats as stats
import logging
import os

import baselines
import bipartite_matching
from globals import *
import globals
from corpus_utils import *

from transformer import Encoder as SelfAttentiveEncoder

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
                d_model, d_inner, dropout=0.1):

        super(TransfoEnc, self).__init__()

        self.encoder = SelfAttentiveEncoder(n_src_vocab, len_max_seq, d_word_vec,
                                            n_layers, n_head, d_k, d_v,
                                            d_model, d_inner, dropout=dropout)
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


def predict_eval(device, encoder, corpus, global_decoding=False):
    statements, proofs = corpus
    # print("start encoding documents (line 151)")
    statement_vecs, proof_vecs = encode_documents(device, encoder, statements, proofs)
    # print("finish encoding documents (line 151)")
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
    # print("===================")
    # print(gold_score)
    # print(pred_score)
    # print(loss)
    loss.backward()
    # print(loss)
    return loss.float()

def compute_similarities_batch(ss, ps, encoder, optimizer, batch_size, MAX_SIZE=8000):
    
    for i in range(len(ss)):
        if len(ss[i]) > MAX_SIZE:
            r = np.random.randint(len(ss[i]-MAX_SIZE))
            ss[i] = ss[i][r:r+MAX_SIZE]
        if len(ps[i]) > MAX_SIZE:
            r = np.random.randint(len(ps[i]-MAX_SIZE))
            ps[i] = ps[i][r:r+MAX_SIZE]

    batch_input = torch.nn.utils.rnn.pad_sequence(ss + ps).transpose(0, 1)

    try:
        encoded_batch = encoder(batch_input)
    except Exception as an_exception:
        logger = logging.getLogger()
        logger.crucial(an_exception)
        logger.crucial("Crashed on example of size {}".format(batch_input.shape))

    encoded_statements, encoded_proofs = encoded_batch.split([batch_size, batch_size])

    # Need to compute: sim for the cross product (statements x proofs)
    # repeated statements: [v1] * batch_size + [v2] * batch_size + ... + [vn] * batch_size
    repeated_statements = encoded_statements.repeat(1, batch_size).view(batch_size**2, -1)

    # repeated statements: [v1, v2, ....., vn] * batch_size
    repeated_proofs = encoded_proofs.repeat(batch_size, 1)

    #print(repeated_statements.shape)
    #print(repeated_proofs.shape)

    #sim_input = torch.cat([repeated_statements, repeated_proofs], dim=1)
    #print(sim_input.shape)
    similarities = encoder.sim(repeated_statements, repeated_proofs).view(batch_size, batch_size)
    #similarities = encoder.sim(sim_input).view(batch_size, batch_size)

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

def train_one_epoch(device, encoder, optimizer, train_tensors, args, epoch_num):
    #(device, encoder, optimizer, train, batch_size, margin=1.0):
    """
        Generic function to train one epoch, global or local loss.
    """
    logger = logging.getLogger()

    statements, proofs = train_tensors
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
        
    for batch in range(n_iterations):

        optimizer.zero_grad()

        next_is_local_step = only_loc or (both and batch % 2 == 1)

        if next_is_local_step:
            next_batch_size = args.N
        else:
            next_batch_size = args.G
        
        # 1. sample batch_size pairs
        sample_ids = [j for j in np.random.choice(N_examples, next_batch_size, replace=False)]
        
        ss = [statements[i] for i in sample_ids]
        ps = [proofs[i] for i in sample_ids]

        try:
            similarities = compute_similarities_batch(ss, ps, encoder, optimizer, next_batch_size)

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
            raise Exception("RuntimeError: {}".format(e))
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

    return normalized_global_loss, normalized_local_loss, batch_acc


def do_training(args, device):

    logger = logging.getLogger()

    train, metatrain, train_cats, train_primary_cats = baselines.initialize_corpus(args.train, "anno_train", args.subset)
    dev, metadev, dev_cats, dev_primary_cats = baselines.initialize_corpus(args.dev, "anno_dev", args.subset)

    # vocabulary map
    vocabulary = get_voc(train)
    i2char = ["<PAD>"] + sorted(vocabulary, key = lambda x : vocabulary[x], reverse=True)
    char2i = {k:i for i, k in enumerate(i2char)}
    
    export_voc(args.model, i2char)

    i2c, c2i = load_voc(args.model)
    assert(c2i == char2i)
    assert(i2c == i2char)
    
    # Encoder: Batch of int tensors -> Batch of fixed size representations

    encoder = TransfoEnc(n_src_vocab=len(i2char), len_max_seq=args.max_length, d_word_vec=args.W,
                         n_layers=args.d, n_head=args.n_heads, d_k=args.dk, d_v=args.dk,
                         d_model=args.W, d_inner=args.W, dropout=0.1)
    encoder.to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(encoder.parameters(), lr=args.l, momentum=0.1, weight_decay=0)
    elif args.optimizer == "asgd":
        optimizer = optim.ASGD(encoder.parameters(), lr=args.l, t0=10000, weight_decay=0)
        # optimizer = optim.ASGD(encoder.parameters(), lr=args.l, weight_decay=0)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(encoder.parameters(), lr=args.l)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(encoder.parameters(), lr=args.l)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(encoder.parameters(), lr=args.l)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(encoder.parameters(), lr=args.l)
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(encoder.parameters(), lr=args.l)
    elif args.optimizer == "nadam":
        optimizer = optim.NAdam(encoder.parameters(), lr=args.l)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.996)

    # Turn texts to tensors and transfer them to GPU
    train_tensors = ([prepare_toks(s.statement, device, args.input, char2i) for s in train],
                     [prepare_toks(s.proof, device, args.input, char2i) for s in train])

    dev_tensors = ([prepare_toks(s.statement, device, args.input, char2i) for s in dev],
                   [prepare_toks(s.proof, device, args.input, char2i) for s in dev])

    n_dev = len(dev_tensors[0])
    sample_train_tensors = (train_tensors[0][:n_dev], train_tensors[1][:n_dev])


    stat_file = open("{}/log_learning".format(args.model), "w", buffering=1)
    stat_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    stat_file.write(stat_line.format("Epoch", "tloss", "tmrr", "tacc", "dmrr", "dacc", "batch_acc", "loc_loss", "lr"))

    global_model = args.G is not None

    best_acc = 0
    for e in tqdm(range(1, args.i + 1)):
        encoder.train()

        for param in encoder.parameters():
            param.requires_grad = True

        tloss, loc_loss, batch_acc = train_one_epoch(device, encoder, optimizer, train_tensors, args, e)
        # print("set requires_grad to false")
        for param in encoder.parameters():
            param.requires_grad = False
        # print("finish set requires_grad to false")
        # print("switching to eval mode")
        encoder.eval()
        # print("finish switching to eval mode")
        if args.l <= 0.002:
            if e >= 300:
                scheduler.step()
        else:
            scheduler.step()

        if e % 5 != 0:
            continue

        with torch.no_grad():
            # TODO: use multiprocessing  or signal to set up a timeout
            # https://stackoverflow.com/questions/492519/timeout-on-a-function-call

            # (Decoding time limit during first epochs)
            # timer = 1 + (60*e if e > 10 else 0)
            # signal.alarm(timer)
            # try:
            logger.info("Epoch done, evaluating on sample of train")
            train_mrr, train_acc, train_ranks = predict_eval(device, encoder, sample_train_tensors, global_model)
            # except Exception as exception:
            #     logger.warning(exception)
            #     train_mrr, train_acc = -1, -1

            # signal.alarm(timer)
            # try:
            logger.info("Evaluation on train done, evaluating on dev")
            dev_mrr, dev_acc, dev_ranks = predict_eval(device, encoder, dev_tensors, global_model)
            logger.info("Done")
            # except Exception as exception:
            #     logger.warning(exception)
            #     dev_mrr, dev_acc = -1, -1

            # signal.alarm(0)

        if dev_acc > best_acc:
            best_acc = dev_acc
            # encoder.cpu()
            torch.save(encoder, "{}/model".format(args.model))
            encoder.to(device=device)
        

        summary = "Epoch {} l={:.4f} tmrr={:.1f} tacc={:.1f} dmrr={:.1f} dacc={:.1f} bat acc={:.1f} loc loss={:.4f} lr={:.4f}"
        print(summary.format(e, 
                             tloss,
                             train_mrr*100, train_acc*100,
                             dev_mrr*100, dev_acc*100,
                             batch_acc*100, loc_loss,
                             optimizer.param_groups[0]['lr']), flush=True)
        #summary = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
        stat_file.write(stat_line.format(e,
                                         tloss,
                                         train_mrr*100, train_acc*100,
                                         dev_mrr*100, dev_acc*100,
                                         batch_acc*100, loc_loss, optimizer.param_groups[0]['lr']))


def encode_documents(device, encoder, statements, proofs):

    batch_size = 25

    statement_vecs = []
    for i in range(0, len(statements), batch_size):
        statement_vec_input = torch.nn.utils.rnn.pad_sequence(statements[i:i+batch_size]).transpose(0, 1)
        #print(statement_vec_input.shape)
        rows = statement_vec_input.size(0)
        cols = statement_vec_input.size(1)
        if cols < 100:
            statement_vec_input = torch.cat([statement_vec_input, torch.zeros(rows, 100, dtype=torch.long).to(device)], 1)
        #statement_vecs.append(encoder(statement_vec_input).cpu().numpy())
        statement_vecs.append(encoder(statement_vec_input))

    #statement_vecs = np.concatenate(statement_vecs)
    statement_vecs = torch.cat(statement_vecs)

    proof_vecs = []
    for i in range(0, len(statements), batch_size):
        proof_vec_input = torch.nn.utils.rnn.pad_sequence(proofs[i:i+batch_size]).transpose(0, 1)
        rows = proof_vec_input.size(0)
        cols = proof_vec_input.size(1)
        if cols < 100:
            proof_vec_input = torch.cat([proof_vec_input, torch.zeros(rows, 100, dtype=torch.long).to(device)], 1)
        #proof_vecs.append(encoder(proof_vec_input).cpu().numpy())
        proof_vecs.append(encoder(proof_vec_input))
    # print(statement_vecs[0])
    # print(proof_vecs[0])
    #proof_vecs = np.concatenate(proof_vecs)
    proof_vecs = torch.cat(proof_vecs)

    return statement_vecs, proof_vecs


def compute_and_print_ranks(sim_fun, statement_vecs, proof_vecs, filename):
    n_examples = statement_vecs.shape[0]
    
    #statement_vecs = statement_vecs.cpu().numpy()
    #proof_vecs = proof_vecs.cpu().numpy()

    logger = logging.getLogger()
    with open(filename, "w") as f:
        for i in range(n_examples):
            if i % 5000 == 0:
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
        if i % 5000 == 0:
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

    # logger.info("Loading statements...")
    # statements, num_statements = load_raw("{}_statements".format(args.test_prefix))
    #
    # logger.info("Loading proofs...")
    # proofs, num_proofs = load_raw("{}_proofs".format(args.test_prefix))

    dev, metadev, dev_cats, dev_primary_cats = baselines.initialize_corpus(args.test_prefix, "anno_dev", None)
    # assert(num_statements == num_proofs)

    logger.info("Loading vocabulary..")
    i2char, char2i = load_voc(args.model)

    logger.info("Loading input type...")
    input_type = None
    with open("{}/meta".format(args.model)) as f:
        for line in f:
            if line.startswith("input-type"):
                input_type = line.strip().split(':')[1]
                logging.info("input-type:{}".format(input_type))
    assert (input_type is not None)

    logger.info("Loading pytorch model...")
    # encoder = torch.load(args.model + '/' + args.model.replace('_only', ''), map_location=device)
    try:
        encoder = torch.load(args.model + '/model', map_location=device)
    except:
        encoder = torch.load(args.model + '/' + args.model, map_location=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dev_tensors = ([prepare_toks(s.statement, device, args.input, char2i) for s in dev],
                   [prepare_toks(s.proof, device, args.input, char2i) for s in dev])
    if "train" in args.test_prefix:
        dev_tensors = (dev_tensors[0][:18408], dev_tensors[1][:18408])
    # logger.info("Encoding documents...")
    # statement_vecs, proof_vecs = encode_documents(device, encoder, stat_tensors, proo_tensors)

    # np.savetxt("{}/statement_vecs".format(globals.rightnow), statement_vecs.cpu().numpy())
    # np.savetxt("{}/proof_vecs".format(globals.rightnow), proof_vecs.cpu().numpy())

    logger.info("Computing and printing ranks...")
    # compute_and_print_ranks(encoder.sim, statement_vecs, proof_vecs, "{}/ranks".format(globals.rightnow))
    # compute_and_print_ranks(encoder.sim, statement_vecs, proof_vecs, args.output)
    dev_mrr, dev_acc, dev_ranks = predict_eval(device, encoder, dev_tensors, args.glob)
    print("{:.2f}\t{:.2f}".format(dev_mrr * 100, dev_acc * 100))

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
    train_parser.add_argument("--glob", type=bool, default=False, help="Global training or not")
    #train_parser.add_argument("--char-embedding-dim", "-c", type=int, default=50)
#    train_parser.add_argument("--kernel-size", "-k", type=int, default=3)
#    train_parser.add_argument("--intermediate-pooling", "-p", type=int, default=None)

    train_parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "asgd", "adagrad", "adadelta", "nadam", "rmsprop", "adamax"],
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
    eval_parser.add_argument("--glob", type=bool, default=False, help="Global decoding or not")
    eval_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    eval_parser.add_argument("--input", type=str, default=BOTH, choices=[BOTH, TEXT, FORMULAE],
                             help="Type of input")

    args = parser.parse_args()

    if args.seed is not None:
        globals.RANDOM_SEED = args.seed
    if args.glob is False or args.glob == 'False':
        args.glob = False
    elif args.glob is True or args.glob == 'True':
        args.glob = True
    else:
        raise ValueError("--glob must be either True or False")
    np.random.seed(globals.RANDOM_SEED)
    torch.manual_seed(globals.RANDOM_SEED)
    args.glob = False
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
        globals.rightnow = "{}/neural_eval_{}_".format(
                            args.model,
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


