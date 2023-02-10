import pdb

from tqdm import tqdm
import string
import random
import os
import re
import pickle
import enchant
import pandas as pd

greek_letters = 'αβγδεζηθικλμνξορστυφχψω'
convert_theorem = False

domain_filter = {
    # 'math.PR': ['p', 'e', 'v', 'σ', 'ρ']
    'math.PR': []
}
# for file in os.listdir("G:/maximin_dataset/dataset_09_05/"):
#     os.rename("G:/maximin_dataset/dataset_09_05/"+ file, "G:/maximin_dataset/dataset_09_05/" + '_'.join(file.split('_')[:5]) + '.' + file.split('.')[-1])


def select_pairs(directory, num_of_pairs):
    if os.path.exists('./selected_by_proportion_dic_{}.pkl'.format(num_of_pairs)):
        with open('./selected_by_proportion_dic_{}.pkl'.format(num_of_pairs), 'rb') as f:
            selected_by_proportion_dic = pickle.load(f)
        f.close()
        with open('./selected_by_proportion_list_{}.pkl'.format(num_of_pairs), 'rb') as f:
            selected_by_proportion_list = pickle.load(f)
        f.close()
        return selected_by_proportion_dic, selected_by_proportion_list

    domain_num_dict = {}
    pairs_by_category = {}
    selected_by_proportion_dic = {}
    selected_by_proportion_list = []
    pairs_frame = pd.DataFrame(columns=['pair', 'category'])
    pairs_frame = pd.read_csv('./pair_frame.csv')
    files = os.listdir(directory)
    categories = pairs_frame['category'].unique()
    for category in tqdm(categories):
        pairs_by_category[category] = pairs_frame[pairs_frame['category'] == category]['pair'].tolist()
        domain_num_dict[category] = len(pairs_by_category[category])
    sum_of_pairs = len(pairs_frame)
    for category in tqdm(categories):
        pairs_needed = round(num_of_pairs * domain_num_dict[category] / sum_of_pairs)
        selected_by_proportion_dic[category] = random.sample(pairs_by_category[category], pairs_needed)
        selected_by_proportion_list += selected_by_proportion_dic[category]
    pickle.dump(selected_by_proportion_dic, open('./selected_by_proportion_dic_{}.pkl'.format(num_of_pairs), 'wb'))
    pickle.dump(selected_by_proportion_list, open('./selected_by_proportion_list_{}.pkl'.format(num_of_pairs), 'wb'))
    return selected_by_proportion_dic, selected_by_proportion_list

    # for idx, file in tqdm(enumerate(files), total=len(files)):
    #     with open(directory+'/'+file, encoding='utf-8') as f:
    #         text = f.read()
    #         meta = text.split('<meta arxivid="')[1].split('"')[0].replace('arXiv:', '')
    #         detail_cat = text.split('category="')[1].split('"')[0].split(' ')[0]
    #         pairs = text.split('</pair>')
    #         for pair in pairs:
    #             if '</root>' in pair:
    #                 continue
    #             if '<root' in pair:
    #                 pair = '<pair>\n' + pair.split('<pair>')[1]
    #             pairs_frame = pairs_frame.append({'pair': pair, 'category': detail_cat, 'meta': meta}, ignore_index=True)
    #     f.close()
    # pairs_frame.to_csv('./pair_frame.csv', index=False)


    return pairs_frame


def get_domain_dict():
    with open(
            'D:/OneDrive - University of Edinburgh/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/data/corpus_train',
            'r', encoding='utf-8') as meta_corpus:
        domain_dict = {}
        meta_text = meta_corpus.read()
        meta = meta_text.split('\n\n')[1]
        for line in meta.split('\n'):
            index = line.split('\t')[0].replace('arXiv:', '')
            category = line.split('\t')[1].split('|')
            for cat in category:
                if cat not in domain_dict:
                    domain_dict[cat] = []
                domain_dict[cat].append(index)
        meta_corpus.close()
        return domain_dict


def full_anonymize(variables, proof_only_vars, candidates=None, domain=None):
    def construct_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</m:mi>'.format(type, var) if type != 'none' else '<m:mi>{}</m:mi>'.format(var)
    def construct_label_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</labelmi>'.format(type, var) if type != 'none' else '<m:mi>{}</labelmi>'.format(var)
    mapping = dict()
    global_candidates = set(string.ascii_lowercase) if candidates is None else set(candidates['english'])
    local_greek_letters = set(greek_letters) if candidates is None else set(candidates['greek'])
    global_mapping = dict()
    proof_only_vars_temp = []
    for item in proof_only_vars:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (
        item.split('>')[1][0], 'none')
        proof_only_vars_temp.append((var, type))
    global_greek_candidates = set(local_greek_letters) - set([var.lower() for var,_ in proof_only_vars_temp])
    global_letter_candidates = global_candidates - set([var.lower() for var,_ in proof_only_vars_temp])
    if domain is not None:
        global_letter_candidates = global_letter_candidates - set(domain_filter[domain])
        global_greek_candidates = global_greek_candidates - set(domain_filter[domain])
    for item in variables:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (item.split('>')[1][0], 'none')
        if domain is not None:
            if var.lower() in domain_filter[domain]:
                continue
        if type == 'double-struck':
            continue
        elif var.lower() in global_mapping.keys():
            target_var = global_mapping[var.lower()]
            mapping[item] = (target_var.upper(), type) if 'A' <= var <= 'Z' else (target_var.lower(), type)
        elif 'A' <= var <= 'Z':
            if len(global_letter_candidates) > 1:
                target_var = random.choice(list(global_letter_candidates-set([var.lower()])))
                mapping[item] = (target_var.upper(), type)
                global_mapping[var.lower()] = target_var
                global_letter_candidates.remove(target_var.lower())
        elif 'a' <= var <= 'z':
            if len(global_letter_candidates) > 1:
                target_var = random.choice(list(global_letter_candidates-set([var])))
                mapping[item] = (target_var, type)
                global_mapping[var.lower()] = target_var
                global_letter_candidates.remove(target_var.lower())
        elif var in greek_letters and var not in mapping.keys():
            if len(global_greek_candidates) > 1:
                target_var = random.choice(list(global_greek_candidates-set([var])))
                mapping[item] = (target_var, type)
                global_greek_candidates.remove(target_var)
        # except IndexError:
        #     print("==========================")
        #     print(len(mapping))
        #     print(mapping)
        #     print(var, type)
        #     print(global_mapping)
        #     raise IndexError("Index error")
        for item in proof_only_vars:
            var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (item.split('>')[1][0], 'none')
            if var.lower() in global_mapping.keys() and type != 'double-struck':
                target_var = global_mapping[var.lower()]
                mapping[item] = (target_var.upper(), type) if 'A' <= var <= 'Z' else (target_var.lower(), type)

    return [(var_key, construct_label_mi_text(mapping[var_key][0], mapping[var_key][1])) for var_key in mapping.keys()]


def adversarial_anonymize(variables, domain=None):
    def construct_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</m:mi>'.format(type, var) if type != 'none' else '<m:mi>{}</m:mi>'.format(var)
    def construct_label_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</labelmi>'.format(type, var) if type != 'none' else '<m:mi>{}</labelmi>'.format(var)
    global_mapping = dict()
    global_letter_candidates = set()
    global_greek_candidates = set()
    mapping = []
    for item in variables:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (item.split('>')[1][0], 'none')
        if domain is not None:
            if var.lower() in domain_filter[domain]:
                continue
        if type == 'double-struck':
            continue
        elif 'A' <= var <= 'Z' or 'a' <= var <= 'z':
            global_letter_candidates.add(var.lower())
        elif var in greek_letters:
            global_greek_candidates.add(var.lower())
    global_letter_candidates_list = list(global_letter_candidates)
    global_greek_candidates_list = list(global_greek_candidates)
    last_key = None
    if len(global_letter_candidates_list) > 1:
        for candidate in global_letter_candidates_list:
            try:
                target_var = random.choice(list(global_letter_candidates-set([candidate])))
            except:
                target_var = global_mapping[last_key]
                global_mapping[last_key] = candidate.lower()
                global_mapping[candidate.lower()] = target_var
                continue
            global_letter_candidates.remove(target_var)
            global_mapping[candidate.lower()] = target_var
            last_key = candidate.lower()
    elif len(global_letter_candidates_list) == 1:
        candidate = global_letter_candidates_list[0].lower()
        global_mapping[candidate] = candidate
    if len(global_greek_candidates_list) > 1:
        for candidate in global_greek_candidates_list:
            try:
                target_var = random.choice(list(global_greek_candidates-set([candidate])))
            except:
                target_var = global_mapping[last_key]
                global_mapping[last_key] = candidate.lower()
                global_mapping[candidate.lower()] = target_var
                continue
            global_greek_candidates.remove(target_var)
            global_mapping[candidate.lower()] = target_var
            last_key = candidate.lower()
    elif len(global_greek_candidates_list) == 1:
        candidate = global_greek_candidates_list[0]
        global_mapping[candidate] = candidate

    for item in variables:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (
        item.split('>')[1][0], 'none')
        if domain is not None:
            if var.lower() in domain_filter[domain]:
                continue
        try:
            new_mapping = (item, construct_label_mi_text(global_mapping[var.lower()].upper(), type)) if 'A' <= var <= 'Z' else (construct_mi_text(var, type), construct_label_mi_text(global_mapping[var], type))
            mapping.append(new_mapping)
        except:
            print(variables)
            print(global_mapping)
            raise Exception("error detected!")


    return mapping


def partial_anonymize(variables, proof_only_vars, range=0.5, candidates=None, domain=None):
    def construct_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</m:mi>'.format(type, var) if type != 'none' else '<m:mi>{}</m:mi>'.format(var)
    def construct_label_mi_text(var, type):
        return '<m:mi mathvariant="{}">{}</labelmi>'.format(type, var) if type != 'none' else '<m:mi>{}</labelmi>'.format(var)

    proof_only_vars_temp = []
    for item in proof_only_vars:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (
        item.split('>')[1][0], 'none')
        proof_only_vars_temp.append((var, type))
    mapping = dict()
    global_candidates = set(string.ascii_lowercase) if candidates is None else set(candidates['english'])
    local_greek_letters = set(greek_letters) if candidates is None else set(candidates['greek'])
    global_mapping = dict()
    global_greek_candidates = set(local_greek_letters) - set([var.lower() for var, _ in proof_only_vars_temp])
    global_letter_candidates = global_candidates - set([var.lower() for var, _ in proof_only_vars_temp])
    if domain is not None:
        global_letter_candidates = global_letter_candidates - set(domain_filter[domain])
        global_greek_candidates = global_greek_candidates - set(domain_filter[domain])
    sample_variables = random.sample(variables, round(len(variables) * range))
    remaining_vars = set(variables) - set(sample_variables)
    assert len(remaining_vars) + len(sample_variables) == len(variables)
    for item in sample_variables:
        # try:
        var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (
        item.split('>')[1][0], 'none')
        if domain is not None:
            if var.lower() in domain_filter[domain]:
                continue
        try:
            if type == 'double-struck':
                continue
            elif var.lower() in global_mapping.keys():
                target_var = global_mapping[var.lower()]
                mapping[item] = (target_var.upper(), type) if 'A' <= var <= 'Z' else (target_var.lower(), type)
            elif 'A' <= var <= 'Z':
                if len(global_letter_candidates) > 1:
                    target_var = random.choice(list(global_letter_candidates-set([var.lower()])))
                    mapping[item] = (target_var.upper(), type)
                    global_mapping[var.lower()] = target_var
                    global_letter_candidates.remove(target_var.lower())
            elif 'a' <= var <= 'z':
                if len(global_letter_candidates) > 1:
                    target_var = random.choice(list(global_letter_candidates-set([var])))
                    mapping[item] = (target_var, type)
                    global_mapping[var.lower()] = target_var
                    global_letter_candidates.remove(target_var.lower())
            elif var in greek_letters and var not in mapping.keys():
                if len(global_greek_candidates) > 1:
                    target_var = random.choice(list(global_greek_candidates-set([var])))
                    mapping[item] = (target_var, type)
                    global_greek_candidates.remove(target_var)
        except IndexError:
            print("==========================")
            print(len(mapping))
            print(mapping)
            print(var, type)
            print(global_mapping)
            raise IndexError("Index error")
        for item in list(proof_only_vars) + list(remaining_vars):
            var, type = (item.split('>')[1][0], item.split('"')[1]) if item.find('mathvariant') > 0 else (
                item.split('>')[1][0], 'none')
            if var.lower() in global_mapping.keys() and type != 'double-struck':
                target_var = global_mapping[var.lower()]
                mapping[item] = (target_var.upper(), type) if 'A' <= var <= 'Z' else (target_var.lower(), type)
    return [(var_key, construct_label_mi_text(mapping[var_key][0], mapping[var_key][1])) for var_key in mapping.keys()]


def find_format(directory):
    if not directory.endswith('\\'):
        directory += '\\'

    files = os.listdir(directory)
    output = set()
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
                output = output.union(set(re.findall(r'____[a-z]*', text)))
            f.close()

    print(output)


def run_anonymize(directory, output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=False, domain=None):
    assert type in ['zero', 'full', 'partial', 'adversarial']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[int(start):]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[int(start):int(end)]
        else:
            files = os.listdir(directory)[int(start):]

    file_num = start if start is not None else 0
    output_str = ''
    domain_active = False

    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
                detail_cat = text.split('category="')[1].split('"')[0].split(' ')
                if not domain in detail_cat:
                    continue
                pairs = text.split('</pair>')
                for pair in pairs:
                    try:
                        theorem, ori_proof = pair.split('</theorem>')
                    except:
                        if '</root>' in pair:
                            continue
                        else:
                            raise Exception("error")
                    groups = check_word(ori_proof)
                    for before in groups.keys():
                        theorem = theorem.replace(before, groups[before])
                        ori_proof = ori_proof.replace(before, groups[before])
                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])
                    theorem_vars_temp = [var.split('>')[1][0] for var in theorem_variables_set]
                    proof_vars_temp = [var.split('>')[1][0] for var in proof_variables_set]
                    common_vars_temp = list(set(theorem_vars_temp) & set(proof_vars_temp))
                    common_variables_set = proof_variables_set.copy()
                    for item in proof_variables_set:
                        if item.split('>')[1][0] not in common_vars_temp:
                            common_variables_set.remove(item)
                    proof_only_variables_set = proof_variables_set - common_variables_set

                    if type == 'full':
                        mapping = full_anonymize(common_variables_set, proof_only_variables_set, domain=domain)
                    elif type == 'adversarial':
                        mapping = adversarial_anonymize(common_variables_set, domain=domain)
                    elif type == 'partial':
                        mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range, domain=domain)
                    elif type == 'zero':
                        mapping = None

                    proof = ori_proof
                    if mapping is not None:
                        for before, after in mapping:
                            proof = proof.replace(before, after)
                            if convert_theorem:
                                theorem = theorem.replace(before, after)
                        proof = proof.replace('</labelmi>', '</m:mi>')
                        if convert_theorem:
                            theorem = theorem.replace('</labelmi>', '</m:mi>')
                    if not convert_to_readable:
                        output_str = output_str + theorem + '</theorem>' + proof + '</pair>'
                    else:
                        output_str = output_str + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>' + '</pair>\n' + '<p>' + str(mapping) + '</p>' + '\n==================================================\n'
                        output_str = output_str.replace('<m:', '<').replace('</m:', '</')

                output_str += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(output_dir+file, 'w', encoding='utf-8') as f_out:
                f_out.write(output_str)
                output_str = ''
            f_out.close()
            file_num += 1
            f.close()


def check_word(text):
    black_list = {
        'Bar', 'Apt', 'fin', 'Sin', 'POS', 'deg', 'int', 'dim', 'inf', 'Dim', 'bit', 'Tan', 'Fix', 'CNN',
        'Var', 'Rep', 'sgd', 'Opt', 'phi', 'Sim', 'fit', 'End', 'std', 'add', 'Odd', 'Min', 'max', 'Chi',
        'sup', 'tan', 'Bit', 'sum', 'sin', 'Hab', 'fum', 'Log', 'Xxv' 'xvi', 'vii', 'map', 'Res', 'Avg',
        'iii', 'log', 'xiv', 'Max', 'Exp', 'xxv', 'top', 'MIN', 'III', 'cos',
        'lots', 'Church', 'Chow', 'Origin', 'Tight', 'Cacti', 'admissible', 'Supp', 'Then', 'range', 'Graph', 'width',
        'stab', 'right', 'Null', 'length', 'spec', 'dist', 'rang', 'Crit', 'Gram', 'resp', 'viii', 'votes',
        'Step', 'Even', 'index', 'char', 'Coll', 'hypo', 'grad', 'Case', 'Prop', 'Ball', 'Belt', 'span', 'PATH', 'NULL',
        'gens', 'Disc', 'Index', 'with', 'Shad', 'Norm', 'constant', 'poly', 'Prob', 'even', 'count', 'dots', 'type',
        'Stab', 'Zero', 'Fred', 'Angle', 'Status', 'case', 'Diff', 'Sing', 'Proc', 'root', 'domain', 'cone', 'Tower',
        'area', 'class', 'Remark', 'Link', 'card', 'Sign', 'comp', 'left', 'disc', 'diam', 'Fort', 'Coin', 'image',
        'dept', 'Trace', 'arcs', 'Area', 'first', 'elements', 'Size', 'Dist', 'Vert', 'part', 'denom', 'supp', 'Hess', 'dens',
        'dads', 'tail', 'baba', 'where', 'Spin', 'meas', 'false', 'Span', 'weak', 'sign', 'point', 'head', 'Spin',
        'respectively', 'relations', 'Type', 'diag', 'rank', 'true', 'rand', 'graph', 'Comm', 'Chain', 'Spec', 'wheres', 'xiii'
    }
    mapping = dict()
    ori_vars = set(re.findall(
        r'(<m:mi>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>(<m:mi mathvariant="[a-z]+">[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>|<m:mi>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>)+)', text) \
           + re.findall(r'(<m:mi mathvariant="[a-z]+">[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>(<m:mi mathvariant="[a-z]+">[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>|<m:mi>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>)+)', text))
    ori_vars = [var[0] for var in ori_vars]
    for var in ori_vars:
        variables = re.findall(r'>[A-Za-zαβγδεζηθικλμνξορστυφχψω]<', var)
        types = set([type.split('"')[1] for type in re.findall(r'mathvariant="[a-z]+"', var)])
        if len(types) > 1:
            continue
        else:
            type = 'none' if len(types) == 0 else types.pop()
        variables = [var.replace('<','').replace('>','') for var in variables]
        token = ''.join(variables)
        if token in black_list:
            mapping[var] = '<m:mi>' + token + '</m:mi>' if type == 'none' else '<m:mi mathvariant="' + type + '">' + token + '</m:mi>'
    # if len(mapping.keys()) > 0:
    #     print(mapping)
    return mapping



def main(args):
    # select_pairs(args.input_dir, 10000)
    selected_pairs_dic, select_pairs_list = select_pairs(args.input_dir, 50000)
    run_anonymize(args.input_dir, args.output_dir+"/", type=args.type, start=args.start, end=args.end, convert_to_readable=False, convert_theorem=convert_theorem)


if __name__ == "__main__":

    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", help="input directory")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("type", help="anonymization type")
    parser.add_argument("--start", default=None, help="start index")
    parser.add_argument("--end", default=None, help="end index")
    args = parser.parse_args()

    main(args)