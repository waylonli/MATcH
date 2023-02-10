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


def run_pair_naive_proportion(pos_output_dir, neg_output_dir, type, partial_range=0.5, convert_to_readable=False, convert_theorem=False, pairs=None):
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    output_str_pos = ''
    output_str_neg = ''
    assert pairs is not None
    file_num = 0
    for pair_idx, pair in tqdm(enumerate(pairs), total=len(pairs)):
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
        pair_aug = None
        while pair_aug is None:
            pair_aug = random.choice(pairs)
            try:
                proof_aug = pair_aug.split('</theorem>')[1]
            except:
                if '</root>' in pair_aug:
                    pair_aug = None
                    continue
                else:
                    print(pair_aug)
                    raise Exception("error")
        for before in groups.keys():
            proof_aug = proof_aug.replace(before, groups[before])


        theorem_variables_set = set([var for var in re.findall(
            r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
            r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
        proof_variables_set = set([var for var in re.findall(
            r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
            r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])
        # print(theorem_variables_set)
        # print(proof_variables_set)
        theorem_vars_temp = [var.split('>')[1][0] for var in theorem_variables_set]
        proof_vars_temp = [var.split('>')[1][0] for var in proof_variables_set]
        common_vars_temp = list(set(theorem_vars_temp) & set(proof_vars_temp))
        common_variables_set = proof_variables_set.copy()
        for item in proof_variables_set:
            if item.split('>')[1][0] not in common_vars_temp:
                common_variables_set.remove(item)
        proof_only_variables_set = proof_variables_set - common_variables_set


        if type == 'full':
            ori_mapping = full_anonymize(common_variables_set, proof_only_variables_set)
        elif type == 'adversarial':
            ori_mapping = adversarial_anonymize(common_variables_set)
        elif type == 'partial':
            ori_mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range)
        elif type == 'zero':
            ori_mapping = None


        candidates = {'greek': [], 'english': []}
        for var_aug in theorem_variables_set:
            var_aug = var_aug.split('>')[1][0]
            if var_aug in greek_letters:
                candidates['greek'].append(var_aug)
            else:
                candidates['english'].append(var_aug.lower())
        for var_aug in proof_variables_set:
            var_aug = var_aug.split('>')[1][0]
            if var_aug in greek_letters:
                candidates['greek'].append(var_aug)
            else:
                candidates['english'].append(var_aug.lower())

        proof = ori_proof

        if ori_mapping is not None:
            for before, after in ori_mapping:
                proof = proof.replace(before, after)
            proof = proof.replace('</labelmi>', '</m:mi>')


        if not convert_to_readable:
            output_str_pos = output_str_pos + theorem + '</theorem>' + proof + '</pair>'
            output_str_neg = output_str_neg + theorem + '</theorem>' + proof_aug + '</pair>'
        else:
            new_mappings = []
            for old_mapping in [ori_mapping, mapping]:
                new_mapping = set()
                for mapping in old_mapping:
                    before, after = mapping
                    before_var = before.split('>')[1][0]
                    after_var = after.split('>')[1][0]
                    new_mapping.add((before_var, after_var))
                new_mappings.append(new_mapping)
            output_str_pos = output_str_pos + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[0]) + '</p>'+ '</p>' + '<p>' + proof_aug + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[1]) + '</p>' + '\n==================================================\n'
            output_str_pos = output_str_pos.replace('<m:', '<').replace('</m:', '</')

        if (pair_idx+1) % 20 == 0:
            output_str_pos += '\n</root>'
            output_str_neg += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                output_str_pos = '<root>\n' + output_str_pos.replace('<root>', '')
                f_out.write(output_str_pos)
                output_str_pos = ''
            f_out.close()
            with open(neg_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                output_str_neg = '<root>\n' + output_str_neg.replace('<root>', '')
                f_out.write(output_str_neg)
                output_str_neg = ''
            f_out.close()
            file_num += 1


def run_pair_naive(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=False):
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    domain_dict = get_domain_dict()
    output_str_pos = ''
    output_str_neg = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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

                    aug_file = random.choice(files)
                    while aug_file == file:
                        aug_file = random.choice(files)
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    proof_aug = pair_aug.split('</theorem>')[1]
                    for before in groups.keys():
                        proof_aug = proof_aug.replace(before, groups[before])


                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])
                    # print(theorem_variables_set)
                    # print(proof_variables_set)
                    theorem_vars_temp = [var.split('>')[1][0] for var in theorem_variables_set]
                    proof_vars_temp = [var.split('>')[1][0] for var in proof_variables_set]
                    common_vars_temp = list(set(theorem_vars_temp) & set(proof_vars_temp))
                    common_variables_set = proof_variables_set.copy()
                    for item in proof_variables_set:
                        if item.split('>')[1][0] not in common_vars_temp:
                            common_variables_set.remove(item)
                    proof_only_variables_set = proof_variables_set - common_variables_set


                    if type == 'full':
                        ori_mapping = full_anonymize(common_variables_set, proof_only_variables_set)
                    elif type == 'adversarial':
                        ori_mapping = adversarial_anonymize(common_variables_set)
                    elif type == 'partial':
                        ori_mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range)
                    elif type == 'zero':
                        ori_mapping = None


                    candidates = {'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())

                    proof = ori_proof

                    if ori_mapping is not None:
                        for before, after in ori_mapping:
                            proof = proof.replace(before, after)
                        proof = proof.replace('</labelmi>', '</m:mi>')


                    if not convert_to_readable:
                        output_str_pos = output_str_pos + theorem + '</theorem>' + proof + '</pair>'
                        output_str_neg = output_str_neg + theorem + '</theorem>' + proof_aug + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [ori_mapping, mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_pos = output_str_pos + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[0]) + '</p>'+ '</p>' + '<p>' + proof_aug + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[1]) + '</p>' + '\n==================================================\n'
                        output_str_pos = output_str_pos.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_pos += '\n</root>'
                output_str_neg += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_pos:
                    output_str_pos = '<root>\n' + output_str_pos
                f_out.write(output_str_pos)
                output_str_pos = ''
            f_out.close()
            with open(neg_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_neg:
                    output_str_neg = '<root>\n' + output_str_neg
                f_out.write(output_str_neg)
                output_str_neg = ''
            file_num += 1
            f.close()


def run_self_naive(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=True):
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_1 = ''
    output_str_2 = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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

                    aug_file = random.choice(files)
                    while aug_file == file:
                        aug_file = random.choice(files)
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    theorem_aug, proof_aug = pair_aug.split('</theorem>')
                    if '<root' in theorem_aug:
                        theorem_aug = '<pair>\n' + theorem_aug.split('<pair>')[1]
                    for before in groups.keys():
                        theorem_aug = theorem_aug.replace(before, groups[before])
                        proof_aug = proof_aug.replace(before, groups[before])

                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])

                    theorem_candidates={'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            theorem_candidates['greek'].append(var_aug)
                        else:
                            theorem_candidates['english'].append(var_aug.lower())
                    proof_candidates = {'greek': [], 'english': []}
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            proof_candidates['greek'].append(var_aug)
                        else:
                            proof_candidates['english'].append(var_aug.lower())

                    if type == 'full':
                        theorem_mapping = full_anonymize(theorem_variables_set, [])
                        # if len(theorem_mapping) == 0:
                        #     print(theorem_candidates)
                        #     print(theorem)
                        proof_mapping = full_anonymize(proof_variables_set, [])
                    elif type == 'partial':
                        theorem_mapping = partial_anonymize(theorem_variables_set, [], partial_range)
                        proof_mapping = partial_anonymize(proof_variables_set, [], partial_range)
                    elif type == 'zero':
                        theorem_mapping = None

                    proof = ori_proof
                    theorem_partial = theorem
                    if theorem_mapping is not None:
                        for before, after in theorem_mapping:
                            theorem_partial = theorem_partial.replace(before, after)
                        theorem_partial = theorem_partial.replace('</labelmi>', '</m:mi>')
                    proof_partial = proof
                    if proof_mapping is not None:
                        for before, after in proof_mapping:
                            proof_partial = proof_partial.replace(before, after)
                        proof_partial = proof_partial.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        output_str_1 = output_str_1 + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        output_str_2 = output_str_2 + theorem_aug + '</theorem>' + proof_aug + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [theorem_mapping, proof_mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_1 = output_str_1 + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + theorem_partial + '</p>' + '<p>' + str(new_mappings[0]) + '</p>' + '<p>' + theorem_aug_partial + '</p>' + '<p>' + str(new_mappings[2]) + '</p>' + '</pair>\n' + '\n==================================================\n'
                        output_str_1 = output_str_1.replace('<m:', '<').replace('</m:', '</')
                        output_str_2 = output_str_2 + '<p>' + ori_proof + '</theorem>' + '</p>' + '<p>' + proof_partial + '</p>' + '<p>' + str(new_mappings[1]) + '</p>' + '<p>' + proof_aug_partial + '</p>' + '<p>' + str(new_mappings[3]) + '</p>' + '</pair>\n' +  '\n==================================================\n'
                        output_str_2 = output_str_2.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_1 += '\n</root>'
                output_str_2 += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_1:
                    output_str_1 = '<root>\n' + output_str_1
                f_out.write(output_str_1)
                output_str_1 = ''
            f_out.close()
            with open(neg_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_2:
                    output_str_2 = '<root>\n' + output_str_2
                f_out.write(output_str_2)
                output_str_2 = ''
            file_num += 1
            f.close()


def run_pair_intersec(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=False):
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_pos = ''
    output_str_neg = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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

                    aug_file = random.choice(files)
                    while aug_file == file:
                        aug_file = random.choice(files)
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    proof_aug = pair_aug.split('</theorem>')[1]
                    for before in groups.keys():
                        proof_aug = proof_aug.replace(before, groups[before])
                    proof_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+">[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', proof_aug)])

                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])
                    # print(theorem_variables_set)
                    # print(proof_variables_set)
                    theorem_vars_temp = [var.split('>')[1][0] for var in theorem_variables_set]
                    proof_vars_temp = [var.split('>')[1][0] for var in proof_variables_set]
                    common_vars_temp = list(set(theorem_vars_temp) & set(proof_vars_temp))
                    common_variables_set = proof_variables_set.copy()
                    for item in proof_variables_set:
                        if item.split('>')[1][0] not in common_vars_temp:
                            common_variables_set.remove(item)
                    proof_only_variables_set = proof_variables_set - common_variables_set


                    if type == 'full':
                        ori_mapping = full_anonymize(common_variables_set, proof_only_variables_set)
                    elif type == 'adversarial':
                        ori_mapping = adversarial_anonymize(common_variables_set)
                    elif type == 'partial':
                        ori_mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range)
                    elif type == 'zero':
                        ori_mapping = None


                    candidates = {'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())


                    mapping = full_anonymize(proof_aug_variables_set, [], candidates=candidates)

                    proof = ori_proof

                    if ori_mapping is not None:
                        for before, after in ori_mapping:
                            proof = proof.replace(before, after)
                        proof = proof.replace('</labelmi>', '</m:mi>')

                    if mapping is not None:
                        for before, after in mapping:
                            proof_aug = proof_aug.replace(before, after)
                        proof_aug = proof_aug.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        output_str_pos = output_str_pos + theorem + '</theorem>' + proof + '</pair>'
                        output_str_neg = output_str_neg + theorem + '</theorem>' + proof_aug + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [ori_mapping, mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_pos = output_str_pos + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[0]) + '</p>'+ '</p>' + '<p>' + proof_aug + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[1]) + '</p>' + '\n==================================================\n'
                        output_str_pos = output_str_pos.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_pos += '\n</root>'
                output_str_neg += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_pos:
                    output_str_pos = '<root>\n' + output_str_pos
                f_out.write(output_str_pos)
                output_str_pos = ''
            f_out.close()
            with open(neg_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_neg:
                    output_str_neg = '<root>\n' + output_str_neg
                f_out.write(output_str_neg)
                output_str_neg = ''
            file_num += 1
            f.close()


def run_self_intersec(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=True):
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_1 = ''
    output_str_2 = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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

                    aug_file = random.choice(files)
                    while aug_file == file:
                        aug_file = random.choice(files)
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    theorem_aug, proof_aug = pair_aug.split('</theorem>')
                    if '<root' in theorem_aug:
                        theorem_aug = '<pair>\n' + theorem_aug.split('<pair>')[1]
                        assert '</root' not in theorem_aug
                    for before in groups.keys():
                        theorem_aug = theorem_aug.replace(before, groups[before])
                        proof_aug = proof_aug.replace(before, groups[before])

                    theorem_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem_aug)])
                    proof_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', proof_aug)])

                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])

                    theorem_candidates={'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            theorem_candidates['greek'].append(var_aug)
                        else:
                            theorem_candidates['english'].append(var_aug.lower())
                    proof_candidates = {'greek': [], 'english': []}
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            proof_candidates['greek'].append(var_aug)
                        else:
                            proof_candidates['english'].append(var_aug.lower())

                    # theorem_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_variables_set]
                    # proof_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_variables_set]
                    # theorem_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_aug_variables_set]
                    # proof_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_aug_variables_set]

                    if type == 'full':
                        theorem_mapping = full_anonymize(theorem_variables_set, [])
                        # if len(theorem_mapping) == 0:
                        #     print(theorem_candidates)
                        #     print(theorem)
                        theorem_aug_mapping = full_anonymize(theorem_aug_variables_set, [], candidates=theorem_candidates)
                        proof_mapping = full_anonymize(proof_variables_set, [])
                        proof_aug_mapping = full_anonymize(proof_aug_variables_set, [], candidates=proof_candidates)
                    elif type == 'partial':
                        theorem_mapping = partial_anonymize(theorem_variables_set, [], partial_range)
                        theorem_aug_mapping = partial_anonymize(theorem_aug_variables_set, [], partial_range, candidates=theorem_candidates)
                        proof_mapping = partial_anonymize(proof_variables_set, [], partial_range)
                        proof_aug_mapping = partial_anonymize(proof_aug_variables_set, [], partial_range, candidates=proof_candidates)
                    elif type == 'zero':
                        theorem_mapping = None

                    proof = ori_proof
                    theorem_partial = theorem
                    if theorem_mapping is not None:
                        for before, after in theorem_mapping:
                            theorem_partial = theorem_partial.replace(before, after)
                        theorem_partial = theorem_partial.replace('</labelmi>', '</m:mi>')
                    theorem_aug_partial = theorem_aug
                    if theorem_aug_mapping is not None:
                        for before, after in theorem_aug_mapping:
                            theorem_aug_partial = theorem_aug_partial.replace(before, after)
                        theorem_aug_partial = theorem_aug_partial.replace('</labelmi>', '</m:mi>')
                    proof_partial = proof
                    if proof_mapping is not None:
                        for before, after in proof_mapping:
                            proof_partial = proof_partial.replace(before, after)
                        proof_partial = proof_partial.replace('</labelmi>', '</m:mi>')
                    proof_aug_partial = proof_aug
                    if proof_aug_mapping is not None:
                        for before, after in proof_aug_mapping:
                            proof_aug_partial = proof_aug_partial.replace(before, after)
                        proof_aug_partial = proof_aug_partial.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        output_str_1 = output_str_1 + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        output_str_2 = output_str_2 + theorem_aug_partial + '</theorem>' + proof_aug_partial + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [theorem_mapping, proof_mapping, theorem_aug_mapping, proof_aug_mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_1 = output_str_1 + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + theorem_partial + '</p>' + '<p>' + str(new_mappings[0]) + '</p>' + '<p>' + theorem_aug_partial + '</p>' + '<p>' + str(new_mappings[2]) + '</p>' + '</pair>\n' + '\n==================================================\n'
                        output_str_1 = output_str_1.replace('<m:', '<').replace('</m:', '</')
                        output_str_2 = output_str_2 + '<p>' + ori_proof + '</theorem>' + '</p>' + '<p>' + proof_partial + '</p>' + '<p>' + str(new_mappings[1]) + '</p>' + '<p>' + proof_aug_partial + '</p>' + '<p>' + str(new_mappings[3]) + '</p>' + '</pair>\n' +  '\n==================================================\n'
                        output_str_2 = output_str_2.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_1 += '\n</root>'
                output_str_2 += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_1:
                    output_str_1 = '<root>\n' + output_str_1
                f_out.write(output_str_1)
                output_str_1 = ''
            f_out.close()
            with open(neg_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_2:
                    output_str_2 = '<root>\n' + output_str_2
                f_out.write(output_str_2)
                output_str_2 = ''
            file_num += 1
            f.close()


def run_pair_domain(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=False):
    def get_domain_dict():
        with open('D:/OneDrive - University of Edinburgh/mathbert/tasks/theory-proof-matching/statement_proof_matching-master/data/corpus_train', 'r', encoding='utf-8') as meta_corpus:
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

    domain_dict = get_domain_dict()

    assert type in ['zero', 'full', 'partial', 'adversarial']
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_pos = ''
    output_str_neg = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
                meta = text.split('<meta arxivid="')[1].split('"')[0].replace('arXiv:', '')
                detail_cat = text.split('category="')[1].split('"')[0].split(' ')
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
                    aug_paper = random.choice(domain_dict[random.choice(detail_cat)])
                    aug_file = None
                    while aug_paper == meta and (aug_file is None or (not os.path.exists(directory+aug_file))):
                        aug_paper = random.choice(domain_dict[random.choice(detail_cat)])
                    try:
                        aug_paper_cat, aug_paper_id = aug_paper.split('/')
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_id[:4] + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    except ValueError:
                        aug_paper_cat, aug_paper_id = aug_paper.split('.')[0], aug_paper.split('.')[1]
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_cat + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    if not os.path.exists(directory+aug_file):
                        continue
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    proof_aug = pair_aug.split('</theorem>')[1]
                    for before in groups.keys():
                        proof_aug = proof_aug.replace(before, groups[before])
                    proof_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', proof_aug)])

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
                        ori_mapping = full_anonymize(common_variables_set, proof_only_variables_set)
                    elif type == 'adversarial':
                        ori_mapping = adversarial_anonymize(common_variables_set)
                    elif type == 'partial':
                        ori_mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range)
                    elif type == 'zero':
                        ori_mapping = None

                    proof_aug_variables = [
                        (var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (
                            var.split('>')[1][0], 'none') for var in proof_aug_variables_set]


                    proof = ori_proof

                    if ori_mapping is not None:
                        for before, after in ori_mapping:
                            proof = proof.replace(before, after)
                        proof = proof.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        output_str_pos = output_str_pos + theorem + '</theorem>' + proof + '</pair>'
                        output_str_neg = output_str_neg + theorem + '</theorem>' + proof_aug + '</pair>'
                    else:
                        output_str_pos = output_str_pos + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>' + '</pair>\n' + '<p>' + str(
                            ori_mapping) + '</p>' + '\n==================================================\n'
                        output_str_neg = output_str_neg + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof_aug + '</p>' + '</pair>\n' + '<p>' + '</p>' + '\n==================================================\n'
                        output_str_pos = output_str_pos.replace('<m:', '<').replace('</m:', '</')
                        output_str_neg = output_str_neg.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_pos += '\n</root>'
                output_str_neg += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_pos:
                    output_str_pos = '<root>\n' + output_str_pos
                f_out.write(output_str_pos)
                output_str_pos = ''
            f_out.close()
            with open(neg_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_neg:
                    output_str_neg = '<root>\n' + output_str_neg
                f_out.write(output_str_neg)
                output_str_neg = ''
            file_num += 1
            f.close()


def run_self_domain(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=True):

    domain_dict = get_domain_dict()
    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_1 = ''
    output_str_2 = ''
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
                meta = text.split('<meta arxivid="')[1].split('"')[0].replace('arXiv:', '')
                detail_cat = text.split('category="')[1].split('"')[0].split(' ')
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
                    aug_paper = random.choice(domain_dict[random.choice(detail_cat)])
                    aug_file = None
                    while aug_paper == meta and (aug_file is None or (not os.path.exists(directory + aug_file))):
                        aug_paper = random.choice(domain_dict[random.choice(detail_cat)])
                    try:
                        aug_paper_cat, aug_paper_id = aug_paper.split('/')
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_id[
                                                            :4] + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    except ValueError:
                        aug_paper_cat, aug_paper_id = aug_paper.split('.')[0], aug_paper.split('.')[1]
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_cat + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    if not os.path.exists(directory + aug_file):
                        continue
                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pair_aug = random.choice(f_aug.read().split('</pair>'))
                    f_aug.close()
                    while len(pair_aug.split()) < 2:
                        aug_file = random.choice(files)
                        while aug_file == file:
                            aug_file = random.choice(files)
                        f_aug = open(directory + aug_file, encoding='utf-8')
                        pair_aug = random.choice(f_aug.read().split('</pair>'))
                        f_aug.close()
                    theorem_aug = pair_aug.split('</theorem>')[0].split('<pair>')[1]
                    proof_aug = pair_aug.split('</theorem>')[1]

                    if '<root' in theorem_aug:
                        theorem_aug = '<pair>\n' + theorem_aug.split('<pair>')[1]
                    for before in groups.keys():
                        theorem_aug = theorem_aug.replace(before, groups[before])
                        proof_aug = proof_aug.replace(before, groups[before])

                    theorem_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem_aug)])
                    proof_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', proof_aug)])

                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])

                    theorem_candidates={'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            theorem_candidates['greek'].append(var_aug)
                        else:
                            theorem_candidates['english'].append(var_aug.lower())
                    proof_candidates = {'greek': [], 'english': []}
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            proof_candidates['greek'].append(var_aug)
                        else:
                            proof_candidates['english'].append(var_aug.lower())

                    # theorem_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_variables_set]
                    # proof_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_variables_set]
                    # theorem_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_aug_variables_set]
                    # proof_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_aug_variables_set]

                    if type == 'full':
                        theorem_mapping = full_anonymize(theorem_variables_set, [])
                        # if len(theorem_mapping) == 0:
                        #     print(theorem_candidates)
                        #     print(theorem)
                        theorem_aug_mapping = full_anonymize(theorem_aug_variables_set, [], candidates=theorem_candidates)
                        proof_mapping = full_anonymize(proof_variables_set, [])
                        proof_aug_mapping = full_anonymize(proof_aug_variables_set, [], candidates=proof_candidates)
                    elif type == 'partial':
                        theorem_mapping = partial_anonymize(theorem_variables_set, [], partial_range)
                        theorem_aug_mapping = partial_anonymize(theorem_aug_variables_set, [], partial_range, candidates=theorem_candidates)
                        proof_mapping = partial_anonymize(proof_variables_set, [], partial_range)
                        proof_aug_mapping = partial_anonymize(proof_aug_variables_set, [], partial_range, candidates=proof_candidates)
                    elif type == 'zero':
                        theorem_mapping = None

                    proof = ori_proof
                    theorem_partial = theorem
                    if theorem_mapping is not None:
                        for before, after in theorem_mapping:
                            theorem_partial = theorem_partial.replace(before, after)
                        theorem_partial = theorem_partial.replace('</labelmi>', '</m:mi>')
                    theorem_aug_partial = theorem_aug
                    if theorem_aug_mapping is not None:
                        for before, after in theorem_aug_mapping:
                            theorem_aug_partial = theorem_aug_partial.replace(before, after)
                        theorem_aug_partial = theorem_aug_partial.replace('</labelmi>', '</m:mi>')
                    proof_partial = proof
                    if proof_mapping is not None:
                        for before, after in proof_mapping:
                            proof_partial = proof_partial.replace(before, after)
                        proof_partial = proof_partial.replace('</labelmi>', '</m:mi>')
                    proof_aug_partial = proof_aug
                    if proof_aug_mapping is not None:
                        for before, after in proof_aug_mapping:
                            proof_aug_partial = proof_aug_partial.replace(before, after)
                        proof_aug_partial = proof_aug_partial.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        if '<pair>' not in theorem_partial:
                            output_str_1 = output_str_1 + '<pair>' + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        else:
                            output_str_1 = output_str_1 + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        if '<pair>' not in theorem_aug_partial:
                            output_str_2 = output_str_2 + '<pair>' + theorem_aug_partial + '</theorem>' + proof_aug_partial + '</pair>'
                        else:
                            output_str_2 = output_str_2 + theorem_aug_partial + '</theorem>' + proof_aug_partial + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [theorem_mapping, proof_mapping, theorem_aug_mapping, proof_aug_mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_1 = output_str_1 + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + theorem_partial + '</p>' + '<p>' + str(new_mappings[0]) + '</p>' + '<p>' + theorem_aug_partial + '</p>' + '<p>' + str(new_mappings[2]) + '</p>' + '</pair>\n' + '\n==================================================\n'
                        output_str_1 = output_str_1.replace('<m:', '<').replace('</m:', '</')
                        output_str_2 = output_str_2 + '<p>' + ori_proof + '</theorem>' + '</p>' + '<p>' + proof_partial + '</p>' + '<p>' + str(new_mappings[1]) + '</p>' + '<p>' + proof_aug_partial + '</p>' + '<p>' + str(new_mappings[3]) + '</p>' + '</pair>\n' +  '\n==================================================\n'
                        output_str_2 = output_str_2.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_1 += '\n</root>'
                output_str_2 += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_1:
                    output_str_1 = '<root>\n' + output_str_1
                f_out.write(output_str_1)
                output_str_1 = ''
            f_out.close()
            with open(neg_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_2:
                    output_str_2 = '<root>\n' + output_str_2
                f_out.write(output_str_2)
                output_str_2 = ''
            file_num += 1
            f.close()


def run_pair_rank(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=False):
    rank_file = "D:/OneDrive - University of Edinburgh/mathbert/tasks/SimCSE/datasets/zero_train_ranks_lap"
    rankings = {}
    for line in open(rank_file, 'r', encoding='utf-8').readlines():
        rankings.update({line.split('\t')[0]: line.split('\t')[1].strip()})
    train_idx_to_meta = {}
    train_meta_to_idx = {}
    with open("D:/OneDrive - University of Edinburgh/mathbert/tasks/theory-proof-matching/anonymized_dataset/zero_anno_train", encoding='utf-8') as f:
        document = f.read().split("\n\n")
        idx = 0
        for chunk in tqdm(document[2:]):
            try:
                docid, _, _ = chunk.split("____________")
                docid = docid.replace("arXiv:", "")
                train_idx_to_meta.update({str(idx): (docid.split('|')[0].strip(), docid.split('|')[1].strip())})
                train_meta_to_idx.update({docid.strip(): idx})
                idx += 1
            except:
                print(chunk)
                continue
    f.close()

    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_pos = ''
    output_str_neg = ''
    skipped = 0
    num_valid_pair = 0
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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
                    pair_id = pair.split(' id="')[1].split('"')[0]

                    file_meta = '/'.join(file.split('.')[-2].split('_')[2:]) + '|' + pair_id
                    if file_meta not in train_meta_to_idx.keys():
                        skipped += 1
                        # print("skipped {}: ".format(str(skipped)) + file_meta)
                        continue
                    file_id = train_meta_to_idx[file_meta]
                    if file_id == int(rankings[str(file_id)]):
                        # print("yes")
                        continue
                    aug_file, aug_pair_id = train_idx_to_meta[rankings[str(file_id)]]


                    if '/' in aug_file:
                        aug_paper_cat, aug_paper_id = aug_file.split('/')
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_id[:4] + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    elif '.' in aug_file:
                        try:
                            aug_paper_cat, aug_paper_id = aug_file.split('.')
                        except:
                            print("nonono")
                            continue
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_cat + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    else:
                        print(aug_file)
                        raise Exception("error")

                    if '0305043' in aug_file or '9904055' in aug_file or '0712_3877' in aug_file or '0508382' in aug_file or '1201_5657' in aug_file:
                        continue

                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pairs = f_aug.read().split('</pair>')
                    for pair_aug in pairs:
                        if aug_pair_id in pair_aug:
                            break
                    f_aug.close()
                    try:
                        proof_aug = pair_aug.split('</theorem>')[1]
                    except:
                        continue
                    for before in groups.keys():
                        proof_aug = proof_aug.replace(before, groups[before])


                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])
                    # print(theorem_variables_set)
                    # print(proof_variables_set)
                    theorem_vars_temp = [var.split('>')[1][0] for var in theorem_variables_set]
                    proof_vars_temp = [var.split('>')[1][0] for var in proof_variables_set]
                    common_vars_temp = list(set(theorem_vars_temp) & set(proof_vars_temp))
                    common_variables_set = proof_variables_set.copy()
                    for item in proof_variables_set:
                        if item.split('>')[1][0] not in common_vars_temp:
                            common_variables_set.remove(item)
                    proof_only_variables_set = proof_variables_set - common_variables_set


                    if type == 'full':
                        ori_mapping = full_anonymize(common_variables_set, proof_only_variables_set)
                    elif type == 'adversarial':
                        ori_mapping = adversarial_anonymize(common_variables_set)
                    elif type == 'partial':
                        ori_mapping = partial_anonymize(common_variables_set, proof_only_variables_set, partial_range)
                    elif type == 'zero':
                        ori_mapping = None


                    candidates = {'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            candidates['greek'].append(var_aug)
                        else:
                            candidates['english'].append(var_aug.lower())

                    proof = ori_proof

                    if ori_mapping is not None:
                        for before, after in ori_mapping:
                            proof = proof.replace(before, after)
                        proof = proof.replace('</labelmi>', '</m:mi>')


                    if not convert_to_readable:
                        output_str_pos = output_str_pos + theorem + '</theorem>' + proof + '</pair>'
                        output_str_neg = output_str_neg + theorem + '</theorem>' + proof_aug + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [ori_mapping, mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_pos = output_str_pos + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + ori_proof + '</p>' + '<p>' + proof + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[0]) + '</p>'+ '</p>' + '<p>' + proof_aug + '</p>'  + '</pair>\n' + '<p>' + str(new_mappings[1]) + '</p>' + '\n==================================================\n'
                        output_str_pos = output_str_pos.replace('<m:', '<').replace('</m:', '</')
                    num_valid_pair += 1
                    f_aug.close()
                if len(output_str_pos) < 20:
                    continue
                output_str_pos += '\n</root>'
                output_str_neg += '\n</root>'
            # if (idx+1) % 100 == 0:
            with open(pos_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_pos:
                    output_str_pos = '<root>\n' + output_str_pos
                f_out.write(output_str_pos)
                output_str_pos = ''
            f_out.close()
            with open(neg_output_dir + str(file_num + 1) + '.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_neg:
                    output_str_neg = '<root>\n' + output_str_neg
                f_out.write(output_str_neg)
                output_str_neg = ''
            file_num += 1
            f.close()
    print('Total number of valid pairs: ', num_valid_pair)


def run_self_rank(directory, pos_output_dir, neg_output_dir, type, partial_range=0.5, start=None, end=None, convert_to_readable=False, convert_theorem=True):
    rank_file = "D:/OneDrive - University of Edinburgh/mathbert/tasks/SimCSE/datasets/zero_train_ranks_lap"
    rankings = {}
    for line in open(rank_file, 'r', encoding='utf-8').readlines():
        rankings.update({line.split('\t')[0]: line.split('\t')[1].strip()})
    train_idx_to_meta = {}
    train_meta_to_idx = {}
    with open(
            "D:/OneDrive - University of Edinburgh/mathbert/tasks/theory-proof-matching/anonymized_dataset/zero_anno_train",
            encoding='utf-8') as f:
        document = f.read().split("\n\n")
        idx = 0
        for chunk in tqdm(document[2:]):
            try:
                docid, _, _ = chunk.split("____________")
                docid = docid.replace("arXiv:", "")
                train_idx_to_meta.update({str(idx): (docid.split('|')[0].strip(), docid.split('|')[1].strip())})
                train_meta_to_idx.update({docid.strip(): idx})
                idx += 1
            except:
                print(chunk)
                continue
    f.close()

    if not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
    if not os.path.exists(neg_output_dir):
        os.makedirs(neg_output_dir)
    if not directory.endswith('/'):
        directory += '/'
    if not pos_output_dir.endswith('/'):
        pos_output_dir += '/'
    if not neg_output_dir.endswith('/'):
        neg_output_dir += '/'
    if start is None:
        files = os.listdir(directory)
    else:
        if end is None:
            files = os.listdir(directory)[start:end]
        elif end <= len(os.listdir(directory)):
            files = os.listdir(directory)[start:end]
        else:
            files = os.listdir(directory)[start:]
    file_num = start if start is not None else 0
    output_str_1 = ''
    output_str_2 = ''
    skipped = 0
    num_valid_pair = 0
    for idx, file in tqdm(enumerate(files), total=len(files)):
        if os.path.isfile(directory+file):
            with open(directory+file, encoding='utf-8') as f:
                text = f.read()
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
                    pair_id = pair.split(' id="')[1].split('"')[0]

                    file_meta = '/'.join(file.split('.')[-2].split('_')[2:]) + '|' + pair_id
                    if file_meta not in train_meta_to_idx.keys():
                        skipped += 1
                        # print("skipped {}: ".format(str(skipped)) + file_meta)
                        continue
                    file_id = train_meta_to_idx[file_meta]
                    if file_id == int(rankings[str(file_id)]):
                        # print("yes")
                        continue
                    aug_file, aug_pair_id = train_idx_to_meta[rankings[str(file_id)]]

                    if '/' in aug_file:
                        aug_paper_cat, aug_paper_id = aug_file.split('/')
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_id[
                                                            :4] + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    elif '.' in aug_file:
                        try:
                            aug_paper_cat, aug_paper_id = aug_file.split('.')
                        except:
                            print("nonono")
                            continue
                        aug_file = 'data_MREC2011.4.439_' + aug_paper_cat + '_' + aug_paper_cat + '_' + aug_paper_id + '.xhtml'
                    else:
                        print(aug_file)
                        raise Exception("error")

                    if '0305043' in aug_file or '9904055' in aug_file or '0712_3877' in aug_file or '0508382' in aug_file or '1201_5657' in aug_file:
                        continue

                    f_aug = open(directory + aug_file, encoding='utf-8')
                    pairs = f_aug.read().split('</pair>')
                    for pair_aug in pairs:
                        if aug_pair_id in pair_aug:
                            break
                    f_aug.close()
                    try:
                        theorem_aug = pair_aug.split('</theorem>')[0].split('<pair>')[1]
                        proof_aug = pair_aug.split('</theorem>')[1]
                    except:
                        continue

                    if '<root' in theorem_aug:
                        theorem_aug = '<pair>\n' + theorem_aug.split('<pair>')[1]
                    for before in groups.keys():
                        theorem_aug = theorem_aug.replace(before, groups[before])
                        proof_aug = proof_aug.replace(before, groups[before])

                    theorem_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem_aug)])
                    proof_aug_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>',
                        proof_aug) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', proof_aug)])

                    theorem_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', theorem)])
                    proof_variables_set = set([var for var in re.findall(
                        r'<m:mi mathvariant="[a-z]+"[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof) + re.findall(
                        r'<m:mi[a-z"=0-9 ]*>[A-Za-zαβγδεζηθικλμνξορστυφχψω]</m:mi>', ori_proof)])

                    theorem_candidates={'greek': [], 'english': []}
                    for var_aug in theorem_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            theorem_candidates['greek'].append(var_aug)
                        else:
                            theorem_candidates['english'].append(var_aug.lower())
                    proof_candidates = {'greek': [], 'english': []}
                    for var_aug in proof_variables_set:
                        var_aug = var_aug.split('>')[1][0]
                        if var_aug in greek_letters:
                            proof_candidates['greek'].append(var_aug)
                        else:
                            proof_candidates['english'].append(var_aug.lower())

                    # theorem_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_variables_set]
                    # proof_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_variables_set]
                    # theorem_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in theorem_aug_variables_set]
                    # proof_aug_variables = [(var.split('>')[1][0], var.split('"')[1]) if var.find('mathvariant') > 0 else (var.split('>')[1][0], 'none') for var in proof_aug_variables_set]

                    if type == 'full':
                        theorem_mapping = full_anonymize(theorem_variables_set, [])
                        # if len(theorem_mapping) == 0:
                        #     print(theorem_candidates)
                        #     print(theorem)
                        theorem_aug_mapping = full_anonymize(theorem_aug_variables_set, [], candidates=theorem_candidates)
                        proof_mapping = full_anonymize(proof_variables_set, [])
                        proof_aug_mapping = full_anonymize(proof_aug_variables_set, [], candidates=proof_candidates)
                    elif type == 'partial':
                        theorem_mapping = partial_anonymize(theorem_variables_set, [], partial_range)
                        theorem_aug_mapping = partial_anonymize(theorem_aug_variables_set, [], partial_range, candidates=theorem_candidates)
                        proof_mapping = partial_anonymize(proof_variables_set, [], partial_range)
                        proof_aug_mapping = partial_anonymize(proof_aug_variables_set, [], partial_range, candidates=proof_candidates)
                    elif type == 'zero':
                        theorem_mapping = None

                    proof = ori_proof
                    theorem_partial = theorem
                    if theorem_mapping is not None:
                        for before, after in theorem_mapping:
                            theorem_partial = theorem_partial.replace(before, after)
                        theorem_partial = theorem_partial.replace('</labelmi>', '</m:mi>')
                    theorem_aug_partial = theorem_aug
                    if theorem_aug_mapping is not None:
                        for before, after in theorem_aug_mapping:
                            theorem_aug_partial = theorem_aug_partial.replace(before, after)
                        theorem_aug_partial = theorem_aug_partial.replace('</labelmi>', '</m:mi>')
                    proof_partial = proof
                    if proof_mapping is not None:
                        for before, after in proof_mapping:
                            proof_partial = proof_partial.replace(before, after)
                        proof_partial = proof_partial.replace('</labelmi>', '</m:mi>')
                    proof_aug_partial = proof_aug
                    if proof_aug_mapping is not None:
                        for before, after in proof_aug_mapping:
                            proof_aug_partial = proof_aug_partial.replace(before, after)
                        proof_aug_partial = proof_aug_partial.replace('</labelmi>', '</m:mi>')

                    if not convert_to_readable:
                        if '<pair>' not in theorem_partial:
                            output_str_1 = output_str_1 + '<pair>' + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        else:
                            output_str_1 = output_str_1 + theorem_partial + '</theorem>' + proof_partial + '</pair>'
                        if '<pair>' not in theorem_aug_partial:
                            output_str_2 = output_str_2 + '<pair>' + theorem_aug_partial + '</theorem>' + proof_aug_partial + '</pair>'
                        else:
                            output_str_2 = output_str_2 + theorem_aug_partial + '</theorem>' + proof_aug_partial + '</pair>'
                    else:
                        new_mappings = []
                        for old_mapping in [theorem_mapping, proof_mapping, theorem_aug_mapping, proof_aug_mapping]:
                            new_mapping = set()
                            for mapping in old_mapping:
                                before, after = mapping
                                before_var = before.split('>')[1][0]
                                after_var = after.split('>')[1][0]
                                new_mapping.add((before_var, after_var))
                            new_mappings.append(new_mapping)
                        output_str_1 = output_str_1 + '<p>' + theorem + '</theorem>' + '</p>' + '<p>' + theorem_partial + '</p>' + '<p>' + str(new_mappings[0]) + '</p>' + '<p>' + theorem_aug_partial + '</p>' + '<p>' + str(new_mappings[2]) + '</p>' + '</pair>\n' + '\n==================================================\n'
                        output_str_1 = output_str_1.replace('<m:', '<').replace('</m:', '</')
                        output_str_2 = output_str_2 + '<p>' + ori_proof + '</theorem>' + '</p>' + '<p>' + proof_partial + '</p>' + '<p>' + str(new_mappings[1]) + '</p>' + '<p>' + proof_aug_partial + '</p>' + '<p>' + str(new_mappings[3]) + '</p>' + '</pair>\n' +  '\n==================================================\n'
                        output_str_2 = output_str_2.replace('<m:', '<').replace('</m:', '</')

                    f_aug.close()

                output_str_1 += '\n</root>'
                output_str_2 += '\n</root>'
            # if (idx+1) % 100 == 0:
            if len(output_str_1) < 20:
                output_str_1 = ''
                output_str_2 = ''
                continue
            with open(pos_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_1:
                    output_str_1 = '<root>\n' + output_str_1
                if '</root>' in output_str_1[:100]:
                    output_str_1 = output_str_1.replace('</root>', '') + '\n</root>'
                f_out.write(output_str_1)
                output_str_1 = ''
            f_out.close()
            with open(neg_output_dir+str(file_num+1)+'.html', 'w', encoding='utf-8') as f_out:
                if '<root' not in output_str_2:
                    output_str_2 = '<root>\n' + output_str_2
                if '</root>' in output_str_2[:100]:
                    output_str_2 = output_str_2.replace('</root>', '') + '\n</root>'
                f_out.write(output_str_2)
                output_str_2 = ''
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
    # run_pair_naive_proportion(args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type, convert_to_readable=False, pairs=select_pairs_list)
    run_anonymize(args.input_dir, args.output_dir+"/", type=args.type, start=args.start, end=args.end, convert_to_readable=False, convert_theorem=convert_theorem)
    # run_pair_naive(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #                    start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_self_naive(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #                start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_self_intersec(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type, start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_pair_intersec(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #                   start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_pair_domain(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type, start=int(args.start), end=int(args.end), convert_to_readable=False, convert_theorem=convert_theorem)
    # run_self_domain(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #                   start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_pair_rank(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #                 start=int(args.start), end=int(args.end), convert_to_readable=False)
    # run_self_rank(args.input_dir, args.output_dir + "_positive/", args.output_dir + "_negative/", type=args.type,
    #               start=int(args.start), end=int(args.end), convert_to_readable=False)


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