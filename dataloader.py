import pickle
import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

class ETDataset(Dataset):
    def __init__(self, args, data_name, e2id, r2id, t2id, c2id, data_flag):
        self.args = args
        self.data_name = data_name
        self.e2id = e2id
        self.r2id = r2id
        self.t2id = t2id
        self.c2id = c2id
        self.sample_et_size = args["sample_et_size"]
        self.sample_kg_size = args["sample_kg_size"]
        self.data = self.load_dataset()
        self.data_flag = data_flag

    def load_dataset(self):
        data_name_path = self.args["data_dir"] + '/' + self.args["dataset"] + '/' + self.data_name
        contents = []

        output_pickle = data_name_path[0: data_name_path.rfind('.')] + '.pkl'
        if os.path.exists(output_pickle):
            with open(output_pickle, 'rb') as handle:
                contents = pickle.load(handle)
                return contents

        with open(data_name_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                mask_ent, et_triples, kg_triples, clu_triples = [_.strip() for _ in line.split('|||')]
                et_content_list = et_triples.split(' [SEP] ')
                et_list = []

                kg_content_list = kg_triples.split(' [SEP] ')
                kg_list = []

                for et_content in et_content_list:
                    et_head, et_rel, et_type = et_content.split(' ')
                    et_head_id = self.e2id[et_head]
                    et_type_id = self.t2id[et_type] + len(self.e2id)
                    et_list.append([et_head_id, self.c2id[et_rel] + len(self.r2id), et_type_id])

                for kg_content in kg_content_list:
                    kg_head, kg_rel, kg_tail = kg_content.split(' ')
                    if kg_rel.startswith('inv-'):
                        kg_rel_id = len(self.r2id) + len(self.c2id) + self.r2id[kg_rel[4:]]
                    else:
                        kg_rel_id = self.r2id[kg_rel]
                    kg_head_id = self.e2id[kg_head]
                    kg_tail_id = self.e2id[kg_tail]
                    kg_list.append([kg_head_id, kg_rel_id, kg_tail_id])

                contents.append((et_list, kg_list, self.e2id[mask_ent]))

        with open(output_pickle, 'wb') as handle:
            pickle.dump(contents, handle)

        return contents

    def __getitem__(self, index):
        et_content = self.data[index][0]
        kg_content = self.data[index][1]
        ent = self.data[index][2]

        single_et_np_list = []
        if self.sample_et_size != 1:
            sampled_index = np.random.choice(range(0, len(et_content)), size=self.sample_et_size,
                                             replace=len(range(0, len(et_content))) < self.sample_et_size)
            for i in sampled_index:
                single_et_np_list.append(et_content[i])
        else:
            single_et_np_list.append(et_content[0])

        single_kg_np_list = []
        if self.sample_kg_size != 1:
            sampled_index = np.random.choice(range(0, len(kg_content)), size=self.sample_kg_size,
                                             replace=len(range(0, len(kg_content))) < self.sample_kg_size)
            for i in sampled_index:
                single_kg_np_list.append(kg_content[i])
        else:
            single_kg_np_list.append(kg_content[0])

        all_et = et_content
        all_kg = kg_content
        sample_et = single_et_np_list
        sample_kg = single_kg_np_list

        gt_ent = ent

        if self.data_flag == 'test':
            # for test, we need all neighbor information
            return all_et, all_kg, gt_ent
        else:
            return sample_et, sample_kg, gt_ent

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sample_et_content_list = []
        sample_et_content_list.append([_[0] for _ in batch])

        sample_kg_content_list = []
        sample_kg_content_list.append([_[1] for _ in batch])

        gt_ent_list = []
        gt_ent_list.append([_[2] for _ in batch])

        et_content = torch.LongTensor(sample_et_content_list[0])
        kg_content = torch.LongTensor(sample_kg_content_list[0])

        gt_ent = torch.LongTensor(gt_ent_list[0])

        return et_content, kg_content, gt_ent
