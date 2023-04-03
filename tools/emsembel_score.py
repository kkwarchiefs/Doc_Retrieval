#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : emsembel_score.py
# @Author: 罗锦文
# @Date  : 2021/9/2
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from collections import defaultdict
import json
import random


def read_txt(query_file):
    qid2txt = {}
    for line in open(query_file, 'r', encoding='utf-8'):
        items = line.strip().split('\t')
        if len(items) < 2:
            print(line, items)
            continue
        qid2txt[items[0]] = items[1]
    return qid2txt


def read_set(query_file):
    qid2txt = defaultdict(set)
    for line in open(query_file, 'r', encoding='utf-8'):
        items = line.strip().split('\t')
        qid2txt[items[0]].add(items[1])
    return qid2txt


def check_round():
    query_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/queries.all.tsv'
    collection_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/collection.tsv'
    addtion_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/qrels.train.addition.tsv'
    qid2txt = read_txt(query_file)
    pid2txt = read_txt(collection_file)
    pid2add = read_set(addtion_file)
    # for line in sys.stdin:
    #     group = json.loads(line)
    #     qid = group['qry']
    #     pos_pid = random.choice(group['pos'])
    #     adds = pid2add[qid]
    #     for round in range(int(pos_pid) - 10, int(pos_pid) + 10):
    #         adds.add(str(round))
    #     neg_group = [nid for nid in group['neg'] if nid not in adds]
    #     if len(neg_group):
    #         print(qid, pos_pid, neg_group)


def formate():
    topk = 50
    queryset = defaultdict(list)
    for line in open('/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/qrels.train.tsv'):
        items = line.strip().split('\t')
        queryset[items[0]].append(items[2])
    last = None
    psg = []
    for line in sys.stdin:
        items = line.strip().split('\t')
        if last is None:
            last = items[0]
            score = float(items[3]) + float(items[4])
            psg = [(items[1], score)]
        elif items[0] != last:
            pos = queryset.get(last)
            psg.sort(key=lambda a: a[1], reverse=True)
            item_set = {
                'qry': last,
                'pos': pos,
                'neg': [idx for idx, _ in psg if idx not in pos][:topk],
            }
            print(json.dumps(item_set))
            last = items[0]
            score = float(items[3]) + float(items[4])
            psg = [(items[1], score)]
        else:
            score = float(items[3]) + float(items[4])
            psg.append((items[1], score))
    if len(psg):
        pos = queryset.get(last)
        psg.sort(key=lambda a: a[1], reverse=True)
        item_set = {
            'qry': last,
            'pos': pos,
            'neg': [idx for idx, _ in psg if idx not in pos][:topk],
        }
        print(json.dumps(item_set))


def convert_classify():
    qid2pids = defaultdict(list)
    queryset = defaultdict(list)
    for line in open('/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/qrels.dev.small.tsv'):
        items = line.strip().split('\t')
        queryset[items[0]].append(items[2])
    for line in sys.stdin:
        items = line.strip().split('\t')
        qid2pids[items[0]].append(items[1])
    for k, v in qid2pids.items():
        pos = queryset[k]
        item_set = {
            'qry': k,
            'pos': pos,
            'neg': v,
        }
        print(json.dumps(item_set))


def sample_top():
    top = {2: 579, 0: 1990, 8: 131, 1: 961, 9: 113, 4: 282, 6: 189, 7: 123, 3: 403, 5: 217, 10: 1992}
    candi_index = []
    for k, v in top.items():
        candi_index += [k] * v
    print(candi_index)
    for line in sys.stdin:
        group = json.loads(line)
        examples = []
        group_batch = []
        qid = group['qry']
        pos_pid = random.choice(group['pos'])
        pos_index = random.choice(candi_index)
        neg_group = group['neg']
        if len(neg_group) < 20:
            negs = random.choices(neg_group, k=10)
        else:
            negs = random.sample(neg_group, k=10)
        if pos_index < 10:
            negs[pos_index] = pos_pid
        for neg_id in negs:
            examples.append((qid, neg_id))
        # collaborative mode, split the group
        print(examples, group['pos'], pos_index)


def convert():
    count = 0
    thread = float(sys.argv[1])
    for line in sys.stdin:
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        # if score[idx] - score[0] > thread:
        # if idx == 1:
        # tmp = neg[0]
        # neg[0] = neg[1]
        # neg[1] = tmp
        # if idx > 0 and score[idx] > thread:
        #     tmp = neg[idx]
        #     neg.remove(tmp)
        #     neg = [tmp] + neg
        #     count += 1
        left, right = [], []
        for nid, se in zip(neg, score):
            if se > thread:
                left.append((nid, se))
            else:
                right.append(nid)
        # left.sort(key=lambda a:a[1], reverse=True)
        neg = [a[0] for a in left] + right
        for i, _neg in enumerate(neg):
            print(qry, _neg, i + 1, sep='\t')
    print(count, file=sys.stderr)


def resort():
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        tups = []
        for a, b in zip(neg, score):
            tups.append((a, b))
        # if score[0] < 0.1:
        tups.sort(key=lambda a: a[1], reverse=True)
        sorted = [_a[0] for _a in tups]
        sorted = sorted + neg[len(sorted):]
        # tmp = neg[idx]
        # neg.remove(tmp)
        # neg = [tmp] + neg
        for i, _neg in enumerate(sorted[:10]):
            print(qry, _neg, i + 1, sep='\t')


def merge_rank():
    res = '../dataset/ms_passage/qrels.dev.tsv.small'
    qry2res = defaultdict(list)
    for line in open(res):
        items = line.strip().split('\t')
        qry2res[items[0]].append(items[1])
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        tups = []
        res = qry2res[qry]
        for a, b in zip(neg, score):
            tups.append((a, b))
        # if score[0] < 0.1:
        tups.sort(key=lambda a: a[1], reverse=True)
        sort_ = [_a[0] for _a in tups]
        sort_ = sort_  # + neg[len(sort_):]
        newdict = {a: idx + 1 for idx, a in enumerate(neg[:len(sort_)])}
        for idx, a in enumerate(sort_):
            newdict[a] += (idx + 1)
        merge = sorted(newdict.items(), key=lambda a: a[1])
        # if merge[0][0] != sort_[0] and merge[0][0] in res and neg[0] not in res:
        #     print(tups)
        #     print(neg[:10])
        #     print(merge)
        for i, _neg in enumerate(merge[:10]):
            print(qry, _neg[0], i + 1, sep='\t')


def resort2():
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        # tups = []
        # for a, b in zip(neg, score):
        #     tups.append((a, b))
        # # if score[0] < 0.1:
        # tups.sort(key=lambda a: a[1], reverse=True)
        # sorted = [_a[0] for _a in tups]
        # sorted = sorted  # + neg[len(sorted):]
        tmp = neg[idx]
        neg.remove(tmp)
        neg = [tmp] + neg
        for i, _neg in enumerate(neg):
            print(qry, _neg, i + 1, sep='\t')


def cal_score():
    res = '../dataset/ms_passage/qrels.dev.tsv.small'
    qry2res = defaultdict(list)
    for line in open(res):
        items = line.strip().split('\t')
        qry2res[items[0]].append(items[1])

    acc = 0
    for line in sys.stdin:
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        top1 = neg[idx]
        if top1 in qry2res[qry]:
            acc += 1
        else:
            idx = -1
            for id in qry2res[qry]:
                if id in neg:
                    idx = neg.index(id)
                    break
            print(qry2res[qry], idx, line.strip(), sep='\t')
    print(acc / len(qry2res))


def doc2doc():
    docs = defaultdict(list)
    for line in sys.stdin:
        items = line.strip().split('\t')
        docs[items[0]].append(items[2])
    for k, v in docs.items():
        print(k, list(v), sep='\t')


def formate_output():
    for line in open(sys.argv[1]):
        items = line.strip().split('\t')
        qry, neg, idx, score = items[0], eval(items[1]), int(items[2]), eval(items[3])
        tups = []
        for a, b in zip(neg, score):
            tups.append((a, b))
        # if score[0] < 0.1:
        tups.sort(key=lambda a: a[1], reverse=True)
        sort_ = [_a[0] for _a in tups]
        sort_ = sort_ + neg[len(sort_):100]
        for i, _neg in enumerate(sort_[:100]):
            print(qry, _neg, i + 1, sep='\t')


def check_sim():
    train = '/cfs/cfs-i125txtf/jamsluo/dataset/coil_psg/trids_condener_top100/rank_top100_pair.json'
    dev = '/cfs/cfs-i125txtf/jamsluo/task_result_output/retrival_inter_condenser_4gpu/score/classify_rank_top100.json'
    sim = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/sim_all_psg'
    simdict = {}
    for line in open(sim):
        items = line.strip().split('\t')
        simdict[items[0]] = eval(items[1])
    for line in open(dev):
        ins = json.loads(line)
        pos = ins['pos']
        neg = ins['neg']
        possim = []
        for _pos in pos:
            possim += simdict.get(_pos, [])
        if len(possim):
            count = [_p for _p in possim if _p in neg]
            print(pos, possim, count, neg, sep='\t')


def sim_query():
    res = '../dataset/ms_passage/qrels.dev.tsv.small'
    qry2res = defaultdict(list)
    for line in open(res):
        items = line.strip().split('\t')
        qry2res[items[1]].append(items[0])
    for k, v in qry2res.items():
        if len(v) > 1:
            print(k, v)


def find_top100():
    candi = '/cfs/cfs-i125txtf/jamsluo/task_result_output/retrival_inter_condenser_4gpu/score/classify_rank_top100.json'
    sim = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/sim_all_psg'
    doc2sim = defaultdict(list)
    for line in open(sim):
        items = line.strip().split('\t')
        doc2sim[items[0]] = eval(items[1])
    right = 0
    pred = 0
    add_pred = 0
    for line in open(candi):
        ins = json.loads(line)
        pos = ins['pos']
        neg = ins['neg'][:10]
        right += len(pos)
        pred += len([p for p in pos if p in neg])
        add_ = []
        for n_ in neg:
            add_.extend(doc2sim.get(n_, []))
        add_ = neg + add_
        add_pred += len([p for p in pos if p in add_])
    print(right, pred, add_pred, 1. * pred / right, 1. * add_pred / right)


def add_score():
    af = sys.argv[1]
    bf = sys.argv[2]
    ascore = defaultdict(dict)
    for line in open(af):
        items = line.strip().split('\t')
        obj = ascore[items[0]]
        obj[items[1]] = float(items[3])
    bscore = defaultdict(dict)
    for line in open(bf):
        items = line.strip().split('\t')
        obj = bscore[items[0]]
        obj[items[1]] = float(items[3])
    for a, rank_a in ascore.items():
        rank_b = bscore[a]
        newrank = []
        for k, v in rank_a.items():
            newrank.append((k, v + rank_b[k]))
        newrank.sort(key=lambda a: a[1], reverse=True)
        for id, (dd, tup) in enumerate(newrank):
            print(a, dd, id, tup, sep='\t')


def compare_diff():
    res = '../dataset/ms_passage/qrels.dev.tsv.small'
    qry2res = defaultdict(list)
    for line in open(res):
        items = line.strip().split('\t')
        qry2res[items[0]].append(items[1])
    af = sys.argv[1]
    bf = sys.argv[2]
    ascore = defaultdict(list)
    for line in open(af):
        items = line.strip().split('\t')
        ascore[items[0]].append(items[1])
    bscore = defaultdict(list)
    for line in open(bf):
        items = line.strip().split('\t')
        bscore[items[0]].append(items[1])
    query_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/queries.all.tsv'
    collection_file = '/cfs/cfs-i125txtf/jamsluo/dataset/ms_passage/loaderboard/collection.tsv'
    qid2txt = read_txt(query_file)
    pid2txt = read_txt(collection_file)
    f1 = open('outputs/v1', 'w')
    f2 = open('outputs/v2', 'w')
    f3 = open('outputs/v3', 'w')
    for k, v in ascore.items():
        if k in qry2res:
            if v[0] in qry2res[k]:
                other = bscore[k]
                if other[0] not in qry2res[k]:
                    for a, b in zip(v[:10], other):
                        print(k, qid2txt[k], a, pid2txt[a], a in qry2res[k], b, pid2txt[b], b in qry2res[k], sep='\t',
                              file=f1)
            else:
                other = bscore[k]
                if other[0] not in qry2res[k]:
                    for a, b in zip(v[:10], other):
                        print(k, qid2txt[k], a, pid2txt[a], a in qry2res[k], b, pid2txt[b], b in qry2res[k], sep='\t',
                              file=f2)
                else:
                    for a, b in zip(v[:10], other):
                        print(k, qid2txt[k], a, pid2txt[a], a in qry2res[k], b, pid2txt[b], b in qry2res[k], sep='\t',
                              file=f3)
    f1.close()
    f2.close()
    f3.close()

def count_pos_neg():
    count, pos, neg = 0, 0, 0
    maxlen = 0
    pos_zero, neg_zero = 0, 0
    poslen, neglen = 0, 0
    for line in sys.stdin:
        ins = json.loads(line)
        count += 1
        pos += len(ins['pos'])
        neg += len(ins['neg'])
        poslen += sum([len(a.split()) for a in ins['pos']])
        neglen += sum([len(a.split()) for a in ins['neg']])
        maxlen = max(maxlen, len(ins['pos']) + len(ins['neg']))
        if len(ins['pos']) == 0:
            pos_zero += 1
        if len(ins['neg']) == 0:
            neg_zero += 1
        # if len(ins['pos']) + len(ins['neg']) > 100:
        #     print(line.strip())
    print('pos:', pos/count, pos_zero, poslen/pos, 'neg:', neg/count, neg_zero, neglen/neg, 'max:', maxlen)

if __name__ == "__main__":
    formate_output()
