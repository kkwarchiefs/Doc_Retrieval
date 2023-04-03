class PredictionDatasetURLTitle(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.tok = tokenizer
        self.max_len = max_len
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        title = self.pid2title.get(pid)
        qry = self.qid2txt[qid]
        if title == '-':
            title = 'null'
        psg = qry + ', title:' + title + ', text: ' + self.pid2txt[pid]
        return self.tok.encode_plus(
            psg,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
