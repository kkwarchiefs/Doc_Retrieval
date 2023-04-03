class PredictionDatasetGroupGlobal(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(self, args: DataArguments, path_to_json: List[str], tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_json,
        )['train']
        self.args = args
        self.tok = tokenizer
        self.max_len = max_len
        query_file = args.corpus_path + 'queries.all.tsv'
        collection_file = args.corpus_path + 'collection.tsv'
        title_file = args.corpus_path + 'para.title.txt'
        self.qid2txt = self.read_txt(query_file)
        self.pid2txt = self.read_txt(collection_file)
        self.pid2title = self.read_txt(title_file)

    def __len__(self):
        return len(self.nlp_dataset)

    def read_txt(self, query_file):
        qid2txt = {}
        for line in open(query_file, 'r', encoding='utf-8'):
            items = line.strip().split('\t')
            qid2txt[items[0]] = items[1]
        return qid2txt

    def create_one_example(self, doc_encoding: str):
        item = self.tok.encode_plus(
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> [List[BatchEncoding], List[int]]:
        group = self.nlp_dataset[item]
        qid = group['qry']
        qry = self.qid2txt[qid]
        negs = group['neg'][:self.args.train_group_size]
        group_batch = []
        for neg_id in negs:
            title = self.pid2title.get(neg_id)
            if title == '-':
                title = 'null'
            psg = qry + ', title:' + title + ', text: ' + self.pid2txt[neg_id]
            group_batch.append(self.create_one_example(psg))
        return group_batch
