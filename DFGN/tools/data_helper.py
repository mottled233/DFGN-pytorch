from os.path import join
import gzip
import pickle
import json
from tqdm import tqdm
from tools.data_iterator_pack import DataIteratorPack
import logging


class DataHelper:
    def __init__(self, gz=True, config=None):
        self.DataIterator = DataIteratorPack
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = 'data'
        self.subset_file = join(self.data_dir, 'subset.json')

        self.__train_features__ = None
        self.__dev_features__ = None
        self.__qat_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None
        self.__qat_examples__ = None

        self.__train_graphs__ = None
        self.__dev_graphs__ = None
        self.__qat_graphs__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None
        self.__qat_example_dict__ = None

        self.config = config

    @property
    def sent_limit(self):
        return 25

    @property
    def entity_limit(self):
        return 80

    @property
    def n_type(self):
        return 2

    def get_feature_file(self, tag):
        return join(self.data_dir, tag + '_feature' + self.suffix)

    def get_example_file(self, tag):
        return join(self.data_dir, tag + '_example' + self.suffix)

    def get_graph_file(self, tag):
        return join(self.data_dir, tag + '_graph' + self.suffix)

    @property
    def train_feature_file(self):
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        return self.get_feature_file('dev')

    @property
    def qat_feature_file(self):
        return self.get_feature_file('qat')

    @property
    def train_example_file(self):
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        return self.get_example_file('dev')

    @property
    def qat_example_file(self):
        return self.get_example_file('qat')

    @property
    def train_graph_file(self):
        return self.get_graph_file('train')

    @property
    def dev_graph_file(self):
        return self.get_graph_file('dev')

    @property
    def qat_graph_file(self):
        return self.get_graph_file('qat')

    @staticmethod
    def compress_pickle(pickle_file_name):
        def abbr(obj):
            obj_str = str(obj)
            if len(obj_str) > 100:
                return obj_str[:20] + ' ... ' + obj_str[-20:]
            else:
                return obj_str

        def get_obj_dict(pickle_obj):
            if isinstance(pickle_obj, list):
                obj = pickle_obj[0]
            elif isinstance(pickle_obj, dict):
                obj = list(pickle_obj.values())[0]
            else:
                obj = pickle_obj
            if isinstance(obj, dict):
                return obj
            else:
                return obj.__dict__

        pickle_obj = pickle.load(open(pickle_file_name, 'rb'))

        for k, v in get_obj_dict(pickle_obj).items():
            logging.info("{} {}".format(k, abbr(v)))
        with gzip.open(pickle_file_name + '.gz', 'wb') as fout:
            pickle.dump(pickle_obj, fout)
        pickle_obj = pickle.load(gzip.open(pickle_file_name + '.gz', 'rb'))
        for k, v in get_obj_dict(pickle_obj).items():
            logging.info("{} {}".format(k, abbr(v)))

    def __load__(self, file):
        if file.endswith('json'):
            return json.load(open(file, 'r'))
        with self.get_pickle_file(file) as fin:
            logging.info('loading {}'.format(file))
            return pickle.load(fin)

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                logging.info('loading {}'.format(file))
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Features
    @property
    def train_features(self):
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    @property
    def qat_features(self):
        return self.__get_or_load__('__qat_features__', self.qat_feature_file)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    @property
    def qat_examples(self):
        return self.__get_or_load__('__qat_examples__', self.qat_example_file)

    # Graphs
    @property
    def train_graphs(self):
        return self.__get_or_load__('__train_graphs__', self.train_graph_file)

    @property
    def dev_graphs(self):
        return self.__get_or_load__('__dev_graphs__', self.dev_graph_file)

    @property
    def qat_graphs(self):
        return self.__get_or_load__('__qat_graphs__', self.qat_graph_file)

    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    @property
    def qat_example_dict(self):
        if self.__qat_example_dict__ is None:
            self.__qat_example_dict__ = {e.qas_id: e for e in self.qat_examples}
        return self.__qat_example_dict__

    # Feature dict
    @property
    def train_feature_dict(self):
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    @property
    def qat_feature_dict(self):
        return {e.qas_id: e for e in self.qat_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict, self.dev_graphs

    def load_train(self):
        return self.train_features, self.train_example_dict, self.train_graphs

    def load_train_subset(self):
        return self.qat_features, self.qat_example_dict, self.qat_graphs

    @property
    def dev_loader(self):
        return self.DataIterator(*self.load_dev(),
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=True,
                                 n_layers=self.config.n_layers)

    @property
    def train_loader(self):
        return self.DataIterator(*self.load_train(),
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=False,
                                 n_layers=self.config.n_layers)

    @property
    def train_sub_loader(self):
        return self.DataIterator(*self.load_train_subset(),
                                 bsz=24,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=False,
                                 n_layers=self.config.n_layers)
