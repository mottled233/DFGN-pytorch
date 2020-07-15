from model.layers import *
from pytorch_pretrained_bert.modeling import BertModel
from model.GFN import *
from model.layers import *
from transformers import BertModel
from transformers import BertConfig as BC


class DFGN(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config, pretrained_bert):
        super(DFGN, self).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(pretrained_bert, cache_dir="data/chn_bert_base")
        if config.without_bert_optimize:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.model = GraphFusionNet(config=config)
        # self.prediction = BaselinePredictionLayer(config=config)

    def forward(self, batch, return_yp=True, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        N = doc_ids.shape[0]
        doc_encoding = self.encoder(input_ids=doc_ids,
                                    # token_type_ids=segment_ids,
                                    attention_mask=doc_mask)[0]
        if self.config.without_bert_optimize:
            doc_encoding = doc_encoding.detach()
        batch['context_encoding'] = doc_encoding
        start, end, sp, Type, softmask, ent, yp1, yp2 = self.model(batch, return_yp)
        return start, end, sp, Type, softmask, yp1, yp2

        # start, end, sp, Type, yp1, yp2 = self.prediction(batch)
        # return start, end, sp, Type, None, yp1, yp2


