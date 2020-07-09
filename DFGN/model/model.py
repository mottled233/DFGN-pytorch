from model.layers import *
from pytorch_pretrained_bert.modeling import BertModel
from model.GFN import *


class DFGN(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config, pretrained_bert):
        super(DFGN, self).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(pretrained_bert)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.model = GraphFusionNet(config=config)

    def forward(self, batch, return_yp=True, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        N = doc_ids.shape[0]
        doc_encoding = self.bert_model(input_ids=doc_ids,
                                       token_type_ids=segment_ids,
                                       attention_mask=doc_mask,
                                       output_all_encoded_layers=False)
        doc_encoding = doc_encoding.detach()
        batch['context_encoding'] = doc_encoding
        start, end, sp, Type, softmask, ent, yp1, yp2 = self.model(batch, return_yp)
        return start, end, sp, Type, softmask, ent, yp1, yp2


