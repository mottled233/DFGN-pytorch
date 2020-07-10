from os.path import join
from model.model import *
from utils import *
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
from config import set_config
from tools.data_helper import DataHelper
from text_to_tok_pack import *
from hotpot_evaluate_v1 import eval


def compute_loss(batch, start, end, sp, Type, masks):
    loss1 = criterion(start, batch['y1']) + criterion(end, batch['y2'])
    loss2 = args.type_lambda * criterion(Type, batch['q_type'])
    loss3 = args.sp_lambda * criterion(sp.view(-1, 2), batch['is_support'].long().view(-1))
    loss = loss1 + loss2 + loss3

    loss4 = 0
    if args.bfs_clf:
        for l in range(args.n_layers):
            pred_mask = masks[l].view(-1)
            gold_mask = batch['bfs_mask'][:, l, :].contiguous().view(-1)
            loss4 += binary_criterion(pred_mask, gold_mask)
        loss += args.bfs_lambda * loss4

    return loss, loss1, loss2, loss3, loss4


def predict(model, dataloader, example_dict, feature_dict, prediction_file):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5
    for batch in tqdm(dataloader):
        start, end, sp, Type, softmask, ent, yp1, yp2 = model(batch, return_yp=True)

        loss_list = compute_loss(batch, start, end, sp, Type, softmask)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], yp1.data.cpu().numpy().tolist(),
                                         yp2.data.cpu().numpy().tolist(), np.argmax(Type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

    eval(prediction_file, "data/dev.json")

    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))
    model.train()


def train_batch(model, batch):
    global global_step, total_train_loss

    start, end, sp, Type, softmask, ent, yp1, yp2 = model(batch, return_yp=True)
    loss_list = compute_loss(batch, start, end, sp, Type, softmask)
    loss_list[0].backward()

    if (global_step + 1) % args.grad_accumulate_step == 0:
        optimizer.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{} Step{}: ".format(args.name, epc, global_step))
        for i, l in enumerate(total_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        total_train_loss = [0] * 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    # Allocate Models on GPU
    model_gpu = 'cuda:{}'.format(args.model_gpu)

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type

    # Set datasets
    Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader

    # Set Model
    model = DFGN(config=args, pretrained_bert=args.bert_model)
    model.cuda(model_gpu)
    model.train()
    print("Model initialized.")

    # Initialize optimizer and criterions
    lr = args.lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(size_average=True)

    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    # lock = threading.Lock()
    VERBOSE_STEP = args.verbose_step
    print("Start training.")
    while True:
        # lock.acquire()
        if epc == args.qat_epochs + args.epochs:
            exit(0)
        epc += 1

        # learning rate decay
        if epc > 1:
            lr = lr * args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('lr = {}'.format(lr))

        # Early Stopping
        if epc > args.early_stop_epoch + 1:
            if test_loss_record[-1] > test_loss_record[-(1 + args.early_stop_epoch)]:
                print("Early Stop in epoch{}".format(epc))
                for i, test_loss in enumerate(test_loss_record):
                    print(i, test_loss_record)
                exit(0)
        print("No need for early stop")

        # if epc <= args.qat_epochs:
        #     Loader = Subset_Loader
        # else:
        Loader = Full_Loader
        Loader.refresh()

        while not Loader.empty():
            batch = next(iter(Loader))
            train_batch(model, batch)

        predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                join(args.prediction_path, 'pred_epoch_{}.json'.format(epc)))
        torch.save(model.state_dict(), join(args.checkpoint_path, "ckpt_epoch_{}.pth".format(epc)))

