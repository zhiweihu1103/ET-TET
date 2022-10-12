import argparse
from utils import *
from TET import TET
from dataloader import ETDataset
from torch.utils.data import DataLoader

def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])
    save_path = os.path.join(args['save_dir'], args['save_path'])

    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    c2id = read_id(os.path.join(data_path, 'clusters.tsv'))
    num_entities = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_clusters = len(c2id)
    train_type_label, test_type_label = load_train_all_labels(data_path, e2id, t2id)
    if use_cuda:
        sample_ent2pair = torch.LongTensor(load_entity_cluster_type_pair_context(args, r2id, e2id)).cuda()
    train_dataset = ETDataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
    valid_dataset = ETDataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')
    test_dataset = ETDataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=ETDataset.collate_fn,
                                  num_workers=6)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args['train_batch_size'],
                                  shuffle=False,
                                  collate_fn=ETDataset.collate_fn,
                                  num_workers=6)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args['test_batch_size'],
                                  shuffle=False,
                                  collate_fn=ETDataset.collate_fn,
                                  num_workers=6)

    model = TET(args, num_entities, num_rels, num_types, num_clusters)

    if use_cuda:
        model = model.to('cuda')
    for name, param in model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    current_learning_rate = args['lr']
    warm_up_steps = args['warm_up_steps']
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )

    max_valid_mrr = 0
    model.train()
    for epoch in range(args['max_epoch']):
        log = []
        for sample_et_content, sample_kg_content, gt_ent in train_dataloader:
            type_label = train_type_label[gt_ent, :]
            if use_cuda:
                sample_et_content = sample_et_content.cuda()
                sample_kg_content = sample_kg_content.cuda()
                type_label = type_label.cuda()
            type_predict = model(sample_et_content, sample_kg_content, sample_ent2pair)

            if args['loss'] == 'BCE':
                bce_loss = torch.nn.BCELoss()
                type_loss = bce_loss(type_predict, type_label)
                type_pos_loss, type_neg_loss = type_loss, type_loss
            elif args['loss'] == 'SFNA':
                type_pos_loss, type_neg_loss = slight_fna_loss(type_predict, type_label, args['beta'])
                type_loss = type_pos_loss + type_neg_loss
            else:
                raise ValueError('loss %s is not defined' % args['loss'])

            log.append({
                "loss": type_loss.item(),
                "pos_loss": type_pos_loss.item(),
                "neg_loss": type_neg_loss.item(),
            })

            optimizer.zero_grad()
            type_loss.requires_grad_(True)
            type_loss.backward()
            optimizer.step()

        if epoch >= warm_up_steps:
            current_learning_rate = current_learning_rate / 5
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 2

        avg_type_loss = sum([_['loss'] for _ in log]) / len(log)
        avg_type_pos_loss = sum([_['pos_loss'] for _ in log]) / len(log)
        avg_type_neg_loss = sum([_['neg_loss'] for _ in log]) / len(log)
        logging.debug('epoch %d: loss: %f\tpos_loss: %f\tneg_loss: %f' %
                      (epoch, avg_type_loss, avg_type_pos_loss, avg_type_neg_loss))

        if epoch != 0 and epoch % args['valid_epoch'] == 0:
            model.eval()
            with torch.no_grad():
                logging.debug('-----------------------valid step-----------------------')
                predict = torch.zeros(num_entities, num_types, dtype=torch.half)
                for sample_et_content, sample_kg_content, gt_ent in valid_dataloader:
                    if use_cuda:
                        sample_et_content = sample_et_content.cuda()
                        sample_kg_content = sample_kg_content.cuda()
                    predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
                valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, test_type_label, e2id, t2id)

                logging.debug('-----------------------test step-----------------------')
                predict = torch.zeros(num_entities, num_types, dtype=torch.half)
                for sample_et_content, sample_kg_content, gt_ent in test_dataloader:
                    if use_cuda:
                        sample_et_content = sample_et_content.cuda()
                        sample_kg_content = sample_kg_content.cuda()
                    predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
                evaluate(os.path.join(data_path, 'ET_test.txt'), predict, test_type_label, e2id, t2id)

            model.train()
            if valid_mrr < max_valid_mrr:
                logging.debug('early stop')
                break
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
                max_valid_mrr = valid_mrr

                # save embedding
                entity_embedding = model.entity.detach().cpu().numpy()
                np.save(
                    os.path.join(save_path, 'entity_embedding'),
                    entity_embedding
                )
                relation_embedding = model.relation.detach().cpu().numpy()
                np.save(
                    os.path.join(save_path, 'relation_embedding'),
                    relation_embedding
                )

    logging.debug('-----------------------best test step-----------------------')
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
        model.eval()
        predict = torch.zeros(num_entities, num_types, dtype=torch.half)
        for sample_et_content, sample_kg_content, gt_ent in test_dataloader:
            if use_cuda:
                sample_et_content = sample_et_content.cuda()
                sample_kg_content = sample_kg_content.cuda()
            predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
        evaluate(os.path.join(data_path, 'ET_test.txt'), predict, test_type_label, e2id, t2id)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--save_path', type=str, default='SFNA')
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--valid_epoch', type=int, default=25)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--loss', type=str, default='SFNA')

    # params for first trm layer
    parser.add_argument('--bert_nlayer', type=int, default=3)
    parser.add_argument('--bert_nhead', type=int, default=4)
    parser.add_argument('--bert_ff_dim', type=int, default=480)
    parser.add_argument('--bert_activation', type=str, default='gelu')
    parser.add_argument('--bert_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
    parser.add_argument('--local_pos_size', type=int, default=200)

    # params for pair trm layer
    parser.add_argument('--pair_layer', type=int, default=3)
    parser.add_argument('--pair_head', type=int, default=4)
    parser.add_argument('--pair_dropout', type=float, default=0.2)
    parser.add_argument('--pair_ff_dim', type=int, default=480)

    # params for second trm layer
    parser.add_argument('--trm_nlayer', type=int, default=3)
    parser.add_argument('--trm_nhead', type=int, default=4)
    parser.add_argument('--trm_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--trm_attn_dropout', type=float, default=0.2)
    parser.add_argument('--trm_ff_dim', type=int, default=480)
    parser.add_argument('--global_pos_size', type=int, default=200)

    parser.add_argument('--pair_pooling', type=str, default='avg', choices=['max', 'avg', 'min'])
    parser.add_argument('--sample_et_size', type=int, default=3)
    parser.add_argument('--sample_kg_size', type=int, default=7)
    parser.add_argument('--sample_ent2pair_size', type=int, default=6)
    parser.add_argument('--warm_up_steps', default=50, type=int)
    parser.add_argument('--tt_ablation', type=str, default='all', choices=['all', 'triple', 'type'],
                        help='ablation choice')
    parser.add_argument('--log_name', type=str, default='log')

    args, _ = parser.parse_known_args()
    print(args)
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
