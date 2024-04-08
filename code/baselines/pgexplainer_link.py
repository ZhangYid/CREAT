import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import pickle
from tqdm.auto import tqdm
from pathlib import Path
from utils import set_seed, print_args, set_config_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from baseline_explainer import HeteroPGExplainer

parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='/home/lab/zyd/CREAT/datasets')
parser.add_argument('--dataset_name', type=str, default='graph_AP+TRQ')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=64)
# parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_dir', type=str, default='/home/lab/zyd/CREAT/code/saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='drug', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='disease', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='treats', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')

'''
Explanation args
'''
parser.add_argument('--lr', type=float, default=0.005, help='explainer learning_rate') 
parser.add_argument('--alpha1', type=float, default=1e-2, help='explainer sparsity regularizer weight') 
parser.add_argument('--alpha2', type=float, default=5e-4, help='explainer entropy regularizer weight') 
parser.add_argument('--mask_generator_hidden_dim', type=int, default=64, 
                    help='hidden dimension of mask generator') 
parser.add_argument('--num_hops', type=int, default=5, help='computation graph number of hops') 
parser.add_argument('--num_epochs', type=int, default=100, help='How many epochs to train PGExplainer')
parser.add_argument('--train_pgexplainer', default=False, action='store_true', 
                    help='Whether to train a new pgexplainer')
parser.add_argument('--save_pgexplainer', default=False, action='store_true', 
                    help='Whether to save the trained pgexplainer')
parser.add_argument('--load_pgexplainer', default=False, action='store_true', 
                    help='Whether to load the saved pgexplainer')
parser.add_argument('--batch_size', type=int, default=0, 
                    help='if > 0, sample a random batch to train the mask generator, else use all training nodes')
parser.add_argument('--save_explanation', default=True, action='store_true', 
                    help='Whether to save the explanation')
parser.add_argument('--saved_explanation_dir', type=str, default='/home/lab/zyd/CREAT/code/saved_explanations',
                    help='directory of saved explanations')
# parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
#                     help='directory of saved explanations')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')


args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pgexp')

if 'HERB' in args.dataset_name:
    args.src_ntype = 'herb'
    args.tgt_ntype = 'disease'
    args.pred_etype = 'has-effects-on'
else:
    args.src_ntype = 'drug'
    args.tgt_ntype = 'disease'
    args.pred_etype = 'treats'

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')
    
if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}
    
if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'

print_args(args)
set_seed(0)

processed_g = load_dataset(args.dataset_dir, args.dataset_name, args.pred_etype, args.valid_ratio, args.test_ratio, eval_exp=False)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in processed_g]

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)    

pgexplainer = HeteroPGExplainer(model, 
                                num_hops=args.num_hops, 
                                ghetero=mp_g,
                                lr=args.lr,
                                alpha1=args.alpha1, 
                                alpha2=args.alpha2, 
                                in_dim=args.out_dim, 
                                mask_generator_hidden_dim=args.mask_generator_hidden_dim,
                                num_epochs=args.num_epochs).to(device)

if args.train_pgexplainer:
    train_src_nids, train_tgt_nids = train_pos_g.edges()
    pgexplainer.train_mask_generator(train_src_nids, train_tgt_nids, mp_g, batch_size=args.batch_size)
    
elif args.load_pgexplainer:
    output_dir = Path.cwd().joinpath(args.saved_explanation_dir)
    saved_pgexplianer = f'pgexp_{args.saved_model_name}_explainer.pth'
    state = torch.load(output_dir.joinpath(saved_pgexplianer))
    pgexplainer.load_state_dict(state)    

test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = range(test_src_nids.shape[0])
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

pred_edge_to_comp_g_edge_mask = {}
for i in tqdm(test_ids):
    # Get the k-hop subgraph
    src_nid, tgt_nid = test_src_nids[i].unsqueeze(0), test_tgt_nids[i].unsqueeze(0)
    
    with torch.no_grad():
        pred = model(src_nid, tgt_nid, mp_g).sigmoid().item() > 0.5

        if pred:
            comp_g_edge_mask_dict = pgexplainer.explain(src_nid, tgt_nid, mp_g)

            src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
            pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict

if args.save_explanation:
    if not os.path.exists(args.saved_explanation_dir):
        os.makedirs(args.saved_explanation_dir)
        
    if args.save_pgexplainer:
        output_dir = Path.cwd().joinpath(args.saved_explanation_dir)
        saved_pgexplianer = f'pgexp_{args.saved_model_name}_explainer.pth'
        torch.save(pgexplainer.state_dict(), output_dir.joinpath(saved_pgexplianer))        

    
    saved_edge_explanation_file = f'pgexp_{args.saved_model_name}_pred_edge_to_comp_g_edge_mask'
    pred_edge_to_comp_g_edge_mask = {edge: {k: v.cpu() for k, v in mask.items()} for edge, mask in pred_edge_to_comp_g_edge_mask.items()}
    saved_edge_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_edge_explanation_file)
    with open(saved_edge_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_comp_g_edge_mask, f)
