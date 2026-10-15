"""Command-line preparation and offline pretrained inference."""
from __future__ import annotations
import argparse,csv,json,sys
import numpy as np
import torch
from . import __version__,load_pretrained,read_cells_csv,prepare_cells,HalfPlane,Corner
from .example import run_example,run_training_example
from .api import load_checkpoint
from .inputs import load_inputs
from .training import fit,TrainingConfig


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(prog='nicheir',description='Reconstruct four-class composition at known acquisition boundaries.')
    parser.add_argument('--version',action='version',version=__version__)
    sub=parser.add_subparsers(dest='command',required=True)
    example=sub.add_parser('example',help='Run a deterministic synthetic-cell example')
    example.add_argument('--output-dir',default='example_run');example.add_argument('--geometry',choices=['halfplane','corner'],default='halfplane')
    prepare=sub.add_parser('prepare',help='Convert a visible annotated-cell CSV to the six-array NPZ')
    prepare.add_argument('--cells',required=True);prepare.add_argument('--queries',required=True,help='CSV with a cell_id column')
    prepare.add_argument('--geometry',choices=['halfplane','corner'],required=True);prepare.add_argument('--cut',type=float,required=True,help='First canonical cut in micrometres')
    prepare.add_argument('--angle',type=float,default=0.,help='Halfplane outward-normal angle in radians')
    prepare.add_argument('--second-cut',type=float,help='Corner second canonical cut in micrometres')
    prepare.add_argument('--orientation',type=int,default=0,help='Corner orientation: 0,1,2,3 quarter turns')
    prepare.add_argument('--output',required=True);prepare.add_argument('--query-index',required=True)
    predict=sub.add_parser('predict',help='Predict a validated input NPZ with bundled weights')
    predict.add_argument('--inputs',required=True);predict.add_argument('--output',required=True);predict.add_argument('--device',default='cpu');predict.add_argument('--batch-windows',type=int,default=4)
    predict.add_argument('--checkpoint',help='Optional tensor-only weights from nicheir train')
    train=sub.add_parser('train',help='Supervised fit on prepared inputs and separate hidden-count targets')
    train.add_argument('--inputs',required=True);train.add_argument('--targets',required=True,help='NPZ containing only hidden_counts [W,Q,2,16,4]')
    train.add_argument('--output',required=True,help='Final checkpoint filename ending in .pt');train.add_argument('--epochs',type=int,default=20)
    train.add_argument('--batch-windows',type=int,default=4);train.add_argument('--seed',type=int,default=42)
    train.add_argument('--initialization',choices=['pretrained','random'],default='pretrained');train.add_argument('--device',default='cpu')
    train.add_argument('--learning-rate',type=float,default=3e-4)
    demo_train=sub.add_parser('train-example',help='Run a short supervised synthetic-cell training example')
    demo_train.add_argument('--output-dir',default='training_example');demo_train.add_argument('--epochs',type=int,default=2)
    args=parser.parse_args(argv);torch.set_num_threads(2)
    try:
        if args.command=='example':print(json.dumps(run_example(args.output_dir,args.geometry),indent=2))
        elif args.command=='train-example':print(json.dumps(run_training_example(args.output_dir,args.epochs),indent=2))
        elif args.command=='train':
            if not args.output.endswith('.pt'):raise ValueError('--output must end in .pt')
            with np.load(args.targets,allow_pickle=False) as z:
                if set(z.files)!={'hidden_counts'}:raise ValueError('Target NPZ must contain exactly hidden_counts')
                targets=z['hidden_counts']
            config=TrainingConfig(epochs=args.epochs,batch_windows=args.batch_windows,seed=args.seed,learning_rate=args.learning_rate)
            result=fit(load_inputs(args.inputs),targets,config=config,initialization=args.initialization,device=args.device)
            result.save(args.output);print(json.dumps(dict(epochs=args.epochs,seed=args.seed,final_loss=result.history[-1]['loss'])))
        elif args.command=='prepare':
            with open(args.queries,newline='',encoding='utf-8-sig') as f:
                reader=csv.DictReader(f)
                if 'cell_id' not in (reader.fieldnames or []):raise ValueError('Query CSV requires cell_id column')
                ids=[r['cell_id'] for r in reader]
            if args.geometry=='corner':
                if args.second_cut is None:raise ValueError('--second-cut is required for a corner')
                if args.angle!=0:raise ValueError('Use --orientation, not --angle, for corner geometry')
                boundary=Corner(args.cut,args.second_cut,args.orientation)
            else:
                if args.second_cut is not None or args.orientation!=0:raise ValueError('Corner-only arguments were supplied to a halfplane')
                boundary=HalfPlane(args.cut,args.angle)
            b=prepare_cells(read_cells_csv(args.cells),ids,boundary);b.save(args.output);b.save_query_index(args.query_index)
            print(json.dumps(dict(queries=len(ids),windows=len(b.inputs['valid']),output=args.output)))
        else:
            predictor=load_pretrained(device=args.device) if args.checkpoint is None else load_checkpoint(args.checkpoint,device=args.device)
            p=predictor.predict_npz(args.inputs,output=args.output,batch_windows=args.batch_windows)
            print(json.dumps(dict(queries=int(p.valid.sum()),output=args.output)))
    except (ValueError,TypeError,OSError,RuntimeError) as e:parser.exit(2,f'nicheir: {e}\n')
    return 0

if __name__=='__main__':sys.exit(main())
