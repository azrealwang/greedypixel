import os
import argparse
import time
import torch
import timm
from robustbench import load_model
from utils import save_all_images,load_samples,predict
from greedypixel import GreedyPixel

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # attack settings
    parser.add_argument("--eps", help="perturbation budget: epsilon/255", type=float, default=4)
    parser.add_argument("--max_query", help="maximum query cost", type=int, default=20000)
    parser.add_argument("--surrogate", help="for computing gradient map; none is random", type=str, default=None)
    parser.add_argument("--early_stop", help="stop when label changes", action="store_true")
    # target
    parser.add_argument("--target", help="target model", type=str, required=True)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=None)
    # input and output
    parser.add_argument("--data", help="cifar10 or imagenet", type=str, default="cifar10")
    parser.add_argument("--input", help="directory", type=str, required=True)
    parser.add_argument("--output", help="directory", type=str, required=True)
    parser.add_argument('--start_idx', help='start idx', type=int, default=0)
    parser.add_argument('--end_idx', help='end idx, id+1', type=int, default=10)
    args = parser.parse_args()

    return args

def main(args):
    # Load examples
    x_test, y_test = load_samples(args.input,args.start_idx,args.end_idx)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).long()

    # Load target model
    if args.target == 'ViT': 
        target = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000).eval()
    else:
        target = load_model(args.target, dataset=args.data, threat_model="Linf")
    if args.batch_size is None:
        args.batch_size = len(y_test)

    # Compute clean accuracy
    predictions = predict(target,x_test,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()
    print(f"Accuracy of clean examples: {accuracy}")
    
    # Initial attack
    os.makedirs(args.output, exist_ok=True)
    eps = args.eps / 255.0  # Normalize eps
    if args.surrogate:
        surrogate = load_model(args.surrogate, dataset=args.data, threat_model="Linf")
    else:
        surrogate = None
    attack = GreedyPixel(
        target,
        surrogate,
        eps,
        args.max_query,
        args.early_stop
        )

    # Run attack
    start_time = time.time()
    x_test_adv = x_test.clone()
    queries = list()
    for idx, (x, y) in enumerate(zip(x_test, y_test)):
        print(f"Sample {args.start_idx + idx + 1}...")
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x_adv, query = attack.attack(x,y)
        print(f"Query Cost: {query}")
        # Save adversarial examples
        save_all_images(x_adv, y, args.output, args.start_idx+idx) 
        x_test_adv[idx] = x_adv[0]
        queries.append(query)   
    end_time = time.time()
    time_cost = end_time - start_time # record time cost
    queries = torch.Tensor(queries)
    print(f"Time cost: {time_cost}s; Avg.Q: {queries.mean()}; Median.Q: {queries.median()}")
    
    # Robust accuracy
    d_linf = (x_test_adv-x_test).abs().max()*255
    predictions = predict(target,x_test_adv,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()
    print(f"Accuracy of adversarial examples: {accuracy}; Linf distance: {d_linf}")
    x_test_adv_load, _= load_samples(args.output,args.start_idx,args.end_idx)
    x_test_adv_load = torch.Tensor(x_test_adv_load)
    load_err = (x_test_adv_load-x_test_adv).abs().max()*255
    predictions = predict(target,x_test_adv_load,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()
    print(f"Accuracy of loaded adversarial examples: {accuracy}; Linf error: {load_err}")

if __name__ == "__main__":
    args = parse_args_and_config()
    print(args)
    main(args)