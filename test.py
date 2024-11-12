import argparse
import torch
import sys
from time import perf_counter

# sys.path = ['/home/hkhedr/Haitham/projects/dev/test_autolirpa/auto_LiRPA'] + sys.path

# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import numpy as np
import yaml
from models.models import CNNb, CNNc, FCModel, CNNa
from tqdm import tqdm
import os
from datasets import load_cifar10, load_mnist, load_staliro
from utils import RecursiveNamespace

import torchattacks
import warnings
from torch.jit._trace import TracerWarning

warnings.filterwarnings(
    action="ignore",
    category=TracerWarning,
)
torch.manual_seed(123123)


def test_robust(model, testloader, device="cuda", eps=0.0, mode="ibp", verbose=True):
    model.eval()
    my_model = model
    mode = mode.lower()
    my_input = (next(iter(testloader))[0]).to(device)
    num_class = my_model(my_input).shape[-1]
    # if mode == "ibp":
    #     my_model = BoundedModule(model, my_input)

    total_cnt = 0
    correct_cnt = 0
    cert_cnt = 0
    for batch_idx, (x, target) in enumerate(tqdm(testloader)):
        total_cnt += x.shape[0]
        x, target = x.to(device), target.to(device)
        with torch.no_grad():
            out = my_model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_cnt += (pred_label == target.data).sum()

        if mode in ["ibp", "bern"]:
            with torch.no_grad():
                in_lb = torch.maximum(x - eps, torch.zeros_like(x))
                in_ub = torch.minimum(x + eps, torch.ones_like(x))
                # num_class = 10
                c = torch.eye(num_class).type_as(x)[target].unsqueeze(1) - torch.eye(
                    num_class
                ).type_as(x).unsqueeze(0)
                # remove specifications to self
                I = ~(
                    target.data.unsqueeze(1)
                    == torch.arange(num_class).type_as(target.data).unsqueeze(0)
                )
                c = c[I].view(x.size(0), num_class - 1, num_class)

                if mode == "bern":
                    inf_ball = torch.cat((in_lb.unsqueeze(-1), in_ub.unsqueeze(-1)), -1)
                    bern_bounds = my_model.forward_subinterval(inf_ball, C=c)
                    lb = bern_bounds[..., 0]
                    ub = bern_bounds[..., 1]
                    # y_one_hot = torch.nn.functional.one_hot(target)
                    # target_lb = (lb *  y_one_hot).sum(axis = -1)
                    # nontarget_ub = (ub *  (1-y_one_hot))
                    # nontarget_ub[nontarget_ub == 0]  = -torch.inf
                    # nontarget_ub = nontarget_ub.max(axis = -1)[0]
                    # robust_cnt = (pred_label == target.data) * (target_lb > nontarget_ub)
                    # robust_cnt = robust_cnt.sum()
                    robust_cnt = torch.sum((lb >= 0).all(dim=1)).item()
                # else:
                #     ptb = PerturbationLpNorm(norm=np.inf, x_L=in_lb, x_U=in_ub)

                #     # Make the input a BoundedTensor with the pre-defined perturbation.
                #     my_input = BoundedTensor(x, ptb)
                #     # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
                #     c = torch.eye(num_class).type_as(x)[target].unsqueeze(
                #         1
                #     ) - torch.eye(num_class).type_as(x).unsqueeze(0)
                #     # remove specifications to self
                #     I = ~(
                #         target.data.unsqueeze(1)
                #         == torch.arange(num_class).type_as(target.data).unsqueeze(0)
                #     )
                #     c = c[I].view(x.size(0), num_class - 1, num_class)
                #     lb, ub = my_model.compute_bounds(x=(my_input,), C=c, method="IBP")
                #     robust_cnt = torch.sum((lb >= 0).all(dim=1)).item()

        elif mode == "pgd":
            alpha = eps / 50
            with torch.enable_grad():
                adversary = torchattacks.PGD(my_model, eps=eps, alpha=alpha, steps=100)
                x_perturbed = adversary(x, target)
                out = my_model(x_perturbed)
            _, pred_label = torch.max(out.data, 1)
            robust_cnt = (pred_label == target.data).sum().item()
        else:
            # clean accuracy
            robust_cnt = 0

        cert_cnt += robust_cnt

    model_acc = correct_cnt * 100 / total_cnt
    cert_acc = cert_cnt * 100 / total_cnt
    if verbose:
        print(f"Test accuracy: {model_acc}% ({correct_cnt} / {total_cnt})")
        print(
            f"Certified accuracy (method={mode},eps = {eps}): {cert_acc}% ({cert_cnt} / {total_cnt})"
        )
    return model_acc, cert_acc

def test_robust_regression(model, testloader, device="cuda", eps=0.0, mode="ibp", verbose=True):
    model.eval()
    model.to(device)
    criterion = torch.nn.MSELoss()  # Use Mean Squared Error for regression
    total_loss = 0.0
    total_samples = 0
    all_worst_case_deviations = []

    for batch_idx, (x, target) in enumerate(tqdm(testloader)):
        x, target = x.to(device), target.to(device)
        batch_size = x.size(0)
        total_samples += batch_size

        with torch.no_grad():
            output = model(x)
            loss = criterion(output, target)
            total_loss += loss.item() * batch_size

        if mode in ["ibp", "bern", "IBP", "BERN"]:
            with torch.no_grad():
                in_lb = torch.maximum(x - eps, torch.zeros_like(x))
                in_ub = torch.minimum(x + eps, torch.ones_like(x))

                inf_ball = torch.cat((in_lb.unsqueeze(-1), in_ub.unsqueeze(-1)), -1)

                out_bounds = model.forward_subinterval(inf_ball)
                out_lb = out_bounds[..., 0]
                out_ub = out_bounds[..., 1]

                deviation_lb = torch.abs(out_lb - target)
                deviation_ub = torch.abs(out_ub - target)
                worst_case_deviation = torch.max(deviation_lb, deviation_ub)

                all_worst_case_deviations.append(worst_case_deviation.cpu())
        else:
            pass
        
    all_worst_case_deviations = torch.cat(all_worst_case_deviations)
    mean_worst_case_deviation = all_worst_case_deviations.mean().item()
    std_worst_case_deviation = all_worst_case_deviations.std().item()

    average_loss = total_loss / total_samples
    if verbose:
        print(f"Average Loss on Test Set: {average_loss:.6f}")
        print(f"Average Worst-Case Deviation: {mean_worst_case_deviation:.6f} ± {std_worst_case_deviation:.6f}")
    return average_loss, mean_worst_case_deviation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument(
        "--load", action=argparse.BooleanOptionalAction, help="LOAD best network"
    )
    parser.add_argument(
        "--result_file", type=str, default="", help="Results file to append results to"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--eps", type=float, help="Epsilon for PGD attack")

    args = parser.parse_args()
    # assert len(sys.argv) == 2, "Please provide a YAML configuration file as argument"
    yaml_cfg_f = args.config
    with open(yaml_cfg_f, "rb") as f:
        cfg = RecursiveNamespace(**yaml.safe_load(f))
    if args.device:
        cfg.TEST.DEVICE = args.device
    if args.eps:
        cfg.ROBUSTNESS.TEST_EPS = args.eps
    cfg.CHECKPOINT.LOAD = True if args.load is True else cfg.CHECKPOINT.LOAD
    degree = cfg.MODEL.DEGREE
    device = cfg.TEST.DEVICE
    is_FC_model = cfg.MODEL.TYPE == "FC"
    if cfg.DATASET == "cifar10":
        trainloader, testloader = load_cifar10(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model, samples_dist=100
        )
    elif cfg.DATASET == "mnist":
        trainloader, testloader = load_mnist(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model, samples_dist=1
        )
    elif cfg.DATASET == "staliro":
        trainloader, testloader = load_staliro(
            batch_size=cfg.TRAIN.BATCH_SIZE, flatten=is_FC_model, samples_dist=1
        )

    num_outs = len(trainloader.dataset.classes)
    in_shape = torch.tensor(next(iter(trainloader))[0][0].shape)
    if cfg.MODEL.TYPE == "FC":
        layer_sizes = [in_shape.item()] + cfg.MODEL.HIDDEN_LAYERS + [num_outs]
        in_bounds = torch.concat(
            (torch.zeros(in_shape, 1), torch.ones(in_shape, 1)), dim=-1
        ).to(device)
        model = FCModel(
            layer_sizes,
            degree=cfg.MODEL.DEGREE,
            act=cfg.MODEL.ACTIVATION,
            input_bounds=in_bounds,
        ).to(device)
    elif cfg.MODEL.TYPE == "CNNa":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNa(degree, input_bounds=in_bounds).to(device)
    elif cfg.MODEL.TYPE == "CNNb":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNb(degree, input_bounds=in_bounds).to(device)
    elif cfg.MODEL.TYPE == "CNNc":
        in_bounds = torch.concat(
            (torch.zeros(*in_shape, 1), torch.ones(*in_shape, 1)), dim=-1
        ).to(device)
        model = CNNc(degree, input_bounds=in_bounds).to(device)
    assert cfg.CHECKPOINT.LOAD, "CHECKPOINT.LOAD must be True"
    # assert cfg.CHECKPOINT.PATH_TO_CKPT is not None, "CHECKPOINT.PATH_TO_CKPT must be provided"
    try:
        if cfg.CHECKPOINT.PATH_TO_CKPT:
            ckpt = torch.load(cfg.CHECKPOINT.PATH_TO_CKPT, map_location="cpu")
        else:
            BASE_DIR = os.path.join(
                cfg.CHECKPOINT.DIR, cfg.EXPERIMENT.NAME, cfg.EXPERIMENT.RUN_NAME
            )
            ckpt = torch.load(
                f"{BASE_DIR}/checkpoint_best_model.pth", map_location="cpu"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
    except Exception as e:
        print(e)
        raise e
    eps = cfg.ROBUSTNESS.TEST_EPS
    if cfg.DATASET == "staliro":
        if cfg.TEST.IBP:
            s_time = perf_counter()
            print("======== IBP verification ========")
            test_loss, mean_worst_case_deviation = test_robust_regression(
                model, testloader, eps=eps, mode="IBP", device=device, verbose=False
            )
            e_time = perf_counter()
            print(f"Test loss: {test_loss}")
            print(f"Mean_worst_case_deviation (eps = {eps}): {mean_worst_case_deviation}")
            print(f"Time taken: {e_time - s_time}s")
        # if cfg.TEST.BERN_IBP:
        #     s_time = perf_counter()
        #     print("======== Bern-IBP verification ========")
        #     test_loss, mean_worst_case_deviation = test_robust_regression(
        #         model, testloader, eps=eps, mode="bern", device=device, verbose=False
        #     )
        #     e_time = perf_counter()
        #     print(f"Test loss: {test_loss}")
        #     print(f"Mean_worst_case_deviation (eps = {eps}): {mean_worst_case_deviation}")
        #     print(f"Time taken: {e_time - s_time}s")
        # if cfg.TEST.PGD:
        #     print("======== PGD verification ========")
        #     pgd_test_acc, pgd_cert_acc = test_robust_regression(
        #         model, testloader, eps=eps, mode="pgd", device=device, verbose=False
        #     )
        #     print(f"Test accuracy: {pgd_test_acc}%")
        #     print(f"Certified accuracy (eps = {eps}): {pgd_cert_acc}%")
    else:
        if cfg.TEST.MODE == "adv":
            if cfg.TEST.IBP:
                s_time = perf_counter()
                print("======== IBP verification ========")
                ibp_test_acc, ibp_cert_acc = test_robust(
                    model, testloader, eps=eps, mode="IBP", device=device, verbose=False
                )
                e_time = perf_counter()
                print(f"Test accuracy: {ibp_test_acc}%")
                print(f"Certified accuracy (eps = {eps}): {ibp_cert_acc}%")
                print(f"Time taken: {e_time - s_time}s")
            if cfg.TEST.BERN_IBP:
                s_time = perf_counter()
                print("======== Bern-IBP verification ========")
                bern_test_acc, bern_cert_acc = test_robust(
                    model, testloader, eps=eps, mode="bern", device=device, verbose=False
                )
                e_time = perf_counter()
                print(f"Test accuracy: {bern_test_acc}%")
                print(f"Certified accuracy (eps = {eps}): {bern_cert_acc}%")
                print(f"Time taken: {e_time - s_time}s")
            if cfg.TEST.PGD:
                print("======== PGD verification ========")
                pgd_test_acc, pgd_cert_acc = test_robust(
                    model, testloader, eps=eps, mode="pgd", device=device, verbose=False
                )
                print(f"Test accuracy: {pgd_test_acc}%")
                print(f"Certified accuracy (eps = {eps}): {pgd_cert_acc}%")
        elif cfg.TEST.MODE == "clean":
            test_acc, cert_acc = test_robust(
                model, testloader, eps=eps, mode="clean", device=device, verbose=False
            )
