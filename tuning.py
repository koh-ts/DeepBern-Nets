import optuna
import wandb
from datasets import load_staliro
from models.models import FCModel
from train import compute_robust_loss_regression
import torch
import time
from tqdm import tqdm
import json
import torch.optim as optim
import torch.nn as nn
from utils import ParamScheduler, RecursiveNamespace


def objective(trial):
    # epochs = trial.suggest_int("epochs", 100, 200)
    epochs = 50
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    # learning_rate = 0.0003
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    # alpha = trial.suggest_uniform("alpha", 0.0, 1.0)
    alpha = 0.76
    # eps = trial.suggest_loguniform("eps", 1e-3, 1e-1)
    eps = 0.1
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    n_layers = trial.suggest_int("n_layers", 3, 4)
    hidden_size = trial.suggest_int("hidden_size", 256, 1024)
    lr_decay_rate = trial.suggest_uniform("lr_decay_rate", 0.7, 1.0)
    # lr_decay_rate = 0.84
    lr_decay_start_epoch = trial.suggest_int("lr_decay_start_epoch", 20, 40)
    model_degree = trial.suggest_int("model_degree", 4, 8)
    l1_lambda = trial.suggest_loguniform("l1_lambda", 1e-5, 1e-3)

    device = "cuda:1"
    torch.cuda.set_device(1)

    # wandb.init(project="parameter_tuning", config={
    #     "epochs": epochs,
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "alpha": alpha,
    #     "eps": eps,
    #     "weight_decay": weight_decay,
    #     "n_layers": n_layers,
    #     "hidden_size": hidden_size,
    #     "lr_decay_rate": lr_decay_rate,
    #     "lr_decay_start_epoch": lr_decay_start_epoch,
    #     "model_degree": model_degree,
    #     "l1_lambda": l1_lambda,
    #     "device": device,
    # })

    trainloader, testloader = load_staliro(batch_size=batch_size, flatten=True)

    in_shape = torch.tensor(next(iter(trainloader))[0][0].shape)
    num_outs = 1
    layer_sizes = [in_shape.item()] + [hidden_size] * n_layers + [num_outs]
    in_bounds = torch.concat(
        (torch.zeros(in_shape, 1), torch.ones(in_shape, 1)), dim=-1
    ).to(device)
    model = FCModel(
        layer_sizes,
        degree=model_degree,
        act="bern",
        input_bounds=in_bounds,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    alpha_scheduler = ParamScheduler(
        "linear",
        0,
        epochs,
        0.9,
        0.1,
    )

    bounding_method = "bern"
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=lr_decay_rate
    )

    best_val_loss = float('inf')
    epoch_times = []
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        epoch_s_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_robust_loss = 0.0
        total_samples = 0
        alpha = alpha_scheduler.step(epoch)

        for batch_idx, (x, y) in enumerate(tqdm(trainloader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            tr_loss = loss_fn(pred, y)

            robust_loss = compute_robust_loss_regression(
                model,
                x,
                y,
                eps=eps,
                bounding_method="bern",
            )

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = alpha * tr_loss + (1 - alpha) * robust_loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model(x[0:1])

            epoch_loss += loss.item() * x.size(0)
            epoch_robust_loss += robust_loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_robust_loss = epoch_robust_loss / total_samples

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": avg_epoch_loss,
        #     "train_robust_loss": avg_epoch_robust_loss,
        #     # "epochs": epochs,
        #     "learning_rate": learning_rate,
        #     "batch_size": batch_size,
        #     # "alpha": alpha,
        #     # "eps": eps,
        #     "weight_decay": weight_decay,
        #     "n_layers": n_layers,
        #     "hidden_size": hidden_size,
        #     "lr_decay_rate": lr_decay_rate,
        #     "lr_decay_start_epoch": lr_decay_start_epoch,
        #     "model_degree": model_degree,
        #     "l1_lambda": l1_lambda,
        #     # "device": device,
        # })

        model.eval()
        val_loss, val_robust_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)

                robust_loss = compute_robust_loss_regression(
                    model,
                    x,
                    y,
                    eps=eps,
                    bounding_method="bern",
                )
                val_robust_loss += robust_loss.item() * x.size(0)

        avg_val_loss = val_loss / len(testloader.dataset)
        avg_val_robust_loss = val_robust_loss / len(testloader.dataset)

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "val_loss": avg_val_loss,
        #     "val_robust_loss": avg_val_robust_loss,
        # })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        epoch_e_time = time.perf_counter()
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_epoch_loss:.4f}\n \
        Train Robust Loss: {avg_epoch_robust_loss:.4f}\n \
        Val Loss: {avg_val_loss:.4f}\n\
        Val Robust Loss: {avg_val_robust_loss:.4f}\n")
        epoch_times.append(epoch_e_time - epoch_s_time)
        print(f"Epoch Time: {epoch_e_time - epoch_s_time:.2f}s\n")

        if epoch > lr_decay_start_epoch:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_robust_loss)
            else:
                lr_scheduler.step()

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",
                                storage='sqlite:///optuna_study_0.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=50)
    trials = study.trials
    trials_json = [trial.params for trial in trials]

    with open('/home/koh/work/DeepBern-Nets/optuna_trials.json', 'w') as f:
        json.dump(trials_json, f, indent=2) 