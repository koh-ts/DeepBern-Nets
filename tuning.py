import optuna
import wandb
from datasets import load_staliro
from models.models import FCModel
from train import compute_robust_loss_regression
import torch
import time
from tqdm import tqdm

def objective(trial):
    epochs = trial.suggest_int("epochs", 50, 100)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    alpha = trial.suggest_uniform("alpha", 0.0, 1.0)
    eps = trial.suggest_loguniform("eps", 1e-3, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    n_layers = trial.suggest_int("n_layers", 2, 10)
    hidden_size = trial.suggest_int("hidden_size", 32, 512)
    lr_decay_rate = trial.suggest_uniform("lr_decay_rate", 0.8, 1.0)
    lr_decay_start_epoch = trial.suggest_int("lr_decay_start_epoch", 10, 50)
    model_degree = trial.suggest_int("model_degree", 8, 32)

    device = "cuda"

    wandb.init(project="parameter_tuning", config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "alpha": alpha,
        "eps": eps,
        "weight_decay": weight_decay,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "lr_decay_rate": lr_decay_rate,
        "lr_decay_start_epoch": lr_decay_start_epoch,
        "model_degree": model_degree,
        "device": device,
    })

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

    best_val_loss = float('inf')
    epoch_times = []
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        epoch_s_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_robust_loss = 0.0
        total_samples = 0

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

            loss = alpha * tr_loss + (1 - alpha) * robust_loss

            loss.backward()
            optimizer.step()

            epoch_loss += tr_loss.item() * x.size(0)
            epoch_robust_loss += robust_loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_robust_loss = epoch_robust_loss / total_samples

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "train_robust_loss": avg_epoch_robust_loss,
        })

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

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_robust_loss": avg_val_robust_loss,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        epoch_e_time = time.perf_counter()
        print(f"Train Loss: {avg_epoch_loss:.4f}\n \
            Train Robust Loss: {avg_epoch_robust_loss:.4f}\n \
            Val Loss: {avg_val_loss:.4f}\n\
            Val Robust Loss: {avg_val_robust_loss:.4f}\n")
        epoch_times.append(epoch_e_time - epoch_s_time)

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)