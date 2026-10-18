from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW, learning_rate_schedule, gradient_clipping
from cs336_basics.nn import TransformerLM, cross_entropy
import numpy as np
import wandb, time, torch
import logging, os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def log_info(logger, iteration, train_loss, valid_loss, _lr, grad_norm, ts_start):
    time_elapsed = time.time() - ts_start
    logger.info(f"Iteration {iteration}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss if valid_loss is not None else float('nan'):.6f}, LR: {_lr:.6e}, Gradient Norm: {grad_norm:.6f}, Time Elapsed: {time_elapsed / 60 :.6f} minutes")
    message = {
            "iteration": iteration, 
            "train_loss": train_loss,
            "lr": _lr,
            "gradient_norm": grad_norm,
            "time_elapsed": time.time() - ts_start
        }

    if valid_loss is not None:
        message.update({"valid_loss": valid_loss})
        message.update({"train_loss/valid_loss": train_loss / valid_loss if valid_loss > 0 else 0.0})

    wandb.log(message)

def save_checkpoint(params, model, optimizer, iteration, checkpoint_save_path):
    #clean up checkpoint when have more than 10 files
    if os.listdir(checkpoint_save_path):
        ckpt_files = [f for f in os.listdir(checkpoint_save_path) if f.startswith("iteration_")]
        if len(ckpt_files) > 10:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            for f in ckpt_files[:-10]:
                os.remove(os.path.join(checkpoint_save_path, f))

    torch.save({
        "params": params,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, f"{checkpoint_save_path}/iteration_{iteration}.pt")

def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]

def run_evaluation(model, dataset, batch_size, context_length, device, max_batches=None):
    model.eval()
    starts = np.arange(0, len(dataset) - context_length, context_length)  # 不重叠
    if max_batches is not None:
        starts = starts[: max_batches * batch_size]
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(starts), batch_size):                     # 分批前向
            chunk = np.array([dataset[j:j + context_length + 1]
                              for j in starts[b:b + batch_size]]).astype(np.int64)
            inputs = torch.from_numpy(chunk[:, :-1]).to(device)
            targets = torch.from_numpy(chunk[:, 1:]).to(device)
            loss = cross_entropy(model(inputs), targets)
            n = targets.numel()                                         # B*T
            total_loss += loss.item() * n
            total_tokens += n
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0

if __name__ == "__main__":
    data_name = "TinyStoriesV2-GPT4-train"
    train_data_file_path = f"out/{data_name}_tokens.bin"
    valid_data_file_path = f"out/{data_name.replace('train', 'valid')}_tokens.bin"
    checkpoint_save_path = "checkpoints"

    vocab_size = 10000
    batch_size = 64
    context_length = 256
    device = "mps"
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0

    os.makedirs(checkpoint_save_path, exist_ok=True)

    model = TransformerLM(vocab_size=vocab_size, 
                          context_length=context_length, 
                          d_model=d_model,
                          num_layers=num_layers, 
                          num_heads=num_heads, 
                          d_ff=d_ff,
                          rope_theta=rope_theta,
                          device=device)

    target_total_tokens = 327_680_000
    max_iterations = target_total_tokens // (batch_size * context_length)
    # max_iterations = 10000
    warmup_steps = 300
    max_learning_rate=3e-4
    min_learning_rate = max_learning_rate * 0.1

    betas=(0.9, 0.95)
    eps=1e-8
    weight_decay=0.01

    params = {
                   "vocab_size": vocab_size,
                   "batch_size": batch_size,
                   "context_length": context_length,
                   "device": device,
                   "d_model": d_model,
                   "num_layers": num_layers,
                   "num_heads": num_heads,
                   "d_ff": d_ff,
                   "rope_theta": rope_theta,
                   "max_iterations": max_iterations,
                   "warmup_steps": warmup_steps,
                   "max_learning_rate": max_learning_rate,
                   "min_learning_rate": min_learning_rate,
                   "betas": betas,
                   "eps": eps,
                   "weight_decay": weight_decay
               }

    def _get_learning_rate(iteration):
        return learning_rate_schedule(iteration,
                                      max_learning_rate, 
                                      min_learning_rate,
                                      warmup_steps,
                                      max_iterations)
    
    optimizer = AdamW(model.parameters(), lr=_get_learning_rate(0),
                    betas=betas, 
                    eps=eps, 
                    weight_decay=weight_decay)
    
    train_dataset = np.memmap(train_data_file_path, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(valid_data_file_path, dtype=np.uint16, mode="r")

    wandb.init(project="cs336-homework1", 
               name=f"data{data_name}_it{max_iterations}_lr{max_learning_rate}_bs{batch_size}_betas{betas}",
               config=params)

    ts_start = time.time()

    save_checkpoint_every = 1000
    evaluate_every = 100

    if device == "mps":
        model = torch.compile(model, backend="aot_eager")

    for iteration in range(max_iterations):
        _lr = _get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = _lr

        if iteration % save_checkpoint_every == 0:
            save_checkpoint(params, model, optimizer, iteration, checkpoint_save_path)

        if iteration % evaluate_every == 0:
            valid_loss = run_evaluation(model, valid_dataset, batch_size, context_length, device, max_batches=20)


        x, y = get_batch(train_dataset, batch_size, context_length, device)
        loss = cross_entropy(model(x), y)

        loss.backward()

        grad_norm = gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        log_info(logger, iteration, loss.item(), valid_loss if iteration % evaluate_every == 0 else None, _lr, grad_norm, ts_start)

    save_checkpoint(params, model, optimizer, max_iterations, checkpoint_save_path)
    final_valid_loss = run_evaluation(model, valid_dataset, batch_size, context_length, device)  # 全量
    logger.info(f"Final valid loss: {final_valid_loss:.6f}")
    wandb.log({"iteration": max_iterations, "final_valid_loss": final_valid_loss})
    wandb.finish()

        