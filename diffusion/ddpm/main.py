import logging
import os
from Diffusion.diffusion.ddpm.diffusers import DDPM
from Diffusion.diffusion.ddpm.models import BasicDiscreteTimeModel, NaiveNeuralNetworkNoiseModel
from typing import Any, List, Tuple
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from matplotlib import animation
from fire import Fire
from tqdm import tqdm
from pydantic import BaseModel

from statistical_distance.layers import SinkhornDistance

# not sure if this is necessary
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TrainResult(BaseModel):
    ema_losses: List[float]
    samples: List[Any]
    sinkhorn_benchmark_value: float
    sinkhorn_values: List[Tuple[int, float]]


def train(
        noise_model: nn.Module,
        ddpm: DDPM,
        device: torch.device,
        batch_size: int = 128,
        n_epochs: int = 1000,
        sample_size: int = 512,
        steps_between_sampling: int = 500,
        seed: int = 42) -> TrainResult:
    """

    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert batch_size > 0 and steps_between_sampling > 0 and sample_size > 0

    N = 1 << 10
    X = make_swiss_roll(n_samples=N, noise=1e-1)[0][:, [0, 2]] / 10.0

    optim = torch.optim.Adam(noise_model.parameters(), 1e-3)
    sinkhorn_calculator = SinkhornDistance(eps=0.1, max_iter=100, device=device)
    ema_losses: List[float] = []
    samples: List[Any] = []
    step = 0
    ema_loss = None  # exponential moving average
    ema_loss_alpha = 0.01
    # Set bench-mark sinkhorn value for this dataset
    logger.info(f"Setting benchmark sinkhorn value")
    sinkhorn_benchmark_sample_size = sample_size * 4
    X_ref_1 = torch.tensor(make_swiss_roll(n_samples=sinkhorn_benchmark_sample_size, noise=1e-1)[0][:, [0, 2]] / 10.0,
                           device=device)
    X_ref_2 = torch.tensor(make_swiss_roll(n_samples=sinkhorn_benchmark_sample_size, noise=1e-1)[0][:, [0, 2]] / 10.0,
                           device=device)
    diff_norm = torch.norm(X_ref_1 - X_ref_2)
    logger.info(f"Norm of diff between two reference sample = {diff_norm}")
    sinkhorn_benchmark_value = sinkhorn_calculator(X_ref_1, X_ref_2)[0].item()
    logger.info(f"*** Benchmark sinkhorn value = {sinkhorn_benchmark_value} ***")
    sinkhorn_values = []
    # total number of train loop iterations = number of batches * num_epochs
    # i.e. the loop will touch each batch n_epoch times
    num_batches = len(X) // batch_size
    n_iter = n_epochs * num_batches
    with tqdm(total=n_iter) as pbar:
        for _ in range(n_epochs):
            ids = np.random.choice(N, N, replace=False)  # shuffling
            for i in range(0, len(ids), batch_size):
                x = torch.tensor(X[ids[i: i + batch_size]], dtype=torch.float32, device=device)
                optim.zero_grad()
                loss = ddpm.diffusion_loss(noise_model, x)
                loss.backward()
                optim.step()

                pbar.update(1)
                ema_losses.append(loss.item())
                if ema_loss is None:
                    ema_loss = ema_losses[-1]
                else:
                    ema_loss = (1 - ema_loss_alpha) * ema_loss + ema_loss_alpha * loss.item()
                if not step % 10:
                    pbar.set_description(f"Iter: {step}. EMA Loss: {ema_loss:.04f}")
                if not step % steps_between_sampling:
                    generated_sample = ddpm.sample(noise_model, n_samples=sinkhorn_benchmark_sample_size)
                    samples.append(generated_sample)
                    # add sinkhorn eval
                    ref_sample = torch.tensor(
                        make_swiss_roll(n_samples=sinkhorn_benchmark_sample_size, noise=1e-1)[0][:, [0, 2]] / 10.0,
                        dtype=generated_sample.dtype, device=device)
                    sinkhorn_value = sinkhorn_calculator(generated_sample, ref_sample)[0].item()
                    logger.info(f"At step = {step}, sinkhorn_value = {sinkhorn_value}")
                    sinkhorn_values.append((step, sinkhorn_value))
                step += 1
    return TrainResult(ema_losses=ema_losses, samples=samples, sinkhorn_benchmark_value=sinkhorn_benchmark_value,
                       sinkhorn_values=sinkhorn_values)


def plot_metrics(result: TrainResult, noise_model_name: str, with_time_emb: bool, activation_name: str,
                 block_arch: str, normalize_output: bool, time_embedding_combination_method: str):
    # plot losses

    x = list(range(len(result.ema_losses)))
    plt.xlabel("Iterations")
    plt.ylabel("EMA loss - Log scale")
    plt.title("Loss-Iterations curve")
    plt.plot(x, np.log(result.ema_losses))

    if noise_model_name == "basic_discrete_time":
        outfile_name = (f"iter_loss_noise_model_{noise_model_name}_arch_{block_arch}_activation_{activation_name}_"
                        f"with_time_emb_{with_time_emb}_normalize_output_{normalize_output}_"
                        f"time_embedding_combination_{time_embedding_combination_method}.png")
    elif noise_model_name == "naive_nn":
        outfile_name = f"iter_loss_noise_model_{noise_model_name}.png"
    else:
        raise ValueError(f"Unknown noise_model_name : {noise_model_name}")
    plt.savefig(outfile_name)

    plt.clf()

    # plot sinkhorn
    x = [item[0] for item in result.sinkhorn_values]
    y1 = [result.sinkhorn_benchmark_value] * len(x)
    y2 = [item[1] for item in result.sinkhorn_values]
    max_log_y = np.maximum(max(np.log(y1)), max(np.log(y2)))
    plt.plot(x, np.log(y1), '--', label="Benchmark")
    plt.plot(x, np.log(y2), '-', label="Model values")
    plt.yticks(list(np.arange(-3.0, 1.0, 0.25)))
    plt.xlabel("Iterations")
    plt.ylabel("Sinkhorn Values - Log Scale")
    plt.legend(loc="upper right")
    plt.grid()

    if noise_model_name == "basic_discrete_time":
        outfile_name = (f"iter_sinkhorn_noise_model_{noise_model_name}_arch_{block_arch}_activation_"
                        f"{activation_name}_with_time_emb_{with_time_emb}_normalize_output_{normalize_output}_"
                        f"time_embedding_combination_{time_embedding_combination_method}.png")
    elif noise_model_name == "naive_nn":
        outfile_name = f"iter_sinkhorn_noise_model_{noise_model_name}.png"
    else:
        raise ValueError(f"Unknown noise_model_name : {noise_model_name}")
    plt.savefig(outfile_name)
    plt.clf()  # clearing figure buffer for any future plotting


def animate(samples: List[Any], noise_model_name: str, with_time_emb: bool, activation_name: str, block_arch: str,
            save: bool = True, normalize_output: bool = True, time_embedding_combination_method: str = "addition"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    first_sample = samples[0].detach().cpu().numpy().T
    scat = ax.scatter(*first_sample, c="k", alpha=0.3)

    def animate(i):
        offsets = samples[i].detach().cpu().numpy()
        scat.set_offsets(offsets)

    logger.info(f"Generating animation file...")
    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples) - 1)
    if save:
        if noise_model_name == "basic_discrete_time":
            outfile_name = (f"animation_noise_model_{noise_model_name}_arch_{block_arch}_activation_{activation_name}"
                            f"_with_time_emb_{str(with_time_emb)}_normalize_output_{normalize_output}_"
                            f"time_embedding_combination_{time_embedding_combination_method}.gif")
        elif noise_model_name == "naive_nn":
            outfile_name = f"animation_noise_model_{noise_model_name}_normalize_output_{normalize_output}.gif"
        else:
            raise ValueError(f"Unknown noise_model_name : {noise_model_name}")
        logger.info(f"Saving animation to file : {outfile_name}")
        anim.save(filename=outfile_name, writer=animation.PillowWriter(fps=5))
    plt.clf()
    return anim


def main(
        noise_model_name: str = "basic_discrete_time",
        normalize_output: bool = True,
        with_time_emb: bool = True,
        time_embedding_combination_method="addition",
        activation_name: str = "GELU",
        block_arch: str = "resnet",
        time_steps: int = 100,
        hidden_dim_model: int = 128,
        time_embedding_dim: int = 128,
        num_layers: int = 2,
        batch_size: int = 128,
        n_epochs: int = 1000,
        sample_size: int = 512,
        steps_between_sampling: int = 50,
        seed: int = 42
):
    # All defaults for params are based on the original ddpm code here
    # https://github.com/Jmkernes/Diffusion
    device = torch.device("cuda")
    logger.info("Creating model")
    if noise_model_name == "basic_discrete_time":
        noise_model = (BasicDiscreteTimeModel(model_dim=hidden_dim_model, num_resnet_layers=num_layers,
                                              with_time_emb=with_time_emb, activation_name=activation_name,
                                              block_arch=block_arch, normalize_output=normalize_output,
                                              time_embedding_dim=time_embedding_dim,
                                              time_embedding_combination_method=time_embedding_combination_method).
                       to(device))
    elif noise_model_name == "naive_nn":
        noise_model = NaiveNeuralNetworkNoiseModel(time_steps=time_steps).to(device)
    else:
        raise ValueError(f"Unsupported noise_model_name = {noise_model_name}")
    logger.info(f"Using noise_model = {noise_model_name}")
    logger.info(f"Type of noise_model instance : {type(noise_model)}")
    ddpm = DDPM(n_steps=time_steps).to(device)

    logger.info("Training")
    result = train(
        noise_model=noise_model,
        ddpm=ddpm,
        device=device,
        batch_size=batch_size,
        n_epochs=n_epochs,
        sample_size=sample_size,
        steps_between_sampling=steps_between_sampling,
        seed=seed,
    )
    animate(samples=result.samples, noise_model_name=noise_model_name, with_time_emb=with_time_emb,
            activation_name=activation_name, block_arch=block_arch, normalize_output=normalize_output,
            time_embedding_combination_method=time_embedding_combination_method)
    plot_metrics(result=result, noise_model_name=noise_model_name, with_time_emb=with_time_emb,
                 activation_name=activation_name, block_arch=block_arch, normalize_output=normalize_output,
                 time_embedding_combination_method=time_embedding_combination_method)


if __name__ == "__main__":
    Fire(main)
