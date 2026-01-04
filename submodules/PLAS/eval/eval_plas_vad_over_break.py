# Run PLAS with different improvement breaks and measure VAD
# Figure 8 in https://arxiv.org/abs/2312.13299 (Self-Organizing Gaussian Grids paper) created from this script
# Results are saved in CSV files

import torch
import pandas as pd
import time
from datetime import datetime
from random_grid import generate_random_colors
from plas import sort_with_plas, compute_vad


def vad_over_break(device, size=512):

    print(f"Running benchmark on {device}...")

    benchmark_start_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    params_np = generate_random_colors(size, size)

    # (channels, height, width)
    params = torch.from_numpy(params_np).permute(2, 0, 1).float().to(device)
    assert params.shape[1] == params.shape[2]

    df = pd.DataFrame(columns=["ib_log", "imp", "duration", "vad"])

    # warmup
    sorted_params, sorted_indices = sort_with_plas(
        params.clone(), improvement_break=1e-2, verbose=False
    )

    for steps in range(4, 15, 1):

        ib_log = -steps / 2

        imp = 10**ib_log

        local_params = params.clone()

        start_time = time.time()

        sorted_params, sorted_indices = sort_with_plas(
            local_params, improvement_break=imp, verbose=False
        )

        duration = time.time() - start_time

        assert (
            params.to(torch.int64).sum().item()
            == sorted_params.to(torch.int64).sum().item()
        )

        vad = compute_vad(sorted_params.permute(1, 2, 0).cpu().numpy())

        print(f"{ib_log=} {imp=} {duration=:.2f} {vad=:.4f}")

        df.loc[df.shape[0]] = [ib_log, imp, duration, vad]

    print(df)
    df.to_csv(f"{benchmark_start_time}_vad_over_break_{device}.csv", index=False)


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        vad_over_break(torch.device("mps"))

    if torch.cuda.is_available():
        vad_over_break(torch.device("cuda"))

    vad_over_break(torch.device("cpu"))
