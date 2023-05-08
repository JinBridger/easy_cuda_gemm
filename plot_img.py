# %%
import pandas as pd
from pathlib import Path

log_files = list(Path("./log").glob("*.log"))
log_frames = [(file.stem, pd.read_csv(file, header="infer")) for file in log_files]

df = pd.DataFrame(
    {"size": log_frames[0][1]["Size"]} | {name: df["GFLOPS"] for name, df in log_frames}
)

df.plot(x="size", ylabel="GFLOPS")

# %%
