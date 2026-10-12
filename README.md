# SyRe

> **Physical laws that hold at every bus, learned as readable equations and evaluated on the grid's
> current topology, detect manipulated telemetry and show where it occurs.**

Code for **Follow the Power: Symbolic and Readable Intrusion Detection for Reconfigurable Power Networks**.

![SyRe pipeline](docs/pipeline.png)

SyRe monitors the IEC 60870-5-104 telemetry that the control center receives from the grid (0, 1).
At every second, it builds a dynamic graph from the reported switch and connection states and computes,
for each bus and line end, the sums over its connected neighbours (2). Offline, symbolic regression
learns one law per target type from benign training data. A gate keeps only laws that are exact at
every bus, and the simplest of them is selected, which yields the power balances and the equality of
line-end and bus voltage (3). Online, every instance of every law is evaluated each second. Its residual
is normalized by the magnitude of the law's terms and a per-type scale. An alarm is raised when the
largest normalized residual exceeds a conformal threshold calibrated on training data, or when one of two
control-center rules fires: a state change without a command, or a missing periodic update (4). The
instance with the largest residual localizes the attack, as in the Drift Off attack on the loads at bus 14 (5).

## Results

On the [Sherlock](https://doi.org/10.5281/zenodo.18467070) dataset, counted as in the Sherlock paper:

| | Attacks detected | False alarms | Mean time to detect |
|---|---|---|---|
| 01-Basic | 15 of 18 | 7 | 50 s |
| 02-Semiurban | 25 of 29 | 3 | 52 s |

Every false alarm starts during a planned operator action (maintenance, switching, a transformer tap step)
and lasts 1 to 10 seconds.

## Try it

```bash
pip install -e .

mkdir -p data && cd data
curl -L -o 01-Basic.zip https://zenodo.org/api/records/18467070/files/01-Basic.zip/content
unzip -q 01-Basic.zip && cd ..

syre run configs/01-Basic.toml
```

It takes about half a minute. The output ends with a check against the paper:

```
same as the paper, second by second: laws alarm yes, command rule yes, update rule yes
```

For the larger grid, download `02-Semiurban.zip` (6.4 GB) and run `configs/02-Semiurban.toml`.
It needs about 4 minutes and 12 GB of RAM.

To see one law at work around one attack:

```bash
syre plot configs/01-Basic.toml --attack 23 --law "bus_P bus.14"
```

![Detection trace](docs/trace_attack23_bus14.png)

Someone drifts the load readings at bus 14 (pink). The power balance there breaks and crosses the
threshold within two minutes. During the attack on bus 10 earlier (yellow), the same law stays quiet.

## Learning the laws yourself

The laws in `configs/` are the ones from the paper. To learn them again from the training data:

```bash
pip install -e ".[fit]"
syre fit configs/01-Basic.toml
```

This runs PySR for up to 30 minutes per quantity type. Because the search stops on a time limit, a new run
can land on slightly different coefficients.

## What is where

```
configs/       one file per grid: laws, thresholds, settings
syre/data.py   telemetry, receive times, control-center log, attack windows
syre/grid.py   grid map from the dataset drawing, neighbour sums
syre/laws.py   residuals and the alarm threshold
syre/rules.py  the two control-center rules
syre/score.py  counting detections
syre/fit.py    PySR search and law selection
syre/plot.py   the trace figure
```

Everything is learned and calibrated on training data. Attack labels are used only to count detections.

Code: MIT license. Dataset: CC BY 4.0, Wagner et al.
