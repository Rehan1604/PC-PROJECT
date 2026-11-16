# Parallel Urban Flood Simulation for Mumbai

Parallel Computing project, ODD-2026, VJTI (Prof. Sandip Shingade)

| Name | Reg. no. |
|---|---|
| Rehan Mehta | 241080039 |
| Bhavya Gothi | 241080021 |

On 26 July 2005, 944 mm of rain fell on Santacruz in one day, at the same time as a high tide. This project replays that day on a 2D flood model of Mumbai and makes the model fast with OpenMP, MPI and SIMD. The solver is in C++ and the input prep and plots are in Python.

The model is the local inertial form of the shallow water equations (Bates, Horritt and Fewtrell, 2010), the same scheme LISFLOOD-FP uses. The city is split into horizontal strips, one per MPI process. Inside a strip, rows are shared out to OpenMP threads, and the inner loops are vectorised.

**Who did what**
- Rehan: the solver, the MPI decomposition and halo exchange, the benchmarking scripts
- Bhavya: the terrain and rainfall input, OpenMP scheduling and SIMD tuning, the animation and plots
- Both of us: debugging the instability (see "Things that went wrong"), the report

---

## Folder layout

```
src/flood.cpp              the solver, one file gives all four builds
src/params.hpp             command line options
src/domain_io.hpp          reading the grid file + rainfall csv
scripts/make_mumbai_dem.py builds the Mumbai-shaped terrain at any resolution
scripts/prepare_real_dem.py same thing from the real Copernicus DEM
scripts/animate.py         flood video + max depth map + depth at hotspots
scripts/plot_scaling.py    speedup / efficiency / breakdown plots + tables
scripts/check_same.py      checks a parallel run against the serial one
scripts/common.py          file readers shared by the scripts
data/rain_26jul2005.csv    hourly rainfall for the replay
run_benchmarks.sh          every run we need for the report
Makefile                   Linux / WSL build
build_windows.bat          native Windows build (MinGW + MS-MPI)
```

## Setup

We used WSL2 (Ubuntu) on Windows 11 because OpenMPI just works there.

```bash
sudo apt install build-essential libopenmpi-dev openmpi-bin python3-pip
pip install numpy scipy matplotlib imageio-ffmpeg
```

You can also build natively on Windows with `build_windows.bat`. It needs MSYS2 g++, plus MS-MPI for the MPI builds.

## Quick start

```bash
make                                          # bin/flood_serial, flood_omp, flood_mpi, flood_hybrid
python3 scripts/make_mumbai_dem.py --dx 30    # data/domain_30m.bin (600 x 1333 cells)
./bin/flood_omp --domain data/domain_30m.bin  # full 24 h, writes out/
python3 scripts/animate.py --run out          # results/flood_out.mp4 + max depth map
```

The run prints one line per simulated hour:

```
  14:30 IST | rain   190 mm/h | tide +1.57 m | flooded    3.34 km2 | deepest  3.45 m | step 3628
  15:30 IST | rain   140 mm/h | tide +1.80 m | flooded   36.52 km2 | deepest  3.68 m | step 4650
```

## Running each version

```bash
./bin/flood_serial  --domain data/domain_30m.bin
OMP_NUM_THREADS=8 ./bin/flood_omp --domain data/domain_30m.bin
mpirun -np 4 --bind-to none ./bin/flood_mpi --domain data/domain_30m.bin --balance
OMP_NUM_THREADS=4 mpirun -np 2 --bind-to none -x OMP_NUM_THREADS ./bin/flood_hybrid --domain data/domain_30m.bin --balance
```

Keep `--bind-to none` for the hybrid runs. By default OpenMPI pins each process to one core, so all four threads of a process end up sharing that single core.

| Option | Default | What it does |
|---|---|---|
| `--domain FILE` | data/domain_30m.bin | input grid |
| `--hours H` | 24 | simulated hours, starting 08:30 IST |
| `--every MIN` | 15 | snapshot interval for the video |
| `--balance` | off | split rows by work, not equally (MPI) |
| `--no-overlap` | off | blocking `MPI_Sendrecv` instead of `Isend/Irecv` overlapped with compute |
| `--no-frames` | off | skip snapshots, use for timing runs |
| `--drain MMHR` | 25 | storm drain capacity. Try 50 for the BRIMSTOWAD upgrade "what if" |
| `--tide M` | 1.8 | tide amplitude |
| `--tag NAME` | run | label in results/timings.csv |

## Benchmarks

```bash
./run_benchmarks.sh            # 30 m grid, 6 simulated hours
./run_benchmarks.sh 15 24      # 15 m grid, full day (overnight job)
```

The script runs serial, then OpenMP with 1, 2, 4 ... threads, then MPI with 2, 4, 8 processes, then the hybrid builds. After that come four experiments:

1. `exp-novec`: SIMD off (scalar `cbrt`, `-fno-tree-vectorize`)
2. `exp-equal`: MPI with equal row strips instead of balanced ones
3. `exp-blocking`: blocking halo exchange instead of overlapped
4. `exp-static`: OpenMP static schedule instead of `dynamic,4`

Every run is compared with the serial answer using `check_same.py`. All runs end with `results/plot_scaling.py`, which writes `results/speedup.png`, `efficiency.png`, `breakdown.png` and `summary.md`. The `summary.md` file holds the tables, including speedup, efficiency, the Karp-Flatt metric and an Amdahl fit.

### Two laptops

MPI can also run across both laptops. Both machines need the same build at the same path and passwordless ssh between them. Halo exchange happens every step, so use a LAN cable or a phone hotspot rather than busy college Wi-Fi.

```bash
# hosts.txt
192.168.1.10 slots=8
192.168.1.11 slots=8
mpirun -np 16 --hostfile hosts.txt --bind-to none ./bin/flood_mpi --balance --domain data/domain_15m.bin
```

## How the parallel part works

**Decomposition.** The grid is cut into horizontal strips of rows, one per MPI process (SPMD). Each process reads only its own rows plus one extra row on each side straight from the file using `fseek`, so nobody holds the whole city.

**Halo exchange.** Only the water depth `h` changes, so it is the only thing exchanged: one row up and one row down after every step. In the default mode, each process first updates its two boundary rows and posts `MPI_Isend` / `MPI_Irecv`. It then updates the interior while the messages are in flight and calls `MPI_Waitall` at the end. `--no-overlap` switches to plain `MPI_Sendrecv` for comparison.

**Time step.** The stable step size depends on the deepest water anywhere (CFL condition), so every step ends with an `MPI_Allreduce(MPI_MIN)` on dt. This is the one global synchronisation per step.

**Active spans.** Each row only loops from one cell before the first land cell to one cell after the last. Open sea on both sides is skipped completely, which matters a lot in the island city, where a row is about 2 km of land in an 18 km wide grid.

**Load balancing.** Because of the spans, a northern row (13 km of land) costs much more than a row near Colaba. With equal strips, the processes holding the south finish early and then wait. `--balance` gives every process the same total span width instead.

**OpenMP.** Each step uses two parallel loops over rows: fluxes, then depths. We use `schedule(dynamic,4)` by default for the same uneven-row reason. Diagnostics such as rain volume and max depth use `reduction`.

**SIMD.** The inner loops are `#pragma omp simd`. Check with `make vecreport`; all three hot loops show "loop vectorized using 32 byte vectors" (AVX2).

**Same answer every time.** Every cell does exactly the same arithmetic in every build. No reduction ever touches the state, and we compile with `-ffp-contract=off` so FMA can't change the rounding. So serial, OpenMP, MPI and hybrid give **bit-for-bit identical** water depths for any number of threads or processes, and `check_same.py` checks exactly that. The model also tracks the water budget (rain in, flow into the sea, water stored on land), and the error is around 1e-15.

## Things that went wrong (and what we changed)

These are going in the report, because most of what we learned came from them.

1. **The model blew up.** The first full-day run had 30 to 40 m of water in single cells and created 10 times more water than had fallen. Two causes: very fast flow on the steep national park slopes, and cells being drained by four faces at once to below zero depth, which we then clipped back to 0. That clipping quietly added water. Fix: a Froude limiter (no faster than critical flow) and a rule that one face can take at most a quarter of the upstream cell's water per step. After the fix, clipping is exactly 0.
2. **40% of the water disappeared.** The active spans first covered only cells near land *in the same row*. A sea cell sitting directly below a land cell in the next row was outside its row's span. Water flowed into it but was never counted, and its sea level was frozen at the starting tide. Fix: a row's span now also covers land in the rows above and below. The volume error went from -0.42 to about 1e-15.
3. **`std::cbrt` stopped the vectorisation.** The friction term needs h^(7/3), and the libm call meant gcc refused to vectorise the flux loops unless we used `-ffast-math`, which breaks reproducibility. We wrote `cbrt_simd`: a bit-trick first guess plus 3 Newton steps, accurate to about 1e-13. A `uint8` sea flag in the same loop also blocked vectorisation ("no vectype for stmt"), so the flag is now a `double` mask. On a 60 m, 8 hour test, the serial time went from about 15.7 s to 4.9 s.
4. **Static OpenMP scheduling was poor** because of the uneven rows. `dynamic,4` fixed it.

## Results

TODO: final numbers from both laptops. Paste `results/summary.md` here, and add speedup.png and the video link.

## Limitations

- The default terrain is generated: the outline was traced by hand, and the hills, Mithi river and flood pockets are placed approximately. That makes it easy to run the same city at 30, 15 or 10 m for the scaling study. For a real terrain run, use `prepare_real_dem.py` with the Copernicus GLO-30 tiles.
- The 944 mm total is IMD's Santacruz number, but the hour-by-hour split in `data/rain_26jul2005.csv` is our own reconstruction from news reports. Rain falls evenly on the whole city.
- Drains are a single capacity (25 mm/h, the old design figure for Mumbai's storm drains), subtracted from the rain on urban cells. There is no actual pipe network.
- The tide is a simple cosine with its peak in the afternoon, not the actual 2005 tide table.

## References

1. P. D. Bates, M. S. Horritt, T. J. Fewtrell, "A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling," *Journal of Hydrology* 387, 33-45, 2010. doi:10.1016/j.jhydrol.2010.03.027
2. J. Neal, T. Fewtrell, M. Trigg, "Parallelisation of storage cell flood models using OpenMP," *Environmental Modelling & Software* 24(7), 872-877, 2009. doi:10.1016/j.envsoft.2008.12.004
3. D. Yu, "Parallelization of a two-dimensional flood inundation model based on domain decomposition," *Environmental Modelling & Software* 25(8), 935-945, 2010.
4. A. R. Brodtkorb, M. L. Saetra, M. Altinakar, "Efficient shallow water simulations on GPUs: Implementation, visualization, verification, and validation," *Computers & Fluids* 55, 1-12, 2012.
