#!/usr/bin/env bash
# run_benchmarks.sh - every run we need for the report, in one go.
# PC project ODD-2026 | Rehan Mehta (241080039), Bhavya Gothi (241080021)
#
#   ./run_benchmarks.sh            30 m grid, 6 simulated hours (~15-20 min total on our laptops)
#   ./run_benchmarks.sh 15 24      15 m grid, full day (takes a while, run it overnight)
#
# Every parallel run is checked against the serial answer with check_same.py.
# Results pile up in results/timings.csv, then plot with scripts/plot_scaling.py

set -e
cd "$(dirname "$0")"

DX=${1:-30}
HOURS=${2:-6}
DOMAIN=data/domain_${DX}m.bin
NCORES=${NCORES:-$(nproc)}
MPIRUN=${MPIRUN:-mpirun}
# --bind-to none matters for the hybrid runs: by default OpenMPI pins each process
# to ONE core and then all its threads fight over that core
MPIFLAGS=${MPIFLAGS:---bind-to none}
PY=${PY:-python3}

make -s all bin/flood_novec
[ -f "$DOMAIN" ] || $PY scripts/make_mumbai_dem.py --dx "$DX" --no-preview

COMMON="--domain $DOMAIN --hours $HOURS --no-frames --quiet"
mkdir -p runs results

run() {   # run <tag> <command...>
    local tag=$1; shift
    echo ">> $tag"
    "$@" $COMMON --out "runs/$tag" --tag "$tag" | sed -n '2p;4p'
    if [ "$tag" != "serial" ] && [ "$tag" != "exp-novec" ]; then
        $PY scripts/check_same.py runs/serial/final_h.bin "runs/$tag/final_h.bin" | sed 's/^/   check: /'
    fi
}

# thread counts 1 2 4 8 ... up to the number of cores, plus the core count itself
TLIST=""
t=1
while [ $t -le "$NCORES" ]; do TLIST="$TLIST $t"; t=$((t * 2)); done
case " $TLIST " in *" $NCORES "*) ;; *) TLIST="$TLIST $NCORES" ;; esac
PMAX=$(( NCORES < 8 ? NCORES : 8 ))
PLIST=""
p=2
while [ $p -le $PMAX ]; do PLIST="$PLIST $p"; p=$((p * 2)); done
PEXP=$(( NCORES < 4 ? NCORES : 4 ))

echo "== $NCORES cores | grid ${DX} m | ${HOURS} h | threads:$TLIST | processes:$PLIST =="

run serial ./bin/flood_serial

for t in $TLIST; do
    OMP_NUM_THREADS=$t run omp-t$t ./bin/flood_omp
done

for p in $PLIST; do
    run mpi-p$p $MPIRUN -np $p $MPIFLAGS ./bin/flood_mpi --balance
done

for p in 2 4; do
    t=$(( NCORES / p ))
    [ $t -ge 1 ] && [ $p -le "$NCORES" ] || continue
    OMP_NUM_THREADS=$t run hybrid-p${p}t$t $MPIRUN -np $p $MPIFLAGS -x OMP_NUM_THREADS ./bin/flood_hybrid --balance
done

# ---- experiments ----
# 1. SIMD: same serial code with vectorisation off and libm cbrt
run exp-novec ./bin/flood_novec
# 2. load balancing off (equal rows per process)
run exp-equal-p$PEXP $MPIRUN -np $PEXP $MPIFLAGS ./bin/flood_mpi
# 3. blocking halo exchange instead of overlapping it with computation
run exp-blocking-p$PEXP $MPIRUN -np $PEXP $MPIFLAGS ./bin/flood_mpi --balance --no-overlap
# 4. OpenMP static scheduling instead of our default dynamic,4
OMP_NUM_THREADS=$NCORES OMP_SCHEDULE="static" run exp-static-t$NCORES ./bin/flood_omp

echo
$PY scripts/plot_scaling.py
