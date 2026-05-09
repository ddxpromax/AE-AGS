# Sourced by run_*.sh: cap BLAS/OpenMP threading per subprocess so `--jobs`(ProcessPoolExecutor)
# does not oversubscribe the CPU (see README § Faster runs).
#
# Existing exports win only if POSIX default-assignment `: "${VAR:=val}"` is used --- actually
# `:=` assigns when unset OR empty; to allow empty string pass-through we'd need `[ -z ${VAR+x} ]`
# tricks. Empty rarely needed; omit for simplicity.

: "${OMP_NUM_THREADS:=1}"
export OMP_NUM_THREADS
: "${MKL_NUM_THREADS:=${OMP_NUM_THREADS}}"
export MKL_NUM_THREADS
: "${OPENBLAS_NUM_THREADS:=${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS
