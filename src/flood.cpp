// flood.cpp - parallel urban flood simulation for Mumbai
// Parallel Computing project, ODD-2026 (VJTI)
// Rehan Mehta (241080039), Bhavya Gothi (241080021)
//
// Solves the local inertial shallow water equations (Bates, Horritt & Fewtrell 2010)
// on a regular grid. Same source gives all four versions, see Makefile:
//
//   flood_serial   plain C++
//   flood_omp      -fopenmp                (threads over rows)
//   flood_mpi      -DUSE_MPI               (row strips + halo exchange)
//   flood_hybrid   -DUSE_MPI -fopenmp      (both)
//
// Every version must give exactly the same final water depths as the serial one
// (scripts/check_same.py checks this). That is why we compile with
// -ffp-contract=off and never do reductions on the actual state.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <type_traits>

#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MAKE_DIR(p) mkdir(p, 0755)
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "params.hpp"
#include "domain_io.hpp"

static const double G = 9.81;
static const double PI = 3.14159265358979323846;

static int g_rank = 0;
static int g_size = 1;

// ---------------------------------------------------------------------------
// small wrappers so the serial / omp builds don't need MPI at all
// ---------------------------------------------------------------------------
static double wtime()
{
#ifdef USE_MPI
    return MPI_Wtime();
#elif defined(_OPENMP)
    return omp_get_wtime();
#else
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
#endif
}

static double all_min(double v)
{
#ifdef USE_MPI
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return r;
#else
    return v;
#endif
}

static double all_max(double v)
{
#ifdef USE_MPI
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return r;
#else
    return v;
#endif
}

static double all_sum(double v)
{
#ifdef USE_MPI
    double r;
    MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return r;
#else
    return v;
#endif
}

static void finish(int code)
{
#ifdef USE_MPI
    if (code != 0) MPI_Abort(MPI_COMM_WORLD, code);
    MPI_Finalize();
#endif
    exit(code);
}

static const char* build_mode()
{
#if defined(USE_MPI) && defined(_OPENMP)
    return "hybrid";
#elif defined(USE_MPI)
    return "mpi";
#elif defined(_OPENMP)
    return "omp";
#else
    return "serial";
#endif
}

static int n_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// mkdir -p
static void make_dirs(const std::string& path)
{
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/' || path[i] == '\\')
            MAKE_DIR(path.substr(0, i).c_str());
}

static void make_parent_dir(const std::string& path)
{
    size_t s = path.find_last_of("/\\");
    if (s != std::string::npos) make_dirs(path.substr(0, s));
}

// ---------------------------------------------------------------------------
// the part of the grid one MPI process owns
//
// local row k = 1..nyl are ours, k = 0 and k = nyl+1 are ghost rows that hold a
// copy of the neighbour's boundary row. qy[k] is the face between row k-1 and k.
// ---------------------------------------------------------------------------
struct Local {
    int nx = 0, ny = 0, nyl = 0, r0 = 0;
    double dx = 0;
    bool top_edge = false, bot_edge = false;

    std::vector<double>  z, h, n, infil, hmax;  // (nyl+2) * nx
    std::vector<uint8_t> type, sea;
    std::vector<double>  land;                  // same as !sea but as 1.0 / 0.0
    std::vector<double>  qx;                    // (nyl+2) * (nx+1)
    std::vector<double>  qy;                    // (nyl+2) * nx

    // In every row we only loop over [jlo, jhi], from one cell before the first
    // land cell to one after the last. Open sea on both sides is skipped completely.
    // This alone gave us ~35% on the island city rows.
    std::vector<int> jlo, jhi;   // per row
    std::vector<int> flo, fhi;   // per y-face (union of the two rows)
};

struct StepSums {
    double hmax_land = 0;  // deepest land cell after the step (for next dt)
    double rain = 0;       // m3 of rain added
    double to_sea = 0;     // m3 that flowed into sea cells
    double clamp = 0;      // m3 added by clipping negative depths to zero
};

static double tide_level(double t, const Params& p)
{
    // semi-diurnal tide, period 12.42 h, peak at tide_peak_h
    double hrs = t / 3600.0;
    return p.tide_amp * std::cos(2.0 * PI * (hrs - p.tide_peak_h) / 12.42);
}

// ---------------------------------------------------------------------------
// row decomposition
// ---------------------------------------------------------------------------
static std::vector<int> split_rows(const DomainHeader& hd, const Params& p)
{
    const int nx = hd.nx, ny = hd.ny;
    std::vector<int> start(g_size + 1, 0);

    if (!p.balance || g_size == 1) {
        for (int r = 0; r <= g_size; r++) start[r] = (int)((long)ny * r / g_size);
        return start;
    }

    // Weight of a row = width of its active span (that is what the loops run over)
    // plus a small fixed cost. Everybody computes the same thing so no comm needed.
    std::vector<uint8_t> type((size_t)nx * ny);
    FILE* f = fopen(p.domain.c_str(), "rb");
    read_rows<uint8_t>(f, array_offset(hd, 3), nx, 0, ny, type.data());
    fclose(f);

    std::vector<double> w(ny);
    double total = 0;
    for (int i = 0; i < ny; i++) {
        const uint8_t* t = &type[(size_t)i * nx];
        int lo = nx, hi = -1;
        for (int j = 0; j < nx; j++)
            if (t[j] != SEA) { if (lo == nx) lo = j; hi = j; }
        w[i] = (hi >= lo ? (hi - lo + 3) : 0) + 0.02 * nx;
        total += w[i];
    }

    double acc = 0;
    int r = 1;
    for (int i = 0; i < ny && r < g_size; i++) {
        acc += w[i];
        while (r < g_size && acc >= total * r / g_size) start[r++] = i + 1;
    }
    start[g_size] = ny;

    // every process needs at least 2 rows (the overlap code assumes it)
    for (int k = 1; k < g_size; k++) start[k] = std::max(start[k], start[k - 1] + 2);
    for (int k = g_size - 1; k >= 1; k--) start[k] = std::min(start[k], start[k + 1] - 2);
    return start;
}

static void load_local(Local& L, const DomainHeader& hd, const Params& p, int r0, int r1)
{
    L.nx = hd.nx; L.ny = hd.ny; L.dx = hd.dx;
    L.r0 = r0; L.nyl = r1 - r0;
    L.top_edge = (r0 == 0);
    L.bot_edge = (r1 == hd.ny);

    const int nx = L.nx;
    const size_t nc = (size_t)(L.nyl + 2) * nx;

    // ghost rows outside the map stay as tall dry walls (they are never used for fluxes anyway)
    L.z.assign(nc, 1000.0);
    L.h.assign(nc, 0.0);
    L.n.assign(nc, 0.035);
    L.infil.assign(nc, 0.0);
    L.hmax.assign(nc, 0.0);
    L.type.assign(nc, URBAN);
    L.sea.assign(nc, 0);
    L.land.assign(nc, 1.0);
    L.qx.assign((size_t)(L.nyl + 2) * (nx + 1), 0.0);
    L.qy.assign(nc, 0.0);

    // global rows we need = our rows plus one ghost row on each side if it exists
    const int g0 = std::max(r0 - 1, 0);
    const int g1 = std::min(r1 + 1, hd.ny);
    const int k0 = g0 - (r0 - 1);  // local row where g0 lands
    const size_t at = (size_t)k0 * nx;

    FILE* f = fopen(p.domain.c_str(), "rb");
    if (!f) { fprintf(stderr, "[rank %d] cannot open %s\n", g_rank, p.domain.c_str()); finish(1); }
    read_rows<double>(f, array_offset(hd, 0), nx, g0, g1 - g0, &L.z[at]);
    read_rows<double>(f, array_offset(hd, 1), nx, g0, g1 - g0, &L.n[at]);
    read_rows<double>(f, array_offset(hd, 2), nx, g0, g1 - g0, &L.infil[at]);
    read_rows<uint8_t>(f, array_offset(hd, 3), nx, g0, g1 - g0, &L.type[at]);
    fclose(f);

    for (size_t i = 0; i < nc; i++) {
        L.sea[i] = (L.type[i] == SEA);
        L.land[i] = L.sea[i] ? 0.0 : 1.0;
        L.infil[i] = L.infil[i] / 1000.0 / 3600.0;  // mm/h -> m/s
    }

    // active spans. First the land extent of every row (ghost rows included)...
    std::vector<int> llo(L.nyl + 2, nx), lhi(L.nyl + 2, -1);
    for (int k = 0; k <= L.nyl + 1; k++) {
        if ((k == 0 && L.top_edge) || (k == L.nyl + 1 && L.bot_edge)) continue;
        const uint8_t* s = &L.sea[(size_t)k * nx];
        for (int j = 0; j < nx; j++)
            if (!s[j]) { if (llo[k] == nx) llo[k] = j; lhi[k] = j; }
    }
    // ...then a row has to cover every cell that touches land, including sea cells
    // that only touch land in the row above or below (missed those at first and
    // lost 40% of the water through them, see notes in README)
    L.jlo.assign(L.nyl + 2, nx);
    L.jhi.assign(L.nyl + 2, -1);
    for (int k = 1; k <= L.nyl; k++) {
        int lo = std::min({llo[k] - 1, llo[k - 1], llo[k + 1]});
        int hi = std::max({lhi[k] + 1, lhi[k - 1], lhi[k + 1]});
        if (lhi[k] < llo[k]) { lo = std::min(llo[k - 1], llo[k + 1]); hi = std::max(lhi[k - 1], lhi[k + 1]); }
        L.jlo[k] = std::max(lo, 0);
        L.jhi[k] = std::min(hi, nx - 1);
    }
    // y faces: anything with land on either side
    L.flo.assign(L.nyl + 2, nx);
    L.fhi.assign(L.nyl + 2, -1);
    for (int k = 1; k <= L.nyl + 1; k++) {
        L.flo[k] = std::min(llo[k - 1], llo[k]);
        L.fhi[k] = std::max(lhi[k - 1], lhi[k]);
    }

    // sea starts at the tide level, land starts dry
    const double eta0 = tide_level(0.0, p);
    for (size_t i = 0; i < nc; i++)
        if (L.sea[i]) L.h[i] = std::max(eta0 - L.z[i], 0.0);
}

// ---------------------------------------------------------------------------
// physics
// ---------------------------------------------------------------------------

// cube root that the compiler can vectorise (std::cbrt is a libm call, which kills
// "omp simd" unless you use -ffast-math, and we can't because of reproducibility).
// Initial guess from the float exponent bits, then Newton steps in double.
// hs is always between 1 mm and a few metres here.
static inline double cbrt_simd(double x)
{
    float xf = (float)x;
    uint32_t i;
    std::memcpy(&i, &xf, 4);
    i = i / 3u + 709921077u;
    float yf;
    std::memcpy(&yf, &i, 4);
    double y = yf;
    y = y - (y * y * y - x) / (3.0 * y * y);
    y = y - (y * y * y - x) / (3.0 * y * y);
    y = y - (y * y * y - x) / (3.0 * y * y);
    return y;
}

// flow per unit width across one face (m^2/s), eq. 7 of Bates et al. 2010
// cell 1 is left/up, cell 2 is right/down, positive q goes from 1 to 2
static inline double face_flux(double q, double z1, double h1, double z2, double h2,
                               double nman, double land_sum,
                               double gdt, double inv_dx, double dxdt, double hmin)
{
    const double e1 = z1 + h1;
    const double e2 = z2 + h2;
    const double hf = std::max(e1, e2) - std::max(z1, z2);   // depth available to flow
    const double hs = std::max(hf, hmin);
#ifdef SCALAR_CBRT
    const double c3 = std::cbrt(hs);      // only for the "no SIMD" baseline build
#else
    const double c3 = cbrt_simd(hs);
#endif
    const double fric = gdt * nman * nman * std::fabs(q) / (hs * hs * c3);  // hs^(7/3)
    double qn = (q - gdt * hs * (e2 - e1) * inv_dx) / (1.0 + fric);
    // Two limiters, both found the hard way:
    //  1. Froude: no faster than critical flow. Without it the steep slopes in the
    //     national park blew up (20+ m of water piling up in single cells).
    //  2. a face may move at most a quarter of the upstream cell's water in one step,
    //     so 4 faces together can never drain a cell below zero. Before this we were
    //     clipping negative depths to 0 and that quietly created ~10x the rainfall
    //     volume out of nothing. Now the volume error is at round-off level.
    const double qcap = hs * std::sqrt(G * hs);
    qn = std::min(std::max(qn, -qcap), qcap);
    const double vcap = 0.25 * dxdt;
    qn = std::min(qn, h1 * vcap);
    qn = std::max(qn, -h2 * vcap);
    return (hf > hmin && land_sum > 0.5) ? qn : 0.0;   // sea-to-sea faces carry nothing
}

static void compute_fluxes(Local& L, double dt, const Params& p)
{
    const int nx = L.nx;
    const double inv_dx = 1.0 / L.dx;
    const double gdt = G * dt;
    const double dxdt = L.dx / dt;
    const double hmin = p.hmin;

    const double*  z = L.z.data();
    const double*  h = L.h.data();
    const double*  n = L.n.data();
    const double*  land = L.land.data();   // 1.0 land, 0.0 sea (double so the loop vectorises)
    double* qx = L.qx.data();
    double* qy = L.qy.data();

    // faces at the very top / bottom of the whole map are closed walls
    const int kb = L.top_edge ? 2 : 1;
    const int ke = L.bot_edge ? L.nyl : L.nyl + 1;

    #pragma omp parallel
    {
        // x faces: face j sits between cell j-1 and cell j
        #pragma omp for schedule(runtime) nowait
        for (int k = 1; k <= L.nyl; k++) {
            const size_t o = (size_t)k * nx;
            const double*  zr = z + o;
            const double*  hr = h + o;
            const double*  nr = n + o;
            const double*  lr = land + o;
            double* q = qx + (size_t)k * (nx + 1);
            const int lo = L.jlo[k] + 1, hi = L.jhi[k];
            #pragma omp simd
            for (int j = lo; j <= hi; j++)
                q[j] = face_flux(q[j], zr[j - 1], hr[j - 1], zr[j], hr[j],
                                 0.5 * (nr[j - 1] + nr[j]), lr[j - 1] + lr[j],
                                 gdt, inv_dx, dxdt, hmin);
        }

        // y faces: face k sits between row k-1 (up) and row k (down)
        #pragma omp for schedule(runtime)
        for (int k = kb; k <= ke; k++) {
            const size_t ou = (size_t)(k - 1) * nx, od = (size_t)k * nx;
            const double *zu = z + ou, *zd = z + od;
            const double *hu = h + ou, *hd = h + od;
            const double *nu = n + ou, *nd = n + od;
            const double *lu = land + ou, *ld = land + od;
            double* q = qy + od;
            const int lo = L.flo[k], hi = L.fhi[k];
            #pragma omp simd
            for (int j = lo; j <= hi; j++)
                q[j] = face_flux(q[j], zu[j], hu[j], zd[j], hd[j],
                                 0.5 * (nu[j] + nd[j]), lu[j] + ld[j],
                                 gdt, inv_dx, dxdt, hmin);
        }
    }
}

// mass balance for one row: new depth = old depth + dt * (net inflow + effective rain)
static void update_row(Local& L, int k, double dt, double rain_ms, double drain_ms,
                       double eta_sea, StepSums& out)
{
    const int nx = L.nx;
    const double inv_dx = 1.0 / L.dx;
    const double area = L.dx * L.dx;
    const size_t o = (size_t)k * nx;

    const double* qxr = L.qx.data() + (size_t)k * (nx + 1);
    const double* qyu = L.qy.data() + o;        // face above row k
    const double* qyd = L.qy.data() + o + nx;   // face below row k
    const double*  zr = L.z.data() + o;
    const double*  ir = L.infil.data() + o;
    const uint8_t* tr = L.type.data() + o;
    const uint8_t* sr = L.sea.data() + o;
    double* hr = L.h.data() + o;
    double* mr = L.hmax.data() + o;

    double hm = 0, rain = 0, sea_in = 0, clamp = 0;
    const int lo = L.jlo[k], hi = L.jhi[k];

    #pragma omp simd reduction(max:hm) reduction(+:rain, sea_in, clamp)
    for (int j = lo; j <= hi; j++) {
        const double div = (qxr[j] - qxr[j + 1] + qyu[j] - qyd[j]) * inv_dx;
        if (sr[j]) {
            // the sea is a fixed-level boundary, just remember what went in
            sea_in += div * dt * area;
            hr[j] = std::max(eta_sea - zr[j], 0.0);
        } else {
            const double loss = (tr[j] == URBAN) ? drain_ms : ir[j];
            const double r = std::max(rain_ms - loss, 0.0);
            double hn = hr[j] + dt * (div + r);
            rain += r * dt * area;
            if (hn < 0.0) { clamp -= hn * area; hn = 0.0; }
            hr[j] = hn;
            mr[j] = std::max(mr[j], hn);
            hm = std::max(hm, hn);
        }
    }
    out.hmax_land = std::max(out.hmax_land, hm);
    out.rain += rain;
    out.to_sea += sea_in;
    out.clamp += clamp;
}

static void update_rows(Local& L, int k0, int k1, double dt, double rain_ms, double drain_ms,
                        double eta_sea, StepSums& tot)
{
    double hm = 0, rain = 0, sea_in = 0, clamp = 0;
    #pragma omp parallel for schedule(runtime) reduction(max:hm) reduction(+:rain, sea_in, clamp)
    for (int k = k0; k <= k1; k++) {
        StepSums s;
        update_row(L, k, dt, rain_ms, drain_ms, eta_sea, s);
        hm = std::max(hm, s.hmax_land);
        rain += s.rain; sea_in += s.to_sea; clamp += s.clamp;
    }
    tot.hmax_land = std::max(tot.hmax_land, hm);
    tot.rain += rain; tot.to_sea += sea_in; tot.clamp += clamp;
}

// ---------------------------------------------------------------------------
// halo exchange (only the depth h changes, z / n / type ghosts are read from file)
// ---------------------------------------------------------------------------
#ifdef USE_MPI
static void halo_blocking(Local& L)
{
    const int up = g_rank > 0 ? g_rank - 1 : MPI_PROC_NULL;
    const int dn = g_rank < g_size - 1 ? g_rank + 1 : MPI_PROC_NULL;
    const int nx = L.nx, nyl = L.nyl;
    double* h = L.h.data();
    // first owned row goes up, ghost below comes from dn
    MPI_Sendrecv(h + (size_t)1 * nx, nx, MPI_DOUBLE, up, 0,
                 h + (size_t)(nyl + 1) * nx, nx, MPI_DOUBLE, dn, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // last owned row goes down, ghost above comes from up
    MPI_Sendrecv(h + (size_t)nyl * nx, nx, MPI_DOUBLE, dn, 1,
                 h, nx, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static void halo_start(Local& L, MPI_Request req[4])
{
    const int up = g_rank > 0 ? g_rank - 1 : MPI_PROC_NULL;
    const int dn = g_rank < g_size - 1 ? g_rank + 1 : MPI_PROC_NULL;
    const int nx = L.nx, nyl = L.nyl;
    double* h = L.h.data();
    MPI_Irecv(h, nx, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(h + (size_t)(nyl + 1) * nx, nx, MPI_DOUBLE, dn, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(h + (size_t)1 * nx, nx, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(h + (size_t)nyl * nx, nx, MPI_DOUBLE, dn, 1, MPI_COMM_WORLD, &req[3]);
}
#endif

// collect one field from everyone onto rank 0 (returns empty vector on other ranks)
template <class T>
static std::vector<T> gather_field(const Local& L, const std::vector<double>& src,
                                   const std::vector<int>& rows)
{
    std::vector<T> mine((size_t)L.nyl * L.nx);
    for (size_t i = 0; i < mine.size(); i++) mine[i] = (T)src[(size_t)L.nx + i];
#ifdef USE_MPI
    std::vector<int> counts(g_size), displs(g_size);
    for (int r = 0; r < g_size; r++) {
        counts[r] = (rows[r + 1] - rows[r]) * L.nx;
        displs[r] = rows[r] * L.nx;
    }
    std::vector<T> all;
    if (g_rank == 0) all.resize((size_t)L.ny * L.nx);
    MPI_Datatype t = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Gatherv(mine.data(), counts[g_rank], t, all.data(), counts.data(), displs.data(),
                t, 0, MPI_COMM_WORLD);
    return all;
#else
    (void)rows;
    return mine;
#endif
}

static void clock_label(double t, char* buf, size_t len)
{
    // run starts at 08:30 IST, which is when IMD's rain day starts
    int mins = (int)std::lround(8.5 * 60 + t / 60.0);
    snprintf(buf, len, "%02d:%02d", (mins / 60) % 24, mins % 60);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
#ifdef USE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
#endif
    const bool root = (g_rank == 0);
    Params p = parse_args(argc, argv, root);
#ifdef _OPENMP
    // rows are not equal work (the island city is 2 km wide, the suburbs 13 km),
    // so plain static scheduling left threads idle. dynamic,4 was the best we tried.
    // OMP_SCHEDULE still overrides this, e.g. OMP_SCHEDULE=static for the comparison.
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_dynamic, 4);
#endif

    DomainHeader hd;
    if (!read_header(p.domain, hd)) {
        if (root) fprintf(stderr, "could not read %s - run scripts/make_mumbai_dem.py first\n", p.domain.c_str());
        finish(1);
    }
    if (hd.ny < 2 * g_size) {
        if (root) fprintf(stderr, "grid has %d rows, too few for %d processes\n", hd.ny, g_size);
        finish(1);
    }
    std::vector<double> rain_mm = read_rain_csv(p.rain);
    if (rain_mm.empty() && root) fprintf(stderr, "warning: no rainfall read from %s, running dry\n", p.rain.c_str());

    std::vector<int> rows = split_rows(hd, p);
    Local L;
    load_local(L, hd, p, rows[g_rank], rows[g_rank + 1]);

    if (root) {
        make_dirs(p.outdir);
        make_parent_dir(p.results);
        if (!p.quiet) {
            printf("Mumbai flood sim | %s build | %d process(es) x %d thread(s)\n",
                   build_mode(), g_size, n_threads());
            printf("grid %d x %d cells, dx = %.1f m (%.0f km2) | %.1f h of rain from %s\n",
                   hd.nx, hd.ny, hd.dx, hd.nx * hd.ny * hd.dx * hd.dx / 1e6, p.hours, p.rain.c_str());
            if (g_size > 1)
                printf("row split: %s, halo: %s\n", p.balance ? "balanced" : "equal",
                       p.overlap ? "overlapped (Isend/Irecv)" : "blocking (Sendrecv)");
        }
    }

    const double T = p.hours * 3600.0;
    const double every = p.every_min * 60.0;
    const double area = hd.dx * hd.dx;
    const double drain_ms = p.drain_mmhr / 1000.0 / 3600.0;

    double t = 0.0, next_out = every;
    long steps = 0;
    int frame = 0;
    double hm_land = 0.0;
    double v_rain = 0, v_sea = 0, v_clamp = 0;
    double t_comp = 0, t_comm = 0, t_io = 0;

    FILE* flog = nullptr;
    if (root && p.frames) {
        flog = fopen((p.outdir + "/frames.csv").c_str(), "w");
        if (flog) fprintf(flog, "frame,t_sec,ist,rain_mmhr,tide_m,flooded_km2,max_depth_m\n");
    }
    auto snapshot = [&](void) {
        double t0 = wtime();
        // flooded area (> 30 cm on land) and deepest point, cheap enough to do every time
        double cnt = 0, mx = 0;
        for (int k = 1; k <= L.nyl; k++)
            for (int j = 0; j < L.nx; j++) {
                size_t i = (size_t)k * L.nx + j;
                if (L.sea[i]) continue;
                if (L.h[i] > 0.3) cnt += 1;
                mx = std::max(mx, L.h[i]);
            }
        double flooded = all_sum(cnt) * area / 1e6;
        double deepest = all_max(mx);
        int hr = (int)(t / 3600.0);
        double r_now = hr < (int)rain_mm.size() ? rain_mm[hr] : 0.0;
        char clk[8];
        clock_label(t, clk, sizeof clk);
        if (p.frames) {
            std::vector<float> f = gather_field<float>(L, L.h, rows);
            if (root) {
                char name[64];
                snprintf(name, sizeof name, "/frame_%04d.bin", frame);
                write_raw(p.outdir + name, f.data(), f.size());
                if (flog) fprintf(flog, "%d,%.1f,%s,%.1f,%.3f,%.4f,%.4f\n", frame, t, clk, r_now,
                                  tide_level(t, p), flooded, deepest);
            }
        }
        if (root && !p.quiet && std::fmod(t + 1.0, 3600.0) < 2.0)
            printf("  %s IST | rain %5.0f mm/h | tide %+5.2f m | flooded %7.2f km2 | deepest %5.2f m | step %ld\n",
                   clk, r_now, tide_level(t, p), flooded, deepest, steps);
        frame++;
        t_io += wtime() - t0;
    };

    snapshot();

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const double t_start = wtime();

    while (t < T - 1e-6) {
        // 1. time step: everyone takes the smallest stable dt
        double c0 = wtime();
        double dt_mine = p.alpha * L.dx / std::sqrt(G * std::max(hm_land, 0.01));
        double dt = all_min(dt_mine);
        t_comm += wtime() - c0;
        dt = std::min(dt, p.dt_max);
        dt = std::min(dt, next_out - t);
        dt = std::min(dt, T - t);

        const int hr = (int)(t / 3600.0);
        const double rain_ms = (hr < (int)rain_mm.size() ? rain_mm[hr] : 0.0) / 1000.0 / 3600.0;
        const double eta = tide_level(t + dt, p);

        // 2. fluxes on all faces
        double c1 = wtime();
        compute_fluxes(L, dt, p);

        // 3. depths, then share boundary rows
        StepSums s;
        if (g_size > 1 && p.overlap) {
#ifdef USE_MPI
            // boundary rows first so we can ship them, then do the interior while they travel
            update_row(L, 1, dt, rain_ms, drain_ms, eta, s);
            update_row(L, L.nyl, dt, rain_ms, drain_ms, eta, s);
            MPI_Request req[4];
            double c2 = wtime();
            halo_start(L, req);
            double c3 = wtime();
            update_rows(L, 2, L.nyl - 1, dt, rain_ms, drain_ms, eta, s);
            double c4 = wtime();
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            double c5 = wtime();
            t_comp += (c2 - c1) + (c4 - c3);
            t_comm += (c3 - c2) + (c5 - c4);
#endif
        } else {
            update_rows(L, 1, L.nyl, dt, rain_ms, drain_ms, eta, s);
            double c2 = wtime();
            t_comp += c2 - c1;
#ifdef USE_MPI
            if (g_size > 1) halo_blocking(L);
#endif
            t_comm += wtime() - c2;
        }

        hm_land = s.hmax_land;
        v_rain += s.rain; v_sea += s.to_sea; v_clamp += s.clamp;
        t += dt;
        steps++;

        if (t >= next_out - 1e-6) {
            snapshot();
            next_out += every;
        }
    }

    const double wall_mine = wtime() - t_start;
    if (flog) fclose(flog);

    // ---- results ----
    double v_land = 0;
    for (int k = 1; k <= L.nyl; k++)
        for (int j = 0; j < L.nx; j++) {
            size_t i = (size_t)k * L.nx + j;
            if (!L.sea[i]) v_land += L.h[i] * area;
        }
    double mx_mine = 0;
    for (int k = 1; k <= L.nyl; k++)
        for (int j = 0; j < L.nx; j++) {
            size_t i = (size_t)k * L.nx + j;
            if (!L.sea[i]) mx_mine = std::max(mx_mine, L.hmax[i]);
        }

    const double wall   = all_max(wall_mine);
    const double comp_a = all_sum(t_comp) / g_size;
    const double comp_m = all_max(t_comp);
    const double comm_m = all_max(t_comm);
    const double io_m   = all_max(t_io);
    const double vl = all_sum(v_land), vr = all_sum(v_rain), vs = all_sum(v_sea), vc = all_sum(v_clamp);
    const double deepest = all_max(mx_mine);
    const double vol_err = (vl - (vr - vs + vc)) / std::max(vr, 1.0);
    const double imbalance = comp_a > 0 ? comp_m / comp_a : 1.0;
    const double mcups = (double)hd.nx * hd.ny * steps / wall / 1e6;

    std::vector<double> final_h = gather_field<double>(L, L.h, rows);
    std::vector<float> hmax = gather_field<float>(L, L.hmax, rows);

    if (root) {
        write_raw(p.outdir + "/final_h.bin", final_h.data(), final_h.size());
        write_raw(p.outdir + "/hmax.bin", hmax.data(), hmax.size());
        FILE* fm = fopen((p.outdir + "/meta.txt").c_str(), "w");
        if (fm) {
            fprintf(fm, "nx %d\nny %d\ndx %.3f\nhours %.3f\nevery_min %.3f\nframes %d\ndomain %s\n",
                    hd.nx, hd.ny, hd.dx, p.hours, p.every_min, p.frames ? frame : 0, p.domain.c_str());
            fclose(fm);
        }

        printf("---- %s | %s | %d proc x %d thr ----\n", p.tag.c_str(), build_mode(), g_size, n_threads());
        printf("simulated %.1f h in %.2f s wall (%.1fx faster than real time)\n", p.hours, wall, T / wall);
        printf("steps %ld | %.1f Mcell-updates/s\n", steps, mcups);
        printf("compute %.2f s | comm %.2f s | output %.2f s | load imbalance %.3f\n",
               comp_m, comm_m, io_m, imbalance);
        printf("rain in %.4g m3 | to sea %.4g m3 | clipped %.3g m3 | volume error %.2e\n",
               vr, vs, vc, vol_err);
        printf("deepest water on land: %.2f m\n", deepest);

        bool fresh = true;
        if (FILE* chk = fopen(p.results.c_str(), "r")) { fresh = false; fclose(chk); }
        FILE* fr = fopen(p.results.c_str(), "a");
        if (fr) {
            if (fresh)
                fprintf(fr, "tag,mode,ranks,threads,balance,overlap,nx,ny,dx,sim_hours,steps,"
                            "wall_s,compute_s,comm_s,io_s,imbalance,mcups,vol_err,max_depth_m\n");
            fprintf(fr, "%s,%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%ld,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3e,%.4f\n",
                    p.tag.c_str(), build_mode(), g_size, n_threads(), (int)p.balance, (int)p.overlap,
                    hd.nx, hd.ny, hd.dx, p.hours, steps, wall, comp_m, comm_m, io_m, imbalance,
                    mcups, vol_err, deepest);
            fclose(fr);
        }
    }

    finish(0);
    return 0;
}
