// params.hpp - command line options for the flood solver
// PC project ODD-2026 | Rehan Mehta (241080039), Bhavya Gothi (241080021)
#pragma once
#include <string>
#include <cstdio>
#include <cstdlib>

struct Params {
    std::string domain  = "data/domain_30m.bin";
    std::string rain    = "data/rain_26jul2005.csv";
    std::string outdir  = "out";
    std::string results = "results/timings.csv";
    std::string tag     = "run";

    double hours       = 24.0;   // simulated time
    double every_min   = 15.0;   // snapshot interval (sim minutes)
    double alpha       = 0.7;    // CFL safety factor, Bates et al. use 0.7
    double dt_max      = 10.0;   // never take a step longer than this (s)
    double hmin        = 0.001;  // below 1 mm we treat a face as dry
    double tide_amp    = 1.8;    // m above/below mean sea level
    double tide_peak_h = 7.0;    // hours after start of run (start = 08:30 IST)
    double drain_mmhr  = 25.0;   // what the storm drains can take away in urban cells

    bool balance = false;  // split rows by amount of land instead of equally
    bool overlap = true;   // non-blocking halo exchange, overlapped with interior update
    bool frames  = true;   // write depth snapshots for the animation
    bool quiet   = false;
};

static void print_usage(const char* prog)
{
    printf("usage: %s [options]\n"
           "  --domain FILE     input grid (default data/domain_30m.bin)\n"
           "  --rain FILE       hourly rainfall csv (default data/rain_26jul2005.csv)\n"
           "  --out DIR         output folder (default out)\n"
           "  --results FILE    timing csv to append to (default results/timings.csv)\n"
           "  --tag NAME        label for this run in the csv\n"
           "  --hours H         simulated hours (default 24)\n"
           "  --every MIN       snapshot every MIN simulated minutes (default 15)\n"
           "  --alpha A         CFL factor (default 0.7)\n"
           "  --tide M          tide amplitude in m (default 1.8)\n"
           "  --drain MMHR      drain capacity in mm/h (default 25)\n"
           "  --balance         load balanced row split (MPI)\n"
           "  --no-overlap      blocking halo exchange (MPI)\n"
           "  --no-frames       don't write snapshots (for timing runs)\n"
           "  --quiet           less printing\n", prog);
}

static Params parse_args(int argc, char** argv, bool is_root)
{
    Params p;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                if (is_root) fprintf(stderr, "missing value after %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--domain")     p.domain = val();
        else if (a == "--rain")       p.rain = val();
        else if (a == "--out")        p.outdir = val();
        else if (a == "--results")    p.results = val();
        else if (a == "--tag")        p.tag = val();
        else if (a == "--hours")      p.hours = atof(val());
        else if (a == "--every")      p.every_min = atof(val());
        else if (a == "--alpha")      p.alpha = atof(val());
        else if (a == "--tide")       p.tide_amp = atof(val());
        else if (a == "--drain")      p.drain_mmhr = atof(val());
        else if (a == "--balance")    p.balance = true;
        else if (a == "--no-overlap") p.overlap = false;
        else if (a == "--no-frames")  p.frames = false;
        else if (a == "--quiet")      p.quiet = true;
        else if (a == "-h" || a == "--help") { if (is_root) print_usage(argv[0]); exit(0); }
        else {
            if (is_root) { fprintf(stderr, "unknown option: %s\n", a.c_str()); print_usage(argv[0]); }
            exit(1);
        }
    }
    return p;
}
