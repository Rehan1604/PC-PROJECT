// domain_io.hpp - reading the grid file and the rainfall csv
// PC project ODD-2026 | Rehan Mehta (241080039), Bhavya Gothi (241080021)
//
// Grid file layout (written by scripts/make_mumbai_dem.py or prepare_real_dem.py):
//   "MFLD" | int32 version(=1) | int32 nx | int32 ny | float64 dx
//   float64 z[ny][nx]        ground elevation (m above MSL)
//   float64 manning[ny][nx]  roughness
//   float64 infil[ny][nx]    infiltration loss in mm/h (green areas)
//   uint8   type[ny][nx]     0 = urban, 1 = green, 2 = sea
// Row 0 is the northern edge.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

enum : uint8_t { URBAN = 0, GREEN = 1, SEA = 2 };

struct DomainHeader {
    int nx = 0, ny = 0;
    double dx = 0;
};

static const long HEADER_BYTES = 24;

static bool read_header(const std::string& path, DomainHeader& hd)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    int32_t ver = 0, nx = 0, ny = 0;
    double dx = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && fread(&ver, 4, 1, f) == 1 &&
              fread(&nx, 4, 1, f) == 1 && fread(&ny, 4, 1, f) == 1 &&
              fread(&dx, 8, 1, f) == 1;
    fclose(f);
    if (!ok || memcmp(magic, "MFLD", 4) != 0 || ver != 1) return false;
    hd.nx = nx; hd.ny = ny; hd.dx = dx;
    return true;
}

// byte offset of array a (0=z, 1=manning, 2=infil, 3=type)
static long array_offset(const DomainHeader& hd, int a)
{
    long n = (long)hd.nx * hd.ny;
    return HEADER_BYTES + (long)a * n * 8;
}

// read rows [row0, row0+nrows) of one array straight into dst
template <class T>
static void read_rows(FILE* f, long offset, int nx, int row0, int nrows, T* dst)
{
    long off = offset + (long)row0 * nx * (long)sizeof(T);
    if (fseek(f, off, SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); exit(2); }
    size_t n = (size_t)nrows * nx;
    if (fread(dst, sizeof(T), n, f) != n) { fprintf(stderr, "domain file is too short\n"); exit(2); }
}

// rainfall csv: "hour,label,mm" or "hour,mm", lines starting with # are skipped.
// returns mm that fell in each hour of the run.
static std::vector<double> read_rain_csv(const std::string& path)
{
    std::vector<double> mm;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return mm;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        int hr; double v;
        if (sscanf(line, "%d,%*[^,],%lf", &hr, &v) == 2 || sscanf(line, "%d,%lf", &hr, &v) == 2) {
            if (hr < 0) continue;
            if ((int)mm.size() <= hr) mm.resize(hr + 1, 0.0);
            mm[hr] = v;
        }
    }
    fclose(f);
    return mm;
}

template <class T>
static bool write_raw(const std::string& path, const T* data, size_t n)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(data, sizeof(T), n, f);
    fclose(f);
    return w == n;
}
