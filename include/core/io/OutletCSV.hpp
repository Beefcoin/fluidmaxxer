#pragma once
#include <vector>
#include <cstdint>
#include <fstream>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "core/Types.hpp"

// logs outlet stats (avg/min/max) to csv
template<typename ADLat, typename UnitConv>
class OutletCSV
{
public:
    OutletCSV(
        const char* path,
        const UnitConv& uc,
        const std::vector<std::uint8_t>& outletMask,
        ADLat& adLat
    )
    : uc_(uc), outletMask_(outletMask), ad_(adLat), out_(path)
    {
        if (!out_)
            throw std::runtime_error("Could not open outlet csv");
        out_ << "step,time_s,phi_out_avg,phi_out_min,phi_out_max\n";
    }

    void log(int step)
    {
        // pull phi
        ad_.downloadPhi();
        const auto& phi = ad_.phiHost();

        double sum = 0.0;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        std::size_t cnt = 0;

        for (std::size_t i = 0; i < outletMask_.size(); ++i)
        {
            if (!outletMask_[i]) continue;
            const double v = static_cast<double>(phi[(int)i]);
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            ++cnt;
        }

        const double avg = (cnt > 0) ? sum / double(cnt) : 0.0;
        if (cnt == 0) { mn = 0.0; mx = 0.0; }

        out_ << step << ","
             << uc_.timeLatticeToPhys(step) << ","
             << avg << ","
             << mn << ","
             << mx << "\n";
    }

    void flush() { out_.flush(); }

private:
    const UnitConv& uc_;
    const std::vector<std::uint8_t>& outletMask_;
    ADLat& ad_;
    std::ofstream out_;
};
