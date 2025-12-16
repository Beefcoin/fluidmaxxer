//needsoverhaul
#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <type_traits>

#include "core/lattice/NSLattice.hpp"
#include "core/Types.hpp"
#include "core/io/UnitConverter.hpp"
#include "core/lattice/ADLattice.hpp"

static constexpr auto VTIWRITER_COLOR = termcolor::green;

template <typename FluidDescriptor>
class VTIWriterPhys
{
public:
    struct Field
    {
        std::string name;
        int components = 1;
        std::function<void(std::ostream &, int /*cell*/)> writeCell;
    };

    explicit VTIWriterPhys(const UnitConverter &uc) : uc_(uc) {}

    void setOutputPattern(std::string prefix, int width = 6)
    {
        filePrefix_ = std::move(prefix);
        fileNumberWidth_ = width;
    }

    void setPVDFile(std::string pvdFilename)
    {
        pvdFile_ = std::move(pvdFilename);
        usePVD_ = true;
    }

    // --------------------------
    // Generic registration
    // --------------------------
    void addPreWriteHook(std::function<void()> hook)
    {
        preWriteHooks_.push_back(std::move(hook));
    }

    void registerScalar(const std::string &name,
                        std::function<DATA_TYPE(int)> getter)
    {
        Field f;
        f.name = name;
        f.components = 1;
        f.writeCell = [getter](std::ostream &out, int c)
        { out << getter(c) << " "; };
        fields_.push_back(std::move(f));
    }

    void registerScalarField(const std::string &name,
                             const std::vector<DATA_TYPE> &field)
    {
        if (field.empty())
            throw std::runtime_error("registerScalarField: empty field");
        registerScalar(name, [&field](int c) -> DATA_TYPE
                           { return field[(std::size_t)c]; });
    }

    void registerVector3(const std::string &name,
                         std::function<void(int, DATA_TYPE &, DATA_TYPE &, DATA_TYPE &)> getter)
    {
        Field f;
        f.name = name;
        f.components = 3;
        f.writeCell = [getter](std::ostream &out, int c)
        {
            DATA_TYPE x = DATA_TYPE(0), y = DATA_TYPE(0), z = DATA_TYPE(0);
            getter(c, x, y, z);
            out << x << " " << y << " " << z << " ";
        };
        fields_.push_back(std::move(f));
    }

    // --------------------------
    // Fluid helpers
    // --------------------------
    void registerRhoPhys(const Lattice<FluidDescriptor> &lat,
                         const std::string &name = "rho_phys")
    {
        // ensure macros are downloaded
        addPreWriteHook([&lat]()
                        { const_cast<Lattice<FluidDescriptor> &>(lat).downloadMacroscopic(); });

        registerScalar(name, [&lat, this](int c) -> DATA_TYPE
                       { return static_cast<DATA_TYPE>(
                             uc_.densityLatticeToPhys(static_cast<double>(lat.rhoHost()[c]))); });
    }

    void registerVelocityPhys(const Lattice<FluidDescriptor> &lat,
                              const std::string &name = "velocity_phys")
    {
        addPreWriteHook([&lat]()
                        { const_cast<Lattice<FluidDescriptor> &>(lat).downloadMacroscopic(); });

        registerVector3(name, [&lat, this](int c, DATA_TYPE &x, DATA_TYPE &y, DATA_TYPE &z)
                        {
            x = static_cast<DATA_TYPE>(uc_.velocityLatticeToPhys(static_cast<double>(lat.uxHost()[c])));
            y = static_cast<DATA_TYPE>(uc_.velocityLatticeToPhys(static_cast<double>(lat.uyHost()[c])));
            z = static_cast<DATA_TYPE>(uc_.velocityLatticeToPhys(static_cast<double>(lat.uzHost()[c]))); });
    }

    void registerSpeedMagPhys(const Lattice<FluidDescriptor> &lat,
                              const std::string &name = "u_mag_phys")
    {
        addPreWriteHook([&lat]()
                        { const_cast<Lattice<FluidDescriptor> &>(lat).downloadMacroscopic(); });

        registerScalar(name, [&lat, this](int c) -> DATA_TYPE
                       {
            const double ux = uc_.velocityLatticeToPhys(static_cast<double>(lat.uxHost()[c]));
            const double uy = uc_.velocityLatticeToPhys(static_cast<double>(lat.uyHost()[c]));
            const double uz = uc_.velocityLatticeToPhys(static_cast<double>(lat.uzHost()[c]));
            return static_cast<DATA_TYPE>(std::sqrt(ux*ux + uy*uy + uz*uz)); });
    }

    void registerObstacle(const Lattice<FluidDescriptor> &lat,
                          const std::string &name = "obstacle")
    {
        addPreWriteHook([&lat]()
                        { const_cast<Lattice<FluidDescriptor> &>(lat).downloadObstacle(); });

        registerScalar(name, [&lat](int c) -> DATA_TYPE
                       { return static_cast<DATA_TYPE>(static_cast<int>(lat.obstacleHost()[c])); });
    }

    // --------------------------
    // AD helpers (concentration)
    // --------------------------
    template <typename ADDescriptor>
    void registerConcentration(const ADLattice<ADDescriptor> &ad,
                               const std::string &name = "concentration")
    {
        // ensure phi is downloaded
        addPreWriteHook([&ad]()
                        { const_cast<ADLattice<ADDescriptor> &>(ad).downloadPhi(); });

        registerScalar(name, [&ad](int c) -> DATA_TYPE
                       { return ad.phiHost()[c]; });
    }

    // --------------------------
    // Write
    // --------------------------
    void writeStep(const Lattice<FluidDescriptor> &lat,
                   double physTime,
                   int stepIndex)
    {
        // ensure all registered fields have fresh host buffers
        for (auto &hook : preWriteHooks_)
            hook();

        const std::string vti = makeStepFilename(stepIndex);

        try
        {
            std::filesystem::path p(vti);
            if (p.has_parent_path())
                std::filesystem::create_directories(p.parent_path());

            if (usePVD_ && !pvdFile_.empty())
            {
                std::filesystem::path pp(pvdFile_);
                if (pp.has_parent_path())
                    std::filesystem::create_directories(pp.parent_path());
            }
        }
        catch (...)
        {
        }

        std::ofstream out(vti);
        if (!out)
        {
            std::cerr << VTIWRITER_COLOR << termcolor::bold
                      << "[VTIWriter]: " << termcolor::reset
                      << "Could not open " << vti << "\n";
            return;
        }

        if constexpr (std::is_same_v<DATA_TYPE, float>)
            out << std::setprecision(7) << std::fixed;
        else
            out << std::setprecision(15) << std::fixed;

        writeHeader(out, lat);
        writePointData(out, lat);
        writeFooter(out);
        out.close();

        if (usePVD_)
            appendToPVD(pvdFile_, vti, physTime);

        std::cout << VTIWRITER_COLOR << termcolor::bold
                  << "[VTIWriter]: " << termcolor::reset
                  << "Wrote VTI: " << vti << "\n";
    }

private:
    static constexpr const char *vtkDataType()
    {
        return std::is_same_v<DATA_TYPE, float> ? "Float32" : "Float64";
    }

    void writeHeader(std::ostream &out, const Lattice<FluidDescriptor> &lat)
    {
        const auto &p = lat.P();
        const int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;

        const double sx = uc_.dx_phys;
        const double sy = uc_.dx_phys;
        const double sz = (Nz > 1 ? uc_.dx_phys : 1.0);

        out << "<?xml version=\"1.0\"?>\n";
        out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "  <ImageData WholeExtent=\"0 " << (Nx - 1)
            << " 0 " << (Ny - 1)
            << " 0 " << (Nz - 1)
            << "\" Origin=\"0 0 0\" Spacing=\""
            << sx << " " << sy << " " << sz << "\">\n";
        out << "    <Piece Extent=\"0 " << (Nx - 1)
            << " 0 " << (Ny - 1)
            << " 0 " << (Nz - 1) << "\">\n";
    }

    void writePointData(std::ostream &out, const Lattice<FluidDescriptor> &lat)
    {
        const auto &p = lat.P();
        const int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;

        out << "      <PointData";
        if (!fields_.empty())
            out << " Scalars=\"" << fields_.front().name << "\"";
        out << ">\n";

        for (const auto &f : fields_)
        {
            out << "        <DataArray type=\"" << vtkDataType()
                << "\" Name=\"" << f.name << "\"";
            if (f.components > 1)
                out << " NumberOfComponents=\"" << f.components << "\"";
            out << " format=\"ascii\">\n          ";

            for (int z = 0; z < Nz; ++z)
                for (int y = 0; y < Ny; ++y)
                    for (int x = 0; x < Nx; ++x)
                        f.writeCell(out, lat.cellIndex(x, y, z));

            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n";
        out << "      <CellData>\n      </CellData>\n";
        out << "    </Piece>\n  </ImageData>\n";
    }

    void writeFooter(std::ostream &out) { out << "</VTKFile>\n"; }

    static void appendToPVD(const std::string &pvdFilename,
                            const std::string &vtiFilename,
                            double time)
    {
        namespace fs = std::filesystem;
        fs::path vtiPath(vtiFilename);
        const std::string fileInPvd = vtiPath.filename().string();

        std::ifstream in(pvdFilename);
        if (!in)
        {
            std::ofstream out(pvdFilename);
            if (!out)
                return;
            out << "<?xml version=\"1.0\"?>\n"
                << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                << "  <Collection>\n"
                << "    <DataSet timestep=\"" << time
                << "\" group=\"\" part=\"0\" file=\"" << fileInPvd << "\"/>\n"
                << "  </Collection>\n"
                << "</VTKFile>\n";
            return;
        }

        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string content = buffer.str();
        const std::string tag = "</Collection>";
        auto pos = content.rfind(tag);
        if (pos == std::string::npos)
            return;

        std::ostringstream ds;
        ds << "    <DataSet timestep=\"" << time
           << "\" group=\"\" part=\"0\" file=\"" << fileInPvd << "\"/>\n";
        content.insert(pos, ds.str());

        std::ofstream out(pvdFilename);
        if (!out)
            return;
        out << content;
    }

    std::string makeStepFilename(int stepIndex) const
    {
        std::ostringstream ss;
        ss << filePrefix_ << std::setw(fileNumberWidth_) << std::setfill('0')
           << stepIndex << ".vti";
        return ss.str();
    }

private:
    const UnitConverter &uc_;
    std::vector<Field> fields_;
    std::vector<std::function<void()>> preWriteHooks_;

    bool usePVD_ = false;
    std::string pvdFile_;

    std::string filePrefix_ = "out_";
    int fileNumberWidth_ = 6;
};
