// Descriptors
#include "core/descriptors/D3Q19.hpp"
#include "core/descriptors/D3Q7.hpp"

// Collision Operators
#include "collision/BGKCollision.hpp"
#include "collision/ADBGKCollision.hpp"

// NS Boundary Conditions
#include "boundary/ns/VelocityDirichlet.hpp"
#include "boundary/ns/PressureDirichlet.hpp"
#include "boundary/ns/BounceBack.hpp"

// AD Boundary Conditions
#include "boundary/ad/ADDirichlet.hpp"
#include "boundary/ad/ADNeumannOutlet.hpp"
#include "boundary/ad/ADDirichletTanhPulse.hpp"

// Adsorption
#include "reaction/AdsorbedField.hpp"
#include "reaction/LinearLDF.hpp"
#include "reaction/LangmuirLDF.hpp"

// Unit Converters
#include "core/io/ADUnitConverter.hpp"
#include "core/io/UnitConverter.hpp"

// IO/FancyPrint
#include "core/io/FancyPrint.hpp"

// IO/VTI/CSV
#include "vti/VTIWriter.hpp"
#include "vti/VTIReader.hpp"
#include "core/io/OutletCSV.hpp"

// Helpers
#include "core/helpers/helpers.hpp"

int main(int argc, char **argv)
{
    // Definitions for Descriptors and Unit Converters
    using Descriptor_NS = D3Q19;
    using Descriptor_AD = D3Q7;
    using ADUC = ADUnitConverter<Descriptor_AD>;

    // get flags vom commandline params
    const bool enableCoupling = hasFlag(argc, argv, "--coupling");
    const bool enablePulse = hasFlag(argc, argv, "--pulse");
    const bool enableAdsorb = hasFlag(argc, argv, "--adsorb");

    fancy::mainTag() << "Starting Coupled LBM Simulation\n";
    fancy::mainTag() << "Coupling: ";
    fancy::onOff(std::cout, enableCoupling);
    std::cout << " | Inlet pulse: ";
    fancy::onOff(std::cout, enablePulse);
    std::cout << "\n";

    // Upload Descriptor Constants to GPU Device
    uploadD3Q19Constants();
    uploadD3Q7Constants();

    // --- Read Geometry ---
    VTIResult geom = readVTI("geometry.vti", 0.5);
    int Nx = geom.Nx;
    int Ny = geom.Ny;
    int Nz = geom.Nz;

    // --- Set Parameters ---
    LBMParams p{};
    // set dimensions from VTIReader Geometry
    p.Nx = Nx;
    p.Ny = Ny;
    p.Nz = Nz;
    // set periodicty
    p.periodicX = false;
    p.periodicY = true;
    p.periodicZ = true;

    // Simulation Params
    const DATA_TYPE Lx_phys = DATA_TYPE(0.0002);
    const DATA_TYPE U_phys = DATA_TYPE(0.000104);
    const DATA_TYPE nu_phys = DATA_TYPE(1.0e-6);
    const DATA_TYPE rho_phys = DATA_TYPE(1000.0);

    // Diffusion
    const DATA_TYPE D_fluid = DATA_TYPE(2e-09);
    const DATA_TYPE D_solid = DATA_TYPE(7.57533e-10);

    // Adsorption
    const DATA_TYPE qMax = DATA_TYPE(1.0);
    const DATA_TYPE KHenry = DATA_TYPE(2);
    const DATA_TYPE kA = DATA_TYPE(7.10943e-06);

    // adsorption params (linear Isotherm)
    p.enableAdsorption = enableAdsorb;
    p.qMax = qMax;
    p.bLangmuir = KHenry;
    p.kLDF = kA;
    p.KHenry = KHenry;
    p.kA = kA;

    // Create UnitConverter and Debug
    UnitConverter uc(Lx_phys, U_phys, nu_phys, rho_phys, Nx, Ny, Nz, 0.75);
    uc.printInfo();

    // set derived values from uc
    p.rho0 = static_cast<DATA_TYPE>(uc.rho_lbm);
    p.tau = static_cast<DATA_TYPE>(uc.tau_lbm);

    // --- Create Lattices ---
    Lattice<Descriptor_NS> Lattice_NS(p);
    ADLattice<Descriptor_AD> Lattice_AD(p);

    // set Collision Operators
    auto Collision_NS = std::make_shared<BGKCollision<Descriptor_NS>>(p);
    Lattice_NS.setCollisionOperator(Collision_NS);

    auto Collision_AD = std::make_shared<ADBGKCollision<Descriptor_AD>>(p);
    Lattice_AD.setCollisionOperator(Collision_AD);

    // --- Set Boundary Conditions
    // create masks for in/outlet
    auto inletMask = buildXPlaneMask(p.Nx, p.Ny, p.Nz, 0);
    auto outletMask = buildXPlaneMask(p.Nx, p.Ny, p.Nz, p.Nx - 1);

    // set NS-Lattice Obstacle Mask for BounceBack BC
    Lattice_NS.setObstacleMask(geom.mask);

    // set boundary conditions for NS-Lattice
    const DATA_TYPE u_in = static_cast<DATA_TYPE>(uc.velocityPhysToLattice(static_cast<double>(U_phys)));
    const DATA_TYPE rho_out = p.rho0;

    Lattice_NS.addBoundary<VelocityDirichletEq<Descriptor_NS>>(p, inletMask, u_in, DATA_TYPE(0), DATA_TYPE(0));
    Lattice_NS.addBoundary<PressureDirichlet<Descriptor_NS>>(p, outletMask, rho_out);
    Lattice_NS.addBoundary<BounceBack<Descriptor_NS>>(p, geom.mask);

    // set boundary conditions for AD-Lattice
    if (enablePulse)
    {
        const DATA_TYPE phi_base = DATA_TYPE(0.0);
        const DATA_TYPE phi_amp = DATA_TYPE(1.0);
        const int t_on = 0;
        const int t_off = 20000;
        const DATA_TYPE k_steps = DATA_TYPE(50.0);

        Lattice_AD.addBoundary<ADDirichletTanhPulse<Descriptor_AD>>(p, inletMask, phi_base, phi_amp, t_on, t_off, k_steps);
    }
    else
    {
        Lattice_AD.addBoundary<ADDirichlet<Descriptor_AD>>(p, inletMask, DATA_TYPE(1.0));
    }
    Lattice_AD.addBoundary<ADNeumannOutlet<Descriptor_AD>>(p, outletMask);

    // --- Diffusion & Adsorption Init ---
    // create AD-Unit Converter
    ADUC UnitConverter_AD(Nx, Ny, Nz, uc.dx_phys, uc.dt_phys);

    // Set tau / diffusivities for AD-Lattice
    Lattice_AD.setTauField(UnitConverter_AD.buildTauField(geom.mask, D_fluid, D_solid));

    // Create Adsorption fields
    AdsorbedField qField(p);
    //LinearLDF<Descriptor_AD> LinearIsothermAdsorption(p);
    LangmuirLDF<Descriptor_AD> LangmuirLDF(p);

    // --- Lattice Initialization ---
    Lattice_NS.initEquilibrium();
    Lattice_AD.initEquilibrium();

    // --- Initialize VTIWriter and CSV ---
    std::filesystem::create_directories("results");

    VTIWriterPhys<Descriptor_NS> w(uc);
    w.setOutputPattern("results/step_", 6);
    w.setPVDFile("results/out.pvd");

    w.registerRhoPhys(Lattice_NS);
    w.registerVelocityPhys(Lattice_NS);
    w.registerSpeedMagPhys(Lattice_NS);
    w.registerObstacle(Lattice_NS);
    w.registerConcentration(Lattice_AD, "concentration");
    w.registerScalarField("adfsorbed amount", qField.host());

    OutletCSV<ADLattice<Descriptor_AD>, UnitConverter> outletCsv(
        "results/outlet_concentration.csv",
        uc,
        outletMask,
        Lattice_AD);

    // --- Step definitions ---
    const int steps = 20000000;
    const int outputEvery = 10000;
    const int csvEvery = 10000;

    // write first step
    w.writeStep(Lattice_NS, uc.timeLatticeToPhys(0), 0);
    outletCsv.log(0);

    // --- Main Simulation Loop ---
    for (int s = 1; s <= steps; ++s)
    {
        Lattice_NS.step(s);
        if (enableCoupling)
        {
            Lattice_NS.computeMacroscopicDevice();
            Lattice_AD.setAdvectionFromFluidDevice(Lattice_NS);
        }

        if (enableAdsorb)
        {
            // reset source
            Lattice_AD.zeroSource();
            // compute source
            //LinearIsothermAdsorption.computeSource(Lattice_AD, qField, Lattice_NS.d_obstacle_ptr());
            LangmuirLDF.computeSource(Lattice_AD, qField, Lattice_NS.d_obstacle_ptr());
        }
        Lattice_AD.step(s);

        if (s % csvEvery == 0)
            outletCsv.log(s);

        if (s % outputEvery == 0)
        {
            if (enableAdsorb)
                qField.download();
            std::cout << "step " << s << "/" << steps << "\n";
            w.writeStep(Lattice_NS, uc.timeLatticeToPhys(s), s);
            outletCsv.flush();
        }
    }
}
