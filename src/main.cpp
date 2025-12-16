// Descriptors
#include "core/descriptors/D3Q19.hpp"
#include "core/descriptors/D3Q7.hpp"

// Collision Operators
#include "collision/BGKCollision.hpp"
#include "collision/ADBGKCollision.hpp"
#include "collision/PorousBGKCollision.hpp"

// NS Boundary Conditions
#include "boundary/ns/VelocityDirichlet.hpp"
#include "boundary/ns/VelocityDirichletNEQ.hpp"
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

/* this calculates the equivalent particle diffusivity based on the particle porosity
it is based on the Mackie-Meares correlation (Mackie and Meares, 1955) */
DATA_TYPE computeEffectiveDiffusivity(DATA_TYPE Dm, DATA_TYPE epsParticle)
{
    DATA_TYPE numerator = epsParticle;
    DATA_TYPE denominator = std::pow(2.0 - epsParticle, 2.0);
    return (numerator / denominator) * Dm;
}

DATA_TYPE calculateInletVelocityFromPeclet(
    DATA_TYPE peclet,           // target Pe (dimensionless)
    DATA_TYPE diffusivity,      // D [m^2/s]
    DATA_TYPE particleDiameter, // particle diameter in meters
    DATA_TYPE porosity          // eps in (0,1]
)
{
    static_assert(std::is_floating_point<DATA_TYPE>::value, "T must be floating point");
    if (peclet <= DATA_TYPE(0) || diffusivity <= DATA_TYPE(0) || particleDiameter <= DATA_TYPE(0))
        throw std::invalid_argument("Pe, D, and Lc must be positive.");
    if (porosity <= DATA_TYPE(0) || porosity > DATA_TYPE(1))
        throw std::invalid_argument("Porosity eps must be in (0,1].");

    return (peclet * diffusivity / particleDiameter) * porosity; // u_s [m/s]
}

DATA_TYPE calculateInterstitialVelocity(
    DATA_TYPE superficialVelocity, // u_s [m/s]
    DATA_TYPE porosity             // eps in (0,1]
)
{
    static_assert(std::is_floating_point<DATA_TYPE>::value, "T must be floating point");
    if (superficialVelocity < DATA_TYPE(0))
        throw std::invalid_argument("Superficial velocity must be non-negative.");
    if (porosity <= DATA_TYPE(0) || porosity > DATA_TYPE(1))
        throw std::invalid_argument("Porosity eps must be in (0,1].");

    return superficialVelocity / porosity;
}

DATA_TYPE calculatekFilm(DATA_TYPE Re, DATA_TYPE kinematicViscosity, DATA_TYPE soluteDiffusivity, DATA_TYPE particlePorosity, DATA_TYPE particleDiameter)
{
    // calculate Schmidt Number
    fancy::kFilmTag() << std::setprecision(2) << "Re = " << Re << std::endl;
    DATA_TYPE Sc = kinematicViscosity / soluteDiffusivity;
    fancy::kFilmTag() << "Sc = " << Sc << std::endl;
    DATA_TYPE Sh = 1.09 * particlePorosity * pow(particlePorosity * Re, 0.33) * pow(Sc, 0.33);
    fancy::kFilmTag() << "Sh = " << Sh << std::endl;
    DATA_TYPE k_film = Sh * soluteDiffusivity / particleDiameter;
    return k_film;
}

DATA_TYPE calculatekLDF(DATA_TYPE geometryPorosity, DATA_TYPE particleDiameter, DATA_TYPE k_film, DATA_TYPE timestep)
{
    DATA_TYPE a_s = 3 * (1 - geometryPorosity) / (particleDiameter / 2);
    DATA_TYPE k_ldf = a_s * k_film;
    DATA_TYPE k_ldf_LU = k_ldf * timestep;
    fancy::configTag() << "Calculated k_ldf = " << k_ldf << " -> " << k_ldf_LU << " [1/timestep]" << std::endl;
    return k_ldf_LU;
}

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
    const bool writeVTI = hasFlag(argc, argv, "--vti");
    // const bool useZouHe = hasFlag(argc, argv, "--zouhe");

    fancy::mainTag() << "Starting Coupled LBM Simulation\n";
    fancy::mainTag() << "Coupling: ";
    fancy::onOff(std::cout, enableCoupling);
    std::cout << " | Inlet pulse: ";
    fancy::onOff(std::cout, enablePulse);
    std::cout << " | Adsorption: ";
    fancy::onOff(std::cout, enableAdsorb);
    std::cout << " | writeVTI: ";
    fancy::onOff(std::cout, writeVTI);
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
    const DATA_TYPE resolution = DATA_TYPE(15);
    const DATA_TYPE particleDiameter = DATA_TYPE(0.0002); // this is the charPhysLength
    const DATA_TYPE Peclet = DATA_TYPE(40);               // This is the Peclet used for calculating inlet velocity

    // NS-Settings
    const DATA_TYPE nu_phys = DATA_TYPE(1.0e-6);
    const DATA_TYPE rho_phys = DATA_TYPE(1000.0);

    // Time Params
    const DATA_TYPE maxPhysT = DATA_TYPE(80);
    const DATA_TYPE injectionTimePhys = DATA_TYPE(0.5);

    // Porosity Params
    const DATA_TYPE geometryPorosity = 0.26;
    const DATA_TYPE particlePorosity = 0.67;

    // Diffusion
    const DATA_TYPE D_fluid = DATA_TYPE(2e-09);
    const DATA_TYPE D_solid = computeEffectiveDiffusivity(D_fluid, particlePorosity);
    fancy::configTag() << "Calculated Deff = " << D_solid << " for ε = " << particlePorosity << std::endl;

    // calculate velocities
    const DATA_TYPE inletVelocity = calculateInletVelocityFromPeclet(Peclet, D_fluid, particleDiameter, geometryPorosity);
    const DATA_TYPE charPhysVelocity = calculateInterstitialVelocity(inletVelocity, geometryPorosity);
    fancy::configTag() << "Using Peclet number to calculate Inlet velocity. Inlet Velocity = " << inletVelocity << " for Pe = " << Peclet << " and particle diameter D = " << particleDiameter << std::endl;
    fancy::configTag() << "Calculated intersitial velocity from geometry porosity ε(geom) = " << geometryPorosity << " to vi = " << charPhysVelocity << std::endl;

    // Create UnitConverter and Debug
    UnitConverter uc(particleDiameter, charPhysVelocity, nu_phys, rho_phys, resolution, Nx, Ny, Nz, 0.75);
    uc.printInfo();
    const DATA_TYPE domainSize = Nx * uc.dx_phys;
    fancy::configTag() << std::fixed << std::setprecision(9) << "DomainSize = " << domainSize << std::endl;
    // Adsorption
    const DATA_TYPE qMax = DATA_TYPE(1.0);
    const DATA_TYPE KHenry = DATA_TYPE(2);
    const DATA_TYPE kA = calculatekFilm(uc.reynolds(), nu_phys, D_fluid, particlePorosity, particleDiameter);
    fancy::configTag() << std::setprecision(9) << "Calulcated Particle Mass Transfer Rate to k_film = " << kA << std::endl;

    // Calculate linear driving force rate in lattice units
    const DATA_TYPE k_ldf_LU = calculatekLDF(geometryPorosity, particleDiameter, kA, uc.dt_phys);

    // adsorption params (linear Isotherm)
    p.enableAdsorption = enableAdsorb;
    p.qMax = qMax;
    p.KHenry = KHenry;
    p.kA = k_ldf_LU;

    // set derived values from uc
    p.rho0 = static_cast<DATA_TYPE>(uc.rho_lbm);
    p.tau = static_cast<DATA_TYPE>(uc.tau_lbm);

    // --- Create Lattices ---
    Lattice<Descriptor_NS> Lattice_NS(p);
    ADLattice<Descriptor_AD> Lattice_AD(p);

    // set Collision Operators
    // auto Collision_NS = std::make_shared<BGKCollision<Descriptor_NS>>(p);
    auto Collision_NS = std::make_shared<PorousBGKCollision<Descriptor_NS>>(p);
    Lattice_NS.setCollisionOperator(Collision_NS);

    auto Collision_AD = std::make_shared<ADBGKCollision<Descriptor_AD>>(p);
    Lattice_AD.setCollisionOperator(Collision_AD);

    // --- Set Boundary Conditions
    // create masks for in/outlet
    auto inletMask = buildXPlaneMask(p.Nx, p.Ny, p.Nz, 0);
    auto outletMask = buildXPlaneMask(p.Nx, p.Ny, p.Nz, p.Nx - 1);

    // set NS-Lattice Obstacle Mask for BounceBack BC
    Lattice_NS.setObstacleMask(geom.mask);
    Lattice_NS.setPorosityFromMask(geom.mask, particlePorosity, DATA_TYPE(1.0));

    std::vector<DATA_TYPE> phiHost(p.Nx * p.Ny * p.Nz);
    cudaMemcpyDtoH(phiHost.data(), Lattice_NS.d_porosity_ptr(),
                   phiHost.size() * sizeof(DATA_TYPE));
    DATA_TYPE mn = phiHost[0], mx = phiHost[0];
    for (auto v : phiHost)
    {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    fancy::debugTag() << "Porosity min=" << mn << " max=" << mx << "\n";

    // set boundary conditions for NS-Lattice
    const DATA_TYPE u_in = static_cast<DATA_TYPE>(uc.velocityPhysToLattice(static_cast<double>(inletVelocity)));
    const DATA_TYPE rho_out = p.rho0;

    // Lattice_NS.addBoundary<VelocityDirichletEq<Descriptor_NS>>(p, inletMask, u_in, DATA_TYPE(0), DATA_TYPE(0));
    Lattice_NS.addBoundary<VelocityDirichletNEQ<Descriptor_NS>>(p, inletMask, u_in, DATA_TYPE(0), DATA_TYPE(0));

    Lattice_NS.addBoundary<PressureDirichlet<Descriptor_NS>>(p, outletMask, rho_out);

    // Lattice_NS.addBoundary<BounceBack<Descriptor_NS>>(p, geom.mask);

    // --- Diffusion & Adsorption Init ---
    // create AD-Unit Converter
    ADUC UnitConverter_AD(Nx, Ny, Nz, uc.dx_phys, uc.dt_phys);

    // Set tau / diffusivities for AD-Lattice
    Lattice_AD.setTauField(UnitConverter_AD.buildTauField(geom.mask, D_fluid, D_solid));

    // Create Adsorption fields
    AdsorbedField qField(p);
    LinearLDF<Descriptor_AD> LinearIsothermAdsorption(p);

    // set inlet/outlet conditions for AD-Lattice
    if (enablePulse)
    {
        const int iTmax = static_cast<int>(uc.timePhysToLattice(injectionTimePhys));

        const DATA_TYPE phi_base = DATA_TYPE(0.0);
        const DATA_TYPE phi_amp = DATA_TYPE(1.0);

        const int t_on = static_cast<int>(DATA_TYPE(0.10) * iTmax);
        const int t_off = static_cast<int>(DATA_TYPE(0.90) * iTmax);

        const DATA_TYPE k_steps = DATA_TYPE(0.05) * DATA_TYPE(iTmax);

        fancy::mainTag() << std::fixed << std::setprecision(0)
                         << "Pulse enabled, iTmax=" << iTmax
                         << " t_on=" << t_on << " t_off=" << t_off
                         << " k=" << k_steps << "\n";

        Lattice_AD.addBoundary<ADDirichletTanhPulse<Descriptor_AD>>(
            p, inletMask, phi_base, phi_amp, t_on, t_off, k_steps);
    }

    else
    {
        Lattice_AD.addBoundary<ADDirichlet<Descriptor_AD>>(p, inletMask, DATA_TYPE(1.0));
    }
    Lattice_AD.addBoundary<ADNeumannOutlet<Descriptor_AD>>(p, outletMask);

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
    const int steps = uc.timePhysToLattice(maxPhysT);
    const int outputEvery = 10000;
    const int csvEvery = 10000;
    fancy::mainTag() << "Time Settings: maxPhysT = " << std::fixed << std::setprecision(2) << maxPhysT << " steps = " << steps << " outputEvery = " << outputEvery << "\n";

    // write first step
    if (writeVTI)
    {
        w.writeStep(Lattice_NS, uc.timeLatticeToPhys(0), 0);
    }
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
            LinearIsothermAdsorption.computeSource(Lattice_AD, qField, Lattice_NS.d_obstacle_ptr());
            // LangmuirLDF.computeSource(Lattice_AD, qField, Lattice_NS.d_obstacle_ptr());
        }
        Lattice_AD.step(s);

        if (s % csvEvery == 0)
            outletCsv.log(s);

        if (s % outputEvery == 0)
        {
            const double pct = 100.0 * double(s) / double(steps);
            if (enableAdsorb)
            {
                qField.download();
            }
            fancy::timerTag()
                << "Step " << s << "/" << steps << " | physT = " << uc.timeLatticeToPhys(s) << " (" << fancy::yellow << fancy::bold << std::fixed << std::setprecision(2) << pct << "%" << fancy::reset << ")" << "\n";

            if (writeVTI)
            {
                w.writeStep(Lattice_NS, uc.timeLatticeToPhys(s), s);
            }
            outletCsv.flush();
        }
    }
}
