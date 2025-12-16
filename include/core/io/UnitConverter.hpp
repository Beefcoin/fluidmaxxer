// needsoverhaul

#pragma once
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "core/Types.hpp"

struct UnitConverter
{
    // ----------------------------
    // input (OpenLB-like)
    // ----------------------------
    double charPhysLength;    // [m] characteristic length (e.g. domain length in x)
    double charPhysVelocity;  // [m/s] characteristic velocity (e.g. inlet velocity)
    double physViscosity;     // [m^2/s]
    double physDensity;       // [kg/m^3]
    int Nx, Ny, Nz;           // grid size (for info only)
    int resolution;           // number of lattice intervals over charPhysLength

    // ----------------------------
    // derived discretization
    // ----------------------------
    double dx_phys;           // [m]
    double dt_phys;           // [s]

    // ----------------------------
    // lattice params
    // ----------------------------
    double tau_lbm;
    double omega_lbm;
    double nu_lbm;
    double u_lbm;
    double rho_lbm;

    static constexpr double cs2_lbm = 1.0 / 3.0;

    // OpenLB-like constructor: resolution + relaxation time
    UnitConverter(double charPhysLength_,
                  double charPhysVelocity_,
                  double physViscosity_,
                  double physDensity_,
                  int resolution_,
                  int Nx_,
                  int Ny_,
                  int Nz_ = 1,
                  double tau_lbm_target = 0.8)
        : charPhysLength(charPhysLength_),
          charPhysVelocity(charPhysVelocity_),
          physViscosity(physViscosity_),
          physDensity(physDensity_),
          Nx(Nx_), Ny(Ny_), Nz(Nz_),
          resolution(resolution_)
    {
        if (charPhysLength <= 0.0) throw std::runtime_error("UnitConverter: charPhysLength must be > 0");
        if (charPhysVelocity <= 0.0) throw std::runtime_error("UnitConverter: charPhysVelocity must be > 0");
        if (physViscosity <= 0.0) throw std::runtime_error("UnitConverter: physViscosity must be > 0");
        if (physDensity <= 0.0) throw std::runtime_error("UnitConverter: physDensity must be > 0");
        if (Nx < 2) throw std::runtime_error("UnitConverter: Nx must be >= 2");
        if (Ny < 1) throw std::runtime_error("UnitConverter: Ny must be >= 1");
        if (Nz < 1) throw std::runtime_error("UnitConverter: Nz must be >= 1");
        if (resolution <= 0) throw std::runtime_error("UnitConverter: resolution must be > 0");
        if (tau_lbm_target <= 0.5) throw std::runtime_error("UnitConverter: tau must be > 0.5");

        // OpenLB: dx = L / resolution
        dx_phys = charPhysLength / double(resolution);

        // prescribe tau
        tau_lbm   = tau_lbm_target;
        omega_lbm = 1.0 / tau_lbm;

        // nu_lbm = cs^2*(tau-0.5)
        nu_lbm = cs2_lbm * (tau_lbm - 0.5);

        // match physical viscosity:
        // nu_phys = nu_lbm * dx^2 / dt  => dt = nu_lbm * dx^2 / nu_phys
        dt_phys = nu_lbm * dx_phys * dx_phys / physViscosity;

        // characteristic lattice velocity
        u_lbm = charPhysVelocity * dt_phys / dx_phys;

        // reference lattice density
        rho_lbm = 1.0;
    }

    // ----------------------------
    // dimensionless numbers
    // ----------------------------
    double reynolds() const { return (charPhysVelocity * charPhysLength) / physViscosity; }
    double mach() const { return u_lbm / std::sqrt(cs2_lbm); }
    double knudsen() const
    {
        const double Re = reynolds();
        return (Re > 0.0) ? (mach() / Re) : 0.0;
    }

    // ----------------------------
    // conversions (same API style as before)
    // ----------------------------
    double velocityPhysToLattice(double u_phys) const { return u_phys * dt_phys / dx_phys; }
    double velocityLatticeToPhys(double u_lbm_val) const { return u_lbm_val * dx_phys / dt_phys; }

    double lengthPhysToLattice(double L_phys) const { return L_phys / dx_phys; }
    double lengthLatticeToPhys(double L_lbm_val) const { return L_lbm_val * dx_phys; }

    double timePhysToLattice(double t_phys) const { return t_phys / dt_phys; }
    double timeLatticeToPhys(double t_lbm_val) const { return t_lbm_val * dt_phys; }

    double viscosityPhysToLattice(double nu_phys_val) const { return nu_phys_val * dt_phys / (dx_phys * dx_phys); }
    double viscosityLatticeToPhys(double nu_lbm_val) const { return nu_lbm_val * dx_phys * dx_phys / dt_phys; }

    double densityPhysToLattice(double rho_phys_val) const { return rho_phys_val / physDensity * rho_lbm; }
    double densityLatticeToPhys(double rho_lbm_val) const { return rho_lbm_val / rho_lbm * physDensity; }

    // ----------------------------
    // pretty print
    // ----------------------------
    void printInfo() const
    {
        using std::cout;
        using std::endl;

        constexpr int W = 64;
        auto line = [&]() { cout << std::string(W, '-') << endl; };

        auto val = [&](const char* name, double v, const char* unit = "")
        {
            cout << " " << std::left << std::setw(18) << name
                 << ": " << std::right << std::setw(14)
                 << std::scientific << std::setprecision(6)
                 << v;
            if (unit && unit[0]) cout << "   " << unit;
            cout << endl;
        };

        cout << "\n";
        line();
        cout << termcolor::green << termcolor::bold
             << " UnitConverter (resolution + tau)"
             << termcolor::reset << endl;
        line();

        cout << termcolor::yellow << termcolor::bold << "-- Physical characteristic" << termcolor::reset << "\n";
        val("char L", charPhysLength, "m");
        val("char U", charPhysVelocity, "m/s");
        val("nu", physViscosity, "m^2/s");
        val("rho", physDensity, "kg/m^3");

        cout << termcolor::yellow << termcolor::bold << "-- Grid" << termcolor::reset << "\n";
        cout << " Nx, Ny, Nz           : " << Nx << " x " << Ny << " x " << Nz << "\n";
        cout << " resolution (intervals): " << resolution << "\n";

        cout << termcolor::yellow << termcolor::bold << "-- Discretization" << termcolor::reset << "\n";
        val("dx", dx_phys, "m");
        val("dt", dt_phys, "s");

        cout << termcolor::yellow << termcolor::bold << "-- LBM" << termcolor::reset << "\n";
        val("tau", tau_lbm);
        val("omega", omega_lbm);
        val("nu_lbm", nu_lbm);
        val("u_lbm", u_lbm);
        val("rho_lbm", rho_lbm);

        cout << termcolor::yellow << termcolor::bold << "-- Dimensionless" << termcolor::reset << "\n";
        val("Re", reynolds());
        val("Ma", mach());
        val("Kn (Ma/Re)", knudsen());

        cout << termcolor::yellow << termcolor::bold << "-- Checks" << termcolor::reset << "\n";
        if (tau_lbm < 0.55) {
            cout << " " << termcolor::red << termcolor::bold
                 << "WARNING: tau close to 0.5 -> can be unstable"
                 << termcolor::reset << "\n";
        }
        if (u_lbm > 0.15) {
            cout << " " << termcolor::red << termcolor::bold
                 << "WARNING: u_lbm high -> Mach might be too large"
                 << termcolor::reset << "\n";
        }
        if (u_lbm < 1e-8) {
            cout << " " << termcolor::red
                 << "NOTE: u_lbm extremely small (can be fine, but looks near-zero)"
                 << termcolor::reset << "\n";
        }

        line();
    }
};
