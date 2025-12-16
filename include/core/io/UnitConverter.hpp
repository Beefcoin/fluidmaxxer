//needsoverhaul

 #pragma once
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "core/Types.hpp"


struct UnitConverter
{
    
    // input
    double Lx_phys;
    double U_phys;
    double nu_phys;
    double rho_phys;
    int Nx, Ny, Nz;

    // physDeltaX
    double dx_phys;

    // physDeltaT (timestep)
    double dt_phys;

    //lattice unit stuff
    double u_lbm;   //characteristic lattice velocity
    double nu_lbm;  //kinematic viscosity
    double rho_lbm; //lbm base density, fix to 1.9

    //relaxation params
    double tau_lbm;
    double omega_lbm;

    static constexpr double cs2_lbm = 1.0 / 3.0; // cs2

    UnitConverter(double Lx_phys_,
                  double U_phys_,
                  double nu_phys_,
                  double rho_phys_,
                  int Nx_,
                  int Ny_,
                  int Nz_ = 1,
                  double tau_lbm_target = 0.8)
        : Lx_phys(Lx_phys_),
          U_phys(U_phys_),
          nu_phys(nu_phys_),
          rho_phys(rho_phys_),
          Nx(Nx_),
          Ny(Ny_),
          Nz(Nz_)
    {
        if (Nx < 2)
            throw std::runtime_error("UnitConverter: Nx must be >= 2");
        if (U_phys <= 0.0)
            throw std::runtime_error("UnitConverter: U_phys must be > 0");
        if (nu_phys <= 0.0)
            throw std::runtime_error("UnitConverter: nu_phys must be > ");
        if (tau_lbm_target <= 0.5)
            throw std::runtime_error("UnitConverter: tau must be > 0.5");

        // calculate physical deltaX
        dx_phys = Lx_phys / static_cast<double>(Nx - 1);

        // set tau, calculate omega
        tau_lbm = tau_lbm_target;
        omega_lbm = 1.0 / tau_lbm;

        // kinematic visc. from omega
        nu_lbm = cs2_lbm * (tau_lbm - 0.5);

        // calculate timestep
        dt_phys = nu_lbm * dx_phys * dx_phys / nu_phys;

        // lattice velocity
        u_lbm = U_phys * dt_phys / dx_phys;

        // lattice density should be 1.0
        rho_lbm = 1.0;
    }


   void printInfo() const
{
    using std::cout;
    using std::endl;

    constexpr int W = 50;

    auto line = [&]()
    {
        cout << std::string(W, '-') << endl;
    };

    auto val = [&](const char *name, double v, const char *unit = "")
    {
        cout << " " << std::left << std::setw(10) << name
             << ": " << std::right << std::setw(12)
             << std::scientific << std::setprecision(5)
             << v;
        if (unit && unit[0])
            cout << "   " << unit;
        cout << endl;
    };

    cout << endl;
    line();
    cout << termcolor::green << termcolor::bold <<" UnitConverter" << termcolor::reset << endl;
    line();

    cout << termcolor::yellow << termcolor::bold <<"-- Physical input" << termcolor::reset << "\n";
    val("Lx", Lx_phys, "m");
    val("U", U_phys, "m/s");
    val("nu", nu_phys, "m^2/s");
    val("rho", rho_phys, "kg/m^3");

    cout << termcolor::yellow << termcolor::bold <<"-- Grid" << termcolor::reset << "\n";
    cout << " Nx, Ny, Nz: "
         << Nx << " x " << Ny << " x " << Nz << endl;

    cout << termcolor::yellow << termcolor::bold <<"-- LBM parameters" << termcolor::reset << "\n";
    val("tau", tau_lbm);
    val("omega", omega_lbm);
    val("cs^2", cs2_lbm);

    cout << termcolor::yellow << termcolor::bold <<"-- Derived" << termcolor::reset << "\n";
    val("dx", dx_phys, "m");
    val("dt", dt_phys, "s");
    val("u_lbm", u_lbm);
    val("nu_lbm", nu_lbm);
    val("rho_lbm", rho_lbm);

    //cout << endl;
    line();
}


    // convert physical velocity into lattice units
    double velocityPhysToLattice(double u_phys) const
    {
        return u_phys * dt_phys / dx_phys;
    }

    //convert lattice unit velocity into physical units
    double velocityLatticeToPhys(double u_lbm_val) const
    {
        return u_lbm_val * dx_phys / dt_phys;
    }

    //convert physical length into Lattice length
    double lengthPhysToLattice(double L_phys) const
    {
        return L_phys / dx_phys;
    }

    //convert lattice length to physical length
    double lengthLatticeToPhys(double L_lbm) const
    {
        return L_lbm * dx_phys;
    }

    //convert physical time to lattice time
    double timePhysToLattice(double t_phys) const
    {
        return t_phys / dt_phys;
    }

    //convert lattice time to physical time
    double timeLatticeToPhys(double t_lbm) const
    {
        return t_lbm * dt_phys;
    }

    //convert physical viscosity to lattice viscosity
    double viscosityPhysToLattice(double nu_phys_val) const
    {
        return nu_phys_val * dt_phys / (dx_phys * dx_phys);
    }

    //convert lattice viscosity to physical viscosity
    double viscosityLatticeToPhys(double nu_lbm_val) const
    {
        return nu_lbm_val * dx_phys * dx_phys / dt_phys;
    }

    //convert physical density to lattice density
    double densityPhysToLattice(double rho_phys_val) const
    {
        return rho_phys_val / rho_phys * rho_lbm;
    }

    //convert lattice density to physical density
    double densityLatticeToPhys(double rho_lbm_val) const
    {
        return rho_lbm_val / rho_lbm * rho_phys;
    }
};
