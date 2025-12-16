#pragma once
#include <vector>
#include <cstdint>

#include "core/Params.hpp"            
#include "core/Types.hpp"             
#include "core/interfaces/NSInterfaces.hpp" 
#include "core/lattice/NSLattice.hpp"         

/* this is a constant rho/pressure Dirichlet for use in NS-Lattices */

template<typename Descriptor>
class PressureDirichlet : public NSBoundaryCondition<Descriptor> {
public:
    PressureDirichlet(const LBMParams& p,
                      std::vector<std::uint8_t> mask,
                      DATA_TYPE rho_out);

    ~PressureDirichlet() override;

    void apply(Lattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    LBMParams params{};
    std::vector<std::uint8_t> mask_h;
    std::uint8_t* mask_d = nullptr;
    DATA_TYPE rho_out = DATA_TYPE(1);
};
