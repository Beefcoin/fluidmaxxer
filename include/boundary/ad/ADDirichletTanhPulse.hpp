#pragma once
#include <vector>
#include <cstdint>

#include "core/Params.hpp"
#include "core/Types.hpp"
#include "core/interfaces/ADInterfaces.hpp"
#include "core/lattice/ADLattice.hpp"

/* This is a constant scalar dirichlet AD BC, that uses a tanh fucntion to implement a "stable" inlet pulse */

template<typename Descriptor>
class ADDirichletTanhPulse : public ADBoundaryCondition<Descriptor> {
public:
    ADDirichletTanhPulse(const LBMParams& p,
                         std::vector<std::uint8_t> mask,
                         DATA_TYPE phi_base, //lower limit for tanh
                         DATA_TYPE phi_amp, //upper limit for tanh
                         int t_on_step, //onset time in steps
                         int t_off_step, //off time in steps
                         DATA_TYPE k_step); //slope in steps

    ~ADDirichletTanhPulse() override;

    void apply(ADLattice<Descriptor>& lat, BoundaryPhase phase, int step) override;

private:
    LBMParams params{};
    std::vector<std::uint8_t> mask_h;
    std::uint8_t* mask_d = nullptr;

    //Tanh params
    DATA_TYPE phi_base = DATA_TYPE(0);
    DATA_TYPE phi_amp  = DATA_TYPE(0);
    int t_on = 0;
    int t_off = 0;
    DATA_TYPE k = DATA_TYPE(1);
};
