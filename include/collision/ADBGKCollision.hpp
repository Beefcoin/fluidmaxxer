#pragma once
#include "core/Params.hpp"
#include "core/interfaces/ADInterfaces.hpp"
#include "core/lattice/ADLattice.hpp"

template <typename Descriptor>
class ADBGKCollision : public ADCollisionOperator<Descriptor> {
public:
    explicit ADBGKCollision(const LBMParams& p) : params(p) {}

    void initEq(ADLattice<Descriptor>& lat) override;
    void collideStream(ADLattice<Descriptor>& lat) override;

private:
    LBMParams params{};
};
