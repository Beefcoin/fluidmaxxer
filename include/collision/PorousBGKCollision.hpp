#pragma once
#include "core/interfaces/NSInterfaces.hpp"
#include "core/lattice/NSLattice.hpp"

// Porous BGK (MovingPorosity-like): equilibrium shift via uPlus = u + (1-phi)*(u_s - u)
template <typename Descriptor>
class PorousBGKCollision : public NSCollisionOperator<Descriptor> {
public:
    explicit PorousBGKCollision(const LBMParams& p) : params(p) {}

    void initEq(Lattice<Descriptor>& lat) override;
    void collideStream(Lattice<Descriptor>& lat) override;

private:
    LBMParams params{};
};

// explicit instantiation elsewhere
