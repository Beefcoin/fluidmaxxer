#pragma once
#include "core/interfaces/NSInterfaces.hpp"
#include "core/lattice/NSLattice.hpp"

template <typename Descriptor>
class BGKCollision : public NSCollisionOperator<Descriptor> {
public:
    explicit BGKCollision(const LBMParams& p) : params(p) {}
    void initEq(Lattice<Descriptor>& lat) override;
    void collideStream(Lattice<Descriptor>& lat) override;
private:
    LBMParams params{};
};
