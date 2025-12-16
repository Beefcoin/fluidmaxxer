#pragma once
#include "core/Params.hpp"

/* this file includes template declarations for the AD CollisionOperators and Boundaryconditions
all AD Operators are build on these templates. */

template <typename Descriptor> class ADLattice;

template <typename Descriptor>
class ADCollisionOperator {
public:
    virtual ~ADCollisionOperator() = default;
    virtual void initEq(ADLattice<Descriptor>& lat) = 0;
    virtual void collideStream(ADLattice<Descriptor>& lat) = 0; // g -> g_new
};

template <typename Descriptor>
class ADBoundaryCondition {
public:
    virtual ~ADBoundaryCondition() = default;
    virtual void apply(ADLattice<Descriptor>& lat, BoundaryPhase phase, int step) = 0;
};
