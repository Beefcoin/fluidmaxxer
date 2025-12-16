#pragma once
#include "core/Params.hpp"

/* this file includes template declarations for the NS CollisionOperators and Boundaryconditions
all NS Operators are build on these templates. */

template <typename Descriptor> class Lattice;

template <typename Descriptor>
class NSCollisionOperator {
public:
    virtual ~NSCollisionOperator() = default;
    virtual void initEq(Lattice<Descriptor>& lat) = 0;
    virtual void collideStream(Lattice<Descriptor>& lat) = 0; // f -> f_new
};

template <typename Descriptor>
class NSBoundaryCondition {
public:
    virtual ~NSBoundaryCondition() = default;
    virtual void apply(Lattice<Descriptor>& lat, BoundaryPhase phase, int step) = 0;
};
