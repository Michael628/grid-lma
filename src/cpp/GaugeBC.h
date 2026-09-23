#ifndef FMGRID_GAUGEBC_H
#define FMGRID_GAUGEBC_H

// Boundary-condition helpers for gauge fields.
//
// applyAPBC implements the same transformation as HadronsMILC's APBCGauge
// module (src/Modules/MGauge/APBCGauge.hpp): for every direction mu with
// boundary[mu] != 1, the mu-links on the last slice of that direction are
// multiplied by boundary[mu] (e.g. -1). Any covariant hop crossing the
// boundary then transports the phase, so spin-taste one-link operators
// built on the modified field (StagGamma::setGaugeField) become
// anti-periodic in that direction.
//
// Usage: grid_lma applies this to the thin gauge field used for non-local
// spin-taste covariant shifts. The fat/long links consumed by the Dirac
// operator already carry APBC from the MILC-generated files and must NOT be
// re-modified. The transformation is in-place on its argument; callers
// work on a copy (U_apbc = U; applyAPBC(U_apbc);).

#include <Grid/Grid.h>
#include <vector>

NAMESPACE_BEGIN(Grid)

// In-place last-slice boundary-phase multiply.
// boundary[mu] == 1  -> direction untouched (periodic).
// boundary[mu] == -1 -> APBC: U_mu(x) *= -1 for x_mu = GlobalDim[mu] - 1.
// The phase is applied in double precision regardless of the field's
// precision (upstream Grid fixed double-vs-float boundary-phase bugs
// twice: commits 9d9692d4, 751fae9f). The last-slice condition uses
// GLOBAL dimensions, so the mask is MPI-correct.
inline void applyAPBC(LatticeGaugeFieldD &U,
                      const std::vector<int> &boundary) {
  GRID_ASSERT(boundary.size() == static_cast<size_t>(Nd));
  Coordinate latt = U.Grid()->GlobalDimensions();
  for (int mu = 0; mu < Nd; mu++) {
    if (boundary[mu] == 1) continue;
    Lattice<iScalar<vInteger>> coord(U.Grid());
    LatticeCoordinate(coord, mu);
    LatticeColourMatrixD link(U.Grid());
    link = PeekIndex<LorentzIndex>(U, mu);
    int dimSize = latt[mu] - 1;
    link = where(coord == dimSize, static_cast<double>(boundary[mu]) * link,
                 link);
    PokeIndex<LorentzIndex>(U, link, mu);
  }
}

// Convenience overload: APBC in time only (boundary = {1,1,1,-1} in Grid's
// XYZT ordering), matching HadronsMILC's APBCGauge default.
inline void applyAPBC(LatticeGaugeFieldD &U) {
  std::vector<int> boundary(Nd, 1);
  boundary[Nd - 1] = -1; // time is the last direction (Tp == Nd-1)
  applyAPBC(U, boundary);
}

NAMESPACE_END(Grid)

#endif // FMGRID_GAUGEBC_H
