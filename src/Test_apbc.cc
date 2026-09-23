// Self-asserting test for the APBC gauge-field helper (src/cpp/GaugeBC.h)
// and its effect on non-local spin-taste covariant shifts.
//
// Checks on a 4^4 unit (cold) gauge field, run on a SINGLE RANK (scalar
// build) because the site-level peeks assert local ownership:
//   1. Flip selectivity: applyAPBC flips ONLY the T-links on the t = Lt-1
//      slice (per-site peeks + global norm2 arithmetic).
//   2. PBC no-op: boundary {1,1,1,1} leaves the field bit-identical.
//   3. Wrap-sign physics: a StagGamma one-link op with a T shift (G1_GT)
//      on a delta source at the origin transports amplitude through the
//      temporal wrap with a -1 sign under APBC vs PBC; every non-wrapped
//      timeslice is unchanged.
//
// Build (from build-scalar/):
//   make test_apbc
// Run:
//   ./test_apbc --grid 4.4.4.4

#include <Grid/Grid.h>
#include <GridMilc/spin/StagGamma.h>
#include <GaugeBC.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace Grid;

static int nFail = 0;

static void check(bool ok, const std::string &what) {
  std::cout << (ok ? "PASS: " : "FAIL: ") << what << std::endl;
  if (!ok) {
    nFail++;
  }
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  auto nsimd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimd, GridDefaultMpi());

  Coordinate latt = UGrid->GlobalDimensions();
  int Lt = latt[Tp];

  // Unit (cold) gauge field: every link = identity.
  LatticeGaugeFieldD U(UGrid);
  SU<Nc>::ColdConfiguration(U);
  LatticeGaugeFieldD U_ref(UGrid);
  U_ref = U;

  // ------------------------------------------------------------------
  // Check 1: flip selectivity. Boundary {1,1,1,-1} (default): only T-links
  // (mu = Tp) on the t = Lt-1 slice become -identity.
  // ------------------------------------------------------------------
  LatticeGaugeFieldD U_apbc(UGrid);
  U_apbc = U;
  applyAPBC(U_apbc);

  {
    // Site object of LatticeGaugeFieldD (iLorentzColourMatrix<ComplexD>):
    // iVector<iScalar<iMatrix<ComplexD, Nc>>, Nd> — one iScalar layer.
    using GSite = iVector<iScalar<iMatrix<ComplexD, Nc>>, Nd>;
    GSite s;
    int ts[] = {0, Lt / 2, Lt - 1};
    for (int ti = 0; ti < 3; ti++) {
      Coordinate site({0, 0, 0, ts[ti]});
      peekLocalSite(s, U_apbc, site);
      for (int mu = 0; mu < Nd; mu++) {
        ComplexD diag = s(mu)()(0, 0);
        double expect = (mu == Tp && ts[ti] == Lt - 1) ? -1.0 : 1.0;
        check(std::abs(diag - ComplexD(expect, 0.0)) < 1e-12 &&
                  std::abs(s(mu)()(0, 1)) < 1e-12 &&
                  std::abs(s(mu)()(1, 0)) < 1e-12,
              "site (0,0,0," + std::to_string(ts[ti]) + ") mu=" +
                  std::to_string(mu) + " link is " +
                  (expect < 0.0 ? "-identity" : "identity"));
      }
    }

    // Global arithmetic: only T-links at t = Lt-1 differ, by -2 on each of
    // 3 diagonal entries: norm2 = V_spatial * 3 * (-2)^2.
    double nspat = double(latt[0]) * latt[1] * latt[2];
    RealD expect2 = nspat * 3.0 * 4.0;
    RealD got2 = norm2(U_apbc - U_ref);
    check(std::abs(got2 - expect2) < 1e-6 * expect2,
          "norm2(U_apbc - U_ref) == " + std::to_string(expect2) + " (got " +
              std::to_string(got2) + ")");
  }

  // ------------------------------------------------------------------
  // Check 2: PBC no-op — all-periodic boundary leaves the field unchanged.
  // ------------------------------------------------------------------
  {
    LatticeGaugeFieldD U_pbc(UGrid);
    U_pbc = U;
    applyAPBC(U_pbc, std::vector<int>(Nd, 1));
    check(norm2(U_pbc - U_ref) == 0.0, "all-periodic boundary is a no-op");
  }

  // ------------------------------------------------------------------
  // Check 3: wrap-sign physics. StagGamma(spin=G1, taste=GT) has a single
  // covariant hop in T. On a delta source at the origin, the forward hop
  // from t = Lt-1 wraps through the boundary link U_T(Lt-1), which APBC
  // negates: the wrapped amplitude flips sign, all other timeslices are
  // untouched.
  // ------------------------------------------------------------------
  {
    using FermionFieldD = LatticeStaggeredFermionD;
    using FSite = iColourVector<ComplexD>;

    FermionFieldD src(UGrid);
    src = Zero();
    FSite c;
    c = Zero();
    // iColourVector<ComplexD> = iScalar<iScalar<iVector<ComplexD, Nc>>>:
    // two iScalar layers to peel before indexing the colour vector.
    c()()(0) = ComplexD(1.0, 0.0);
    Coordinate origin({0, 0, 0, 0});
    pokeLocalSite(c, src, origin);

    // PBC transport (periodic unit gauge).
    LatticeGaugeFieldD U_pbc(UGrid);
    U_pbc = U;
    StagGamma gammaPBC(StagGamma::StagAlgebra::G1,
                       StagGamma::StagAlgebra::GT);
    gammaPBC.setGaugeField(U_pbc);
    FermionFieldD rPBC(UGrid);
    gammaPBC.applyGamma(rPBC, src);

    // APBC transport (Check 1's field).
    StagGamma gammaAPBC(StagGamma::StagAlgebra::G1,
                        StagGamma::StagAlgebra::GT);
    gammaAPBC.setGaugeField(U_apbc);
    FermionFieldD rAPBC(UGrid);
    gammaAPBC.applyGamma(rAPBC, src);

    FermionFieldD diff(UGrid);
    diff = rAPBC - rPBC;
    FermionFieldD sumLA(UGrid);
    sumLA = rAPBC + rPBC;
    FermionFieldD zeroF(UGrid);
    zeroF = Zero();
    Lattice<iScalar<vInteger>> tcoor(UGrid);
    LatticeCoordinate(tcoor, Tp);

    // Everything away from the wrapped timeslice is bit-identical.
    FermionFieldD interior(UGrid);
    interior = where(tcoor != Lt - 1, diff, zeroF);
    check(norm2(interior) == 0.0,
          "covariant shift unchanged away from the wrapped timeslice");

    // At t = Lt-1: PBC amplitude nonzero, APBC amplitude is exactly -PBC.
    FermionFieldD wrapPBC(UGrid);
    wrapPBC = where(tcoor == Lt - 1, rPBC, zeroF);
    FermionFieldD wrapSum(UGrid);
    wrapSum = where(tcoor == Lt - 1, sumLA, zeroF);
    FermionFieldD wrapDiff(UGrid);
    wrapDiff = where(tcoor == Lt - 1, diff, zeroF);
    check(norm2(wrapPBC) > 1e-6, "PBC wrap amplitude is nonzero");
    check(norm2(wrapSum) == 0.0 && norm2(wrapDiff) > 1e-6,
          "APBC wrap amplitude is exactly -PBC (anti-periodic transport)");
  }

  Grid_finalize();

  if (nFail != 0) {
    std::cerr << nFail << " check(s) FAILED" << std::endl;
    return 1;
  }
  std::cout << "All APBC checks passed" << std::endl;
  return 0;
}
