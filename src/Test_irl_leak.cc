// Standalone reproducer for memory-leak testing of the implicitly restarted
// Lanczos solver applied to the Schur-preconditioned ImprovedStaggered fermion
// operator. Random gauge fields populate the action; the solve is repeated in
// an infinite loop so RSS can be observed externally.
//
// Build (from build-scalar/):
//   make test_irl_leak
// Run:
//   ./test_irl_leak --grid 4.4.4.4

#include <Grid/Grid.h>
#include <ImplicitlyRestartedLanczos.h>
#include <limits>

using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  typedef ImprovedStaggeredFermionD FermionOpD;
  typedef typename FermionOpD::ImplParams ImplParams;
  typedef typename FermionOpD::FermionField FermionFieldD;

  auto nsimd = GridDefaultSimd(Nd, vComplexD::Nsimd());

  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimd, GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG rng(UGrid);
  rng.SeedUniqueString("irl-leak-repro");

  // Lanczos / Chebyshev parameters. resid is set tighter than is achievable in
  // double precision and MaxIt is effectively infinite, so the IRL never
  // converges and never bails -- it just keeps restarting, which is what we
  // want for a memory-leak observation window.
  const int Nstop = 10;
  const int Nk    = 20;
  const int Nm    = 30;
  const int MaxIt = std::numeric_limits<int>::max();
  const RealD resid = 1.0e-300;

  const RealD chebyAlpha = 0.05;
  const RealD chebyBeta  = 24.0;
  const int   chebyNpoly = 11;

  // Action parameters
  const RealD mass = 0.01;
  const RealD c1   = 1.0;
  const RealD c2   = 1.0;
  const RealD tad  = 1.0;
  const int   cb   = Odd;

  ImplParams implParams;

  long iter = 0;
  while (true) {
    std::cout << GridLogMessage
              << "================ IRL leak iteration " << iter
              << " ================" << std::endl;

    // Fresh random gauge fields each iteration.
    LatticeGaugeFieldD U(UGrid);
    LatticeGaugeFieldD U_fat(UGrid);
    LatticeGaugeFieldD U_long(UGrid);
    SU<Nc>::HotConfiguration(rng, U);
    SU<Nc>::HotConfiguration(rng, U_fat);
    SU<Nc>::HotConfiguration(rng, U_long);

    auto stagMat = std::make_shared<FermionOpD>(
        *UGrid, *UrbGrid, 2.0 * mass, 2.0 * c1, 2.0 * c2, tad, implParams);
    stagMat->ImportGaugeSimple(U_long, U_fat);

    SchurStaggeredOperator<FermionOpD, FermionFieldD> hermOp(*stagMat);
    Chebyshev<FermionFieldD> Cheby(chebyAlpha, chebyBeta, chebyNpoly);

    FunctionHermOp<FermionFieldD> OpCheby(Cheby, hermOp);
    PlainHermOp<FermionFieldD> Op(hermOp);

    ImplicitlyRestartedLanczosFM<FermionFieldD> IRL(
        OpCheby, Op, Nstop, Nk, Nm, resid, MaxIt);

    FermionFieldD gauss(UGrid);
    FermionFieldD src(UrbGrid);
    gaussian(rng, gauss);
    pickCheckerboard(cb, src, gauss);

    std::vector<RealD>          eval(Nm);
    std::vector<FermionFieldD>  evec(Nm, UrbGrid);
    int Nconv = 0;
    IRL.calc(eval, evec, src, Nconv);

    std::cout << GridLogMessage << "Iteration " << iter
              << " converged " << Nconv << " eigenvectors" << std::endl;
    ++iter;
  }

  Grid_finalize();
}
