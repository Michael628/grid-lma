#ifndef FMGRID_EPACK_H
#define FMGRID_EPACK_H

#include <Eigenpack.h>
#include <Grid/Grid.h>
#include <IO.h>
#include <ImplicitlyRestartedLanczos.h>

NAMESPACE_BEGIN(Grid);

template <typename FermionOpD, typename FermionFieldD>
std::shared_ptr<EigenPack<FermionFieldD>>
loadOrSolveEigenpack(const EpackPar &epackPar, GlobalPar &inputParams,
                     GridCartesian *UGrid, GridRedBlackCartesian *UrbGrid,
                     GridParallelRNG &rng, LatticeGaugeFieldD &U_fat,
                     LatticeGaugeFieldD &U_long,
                     typename FermionOpD::ImplParams &implParams, int traj) {

  auto epack = std::make_shared<EigenPack<FermionFieldD>>();
  int cb = epackPar.checker;

  auto makeActionD = [&](const ImprovedStaggeredPar &actionPar) {
    auto action = std::make_shared<FermionOpD>(
        *UGrid, *UrbGrid, 2. * actionPar.mass, 2. * actionPar.c1,
        2. * actionPar.c2, actionPar.tad, implParams);
    action->ImportGaugeSimple(U_long, U_fat);
    return action;
  };

  if (epackPar.type == EpackPar::EpackType::solve) {
    auto &actionParIRL = epackPar.action;

    std::cout << GridLogMessage
              << "\n========================================" << std::endl;
    std::cout << GridLogMessage << "MODULE: MSolver::StagFermionIRL"
              << std::endl;
    std::cout << GridLogMessage
              << "========================================" << std::endl;

    auto &lanczosPar = epackPar.irl.lanczosParams;
    const int Nstop = lanczosPar.Nstop;
    const int Nk = lanczosPar.Nk;
    const int Nm = lanczosPar.Nm;
    const int MaxIt = lanczosPar.MaxIt;
    RealD resid = lanczosPar.resid;

    std::cout << GridLogMessage << "IRL Parameters:" << std::endl;
    std::cout << GridLogMessage << "  Nstop = " << Nstop << std::endl;
    std::cout << GridLogMessage << "  Nk = " << Nk << std::endl;
    std::cout << GridLogMessage << "  Nm = " << Nm << std::endl;
    std::cout << GridLogMessage << "  MaxIt = " << MaxIt << std::endl;
    std::cout << GridLogMessage << "  resid = " << resid << std::endl;

    auto stagMatIRL = makeActionD(actionParIRL);

    SchurStaggeredOperator<FermionOpD, FermionFieldD> hermOpIRL(*stagMatIRL);
    Chebyshev<FermionFieldD> Cheby(
        lanczosPar.Cheby.alpha, lanczosPar.Cheby.beta, lanczosPar.Cheby.Npoly);

    FunctionHermOp<FermionFieldD> OpCheby(Cheby, hermOpIRL);
    PlainHermOp<FermionFieldD> Op(hermOpIRL);

    ImplicitlyRestartedLanczosFM<FermionFieldD> IRL(OpCheby, Op, Nstop, Nk, Nm,
                                                    resid, MaxIt);

    FermionFieldD src(UrbGrid);

    std::cout << GridLogMessage
              << "Generating random source (checkerboard = " << cb << ")"
              << std::endl;
    FermionFieldD gauss(UGrid);
    std::string seed = getSeed(inputParams, epackPar.seed);
    rng.SeedUniqueString(seed);
    gaussian(rng, gauss);
    pickCheckerboard(cb, src, gauss);

    std::cout << GridLogMessage << "Running IRL eigensolver..." << std::endl;
    int Nconv;
    epack->eval.resize(Nm);
    epack->evec.resize(Nm, UrbGrid);
    IRL.calc(epack->eval, epack->evec, src, Nconv);

    std::cout << GridLogMessage << "Converged " << Nconv << " eigenvectors"
              << std::endl;

    epack->eval.resize(Nstop);
    epack->evec.resize(Nstop, UGrid);
    epack->record.operatorXml = actionParIRL.parString();
    epack->record.solverXml = epackPar.irl.parString();

    if (!epackPar.file.empty()) {
      std::cout << GridLogMessage << "Saving eigenpack to " << epackPar.file
                << std::endl;
      epack->write(epackPar.file, epackPar.multiFile, traj);
    }
  }

  if (epackPar.type == EpackPar::EpackType::load) {
    std::cout << GridLogMessage << "Loading eigenpack from " << epackPar.file
              << std::endl;
    assert(!epackPar.file.empty());
    epack->eval.resize(epackPar.size);
    epack->evec.resize(epackPar.size, UrbGrid);
    epack->read(epackPar.file, epackPar.multiFile, traj);
  }

  if (!epackPar.evalSave.empty()) {
    std::cout << GridLogMessage << "Saving eigenvalues to " << epackPar.evalSave
              << std::endl;
    saveResult(UGrid, epackPar.evalSave, "evals", epack->eval, inputParams,
               "h5", false);
  }

  std::cout << GridLogMessage << "Setting checkerboard of eigenvectors to "
            << (cb == Even ? "Even" : "Odd") << std::endl;
  for (auto &e : epack->evec) {
    e.Checkerboard() = cb;
  }

  return epack;
}

NAMESPACE_END(Grid);

#endif
