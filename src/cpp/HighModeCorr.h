#ifndef FMGRID_HIGHMODECORR_H
#define FMGRID_HIGHMODECORR_H

#include <DilutedNoise.h>
#include <Eigenpack.h>
#include <Grid/Grid.h>
#include <IO.h>
#include <StagGamma.h>
#include <functional>
#include <map>
#include <tuple>

NAMESPACE_BEGIN(Grid);

template <typename FImpl, typename FermionOpD, typename FermionOpF>
void computeHighModeCorrelators(
    GlobalPar &inputParams, GridCartesian *UGrid,
    GridRedBlackCartesian *UrbGrid, GridCartesian *UGridF,
    GridRedBlackCartesian *UrbGridF, GridParallelRNG &rng,
    LatticeGaugeFieldD &U, LatticeGaugeFieldD &U_fat,
    LatticeGaugeFieldD &U_long, LatticeGaugeFieldF &U_fat_f,
    LatticeGaugeFieldF &U_long_f,
    typename FermionOpD::ImplParams &implParams,
    std::shared_ptr<EigenPack<typename FImpl::FermionField>> epack) {

  using FermionFieldD = typename FImpl::FermionField;
  using FermionFieldF = typename FermionOpF::FermionField;
  using PropagatorFieldD = typename FImpl::PropagatorField;
  using SolverFunc = std::function<void()>;

  bool hasEigs = (epack != nullptr);
  int Nt = UGrid->GlobalDimensions()[Tp];
  int traj = inputParams.trajectory;

  // Action factory
  auto makeAction = [&](auto &action, const ImprovedStaggeredPar &actionPar) {
    std::cout << GridLogMessage << "\nCreating ImprovedStaggeredFermion "
              << std::endl;
    std::cout << GridLogMessage << "  mass = " << actionPar.mass << std::endl;
    std::cout << GridLogMessage << "  c1 = " << actionPar.c1 << std::endl;
    std::cout << GridLogMessage << "  c2 = " << actionPar.c2 << std::endl;
    std::cout << GridLogMessage << "  tadpole = " << actionPar.tad << std::endl;

    using T = std::decay_t<decltype(action)>;
    if constexpr (std::is_same_v<T, std::shared_ptr<FermionOpF>>) {
      action = std::make_shared<FermionOpF>(
          *UGridF, *UrbGridF, 2. * actionPar.mass, 2. * actionPar.c1,
          2. * actionPar.c2, actionPar.tad, implParams);
      action->ImportGaugeSimple(U_long_f, U_fat_f);
    } else if constexpr (std::is_same_v<T, std::shared_ptr<FermionOpD>>) {
      action = std::make_shared<FermionOpD>(
          *UGrid, *UrbGrid, 2. * actionPar.mass, 2. * actionPar.c1,
          2. * actionPar.c2, actionPar.tad, implParams);
      action->ImportGaugeSimple(U_long, U_fat);
    }
  };

  // Temporary fields for solves
  auto fermOut = std::make_shared<FermionFieldD>(UGrid);
  auto fermIn = std::make_shared<FermionFieldD>(UGrid);
  std::shared_ptr<FermionFieldD> fermGuess;
  if (hasEigs) {
    fermGuess = std::make_shared<FermionFieldD>(UGrid);
  }

  // LMA solver infrastructure
  std::shared_ptr<FermionFieldD> rbFerm, rbFermNeg, MrbFermNeg, rbTemp,
      rbTempNeg;
  unsigned int eigStart = 0;
  int nEigs = 0;
  bool projector = false;

  if (hasEigs) {
    eigStart = inputParams.lma.eigStart;
    nEigs = inputParams.lma.nEigs;
    projector = inputParams.lma.projector;

    if (nEigs < 1) {
      nEigs = epack->evec.size();
    }

    if (eigStart > static_cast<unsigned int>(nEigs) ||
        eigStart > epack->evec.size() ||
        nEigs - eigStart > static_cast<int>(epack->evec.size()) - eigStart) {
      std::cerr << "ERROR: Requested eigs (eigStart and nEigs) out of bounds"
                << std::endl;
      exit(1);
    }

    std::cout << GridLogMessage << "Setting up low mode projector" << std::endl;
    std::cout << GridLogMessage << "  eigStart = " << eigStart << std::endl;
    std::cout << GridLogMessage << "  nEigs = " << nEigs << std::endl;
    std::cout << GridLogMessage
              << "  projector = " << (projector ? "true" : "false")
              << std::endl;

    rbFerm = std::make_shared<FermionFieldD>(UrbGrid);
    rbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
    MrbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
    rbTemp = std::make_shared<FermionFieldD>(UrbGrid);
    rbTempNeg = std::make_shared<FermionFieldD>(UrbGrid);

    std::cout << GridLogMessage << "Low mode projector setup complete"
              << std::endl;
  }

  // Lambda to create LMA solver function
  auto makeLMASolver = [epack, rbFerm, rbFermNeg, MrbFermNeg, rbTemp,
                        rbTempNeg, fermOut,
                        fermIn](std::shared_ptr<FermionOpD> actionMat,
                                RealD sMass, bool projectorFlag,
                                unsigned int eStart, int nE, bool subGuess) {
    return [actionMat, epack, subGuess, sMass, projectorFlag, eStart, nE,
            rbFerm, rbFermNeg, MrbFermNeg, rbTemp, rbTempNeg, fermOut,
            fermIn]() {
      int cb = epack->evec[0].Checkerboard();
      int cbNeg = (cb == Even) ? Odd : Even;

      RealD norm = 1.0 / ::sqrt(norm2(epack->evec[0]));

      *rbTemp = Zero();
      rbTemp->Checkerboard() = cb;
      *rbTempNeg = Zero();
      rbTempNeg->Checkerboard() = cb;

      rbFerm->Checkerboard() = cb;
      rbFermNeg->Checkerboard() = cbNeg;
      MrbFermNeg->Checkerboard() = cb;

      pickCheckerboard(cb, *rbFerm, *fermIn);
      pickCheckerboard(cbNeg, *rbFermNeg, *fermIn);

      actionMat->MeooeDag(*rbFermNeg, *MrbFermNeg);

      for (int k = (eStart + nE - 1); k >= static_cast<int>(eStart); k--) {
        const FermionFieldD &e = epack->evec[k];

        const RealD lam_DD = epack->eval[k];
        const RealD invlam_DD = 1.0 / lam_DD;
        const RealD invmag = 1.0 / (sMass * sMass + lam_DD);

        if (!projectorFlag) {
          const ComplexD ip = TensorRemove(innerProduct(e, *rbFerm)) * invmag;
          const ComplexD ipNeg =
              TensorRemove(innerProduct(e, *MrbFermNeg)) * invmag;
          axpy(*rbTemp, sMass * ip + ipNeg, e, *rbTemp);
          axpy(*rbTempNeg, sMass * ipNeg * invlam_DD - ip, e, *rbTempNeg);
        } else {
          const ComplexD ip = TensorRemove(innerProduct(e, *rbFerm));
          const ComplexD ipNeg = TensorRemove(innerProduct(e, *MrbFermNeg));
          axpy(*rbTemp, ip, e, *rbTemp);
          axpy(*rbTempNeg, ipNeg * invlam_DD, e, *rbTempNeg);
        }
      }

      actionMat->Meooe(*rbTempNeg, *rbFermNeg);

      setCheckerboard(*fermOut, *rbTemp);
      setCheckerboard(*fermOut, *rbFermNeg);

      *fermOut *= norm;

      if (subGuess) {
        if (projectorFlag) {
          *fermOut = *fermIn - *fermOut;
        } else {
          std::cerr << "ERROR: Subtracted solver only supported for "
                       "projector=true"
                    << std::endl;
          exit(1);
        }
      }
    };
  };

  // Lambda to create MPCG solver function
  auto makeMPCGSolver = [fermOut, fermIn, fermGuess, &UGrid, &UGridF,
                         &inputParams](std::shared_ptr<FermionOpD> actionMatD,
                                       std::shared_ptr<FermionOpF> actionMatF,
                                       bool subGuess) {
    auto hermOpOuter =
        std::make_shared<MdagMLinearOperator<FermionOpD, FermionFieldD>>(
            *actionMatD);
    auto hermOpInner =
        std::make_shared<MdagMLinearOperator<FermionOpF, FermionFieldF>>(
            *actionMatF);
    auto temp = std::make_shared<FermionFieldD>(UGrid);
    auto &mpcgPar = inputParams.mpcg;

    return [actionMatD, subGuess, fermOut, fermIn, fermGuess, temp,
            hermOpInner, hermOpOuter, &UGridF, &mpcgPar]() {
      MixedPrecisionConjugateGradient<FermionFieldD, FermionFieldF> mpcg(
          mpcgPar.residual, mpcgPar.maxInnerIteration,
          mpcgPar.maxOuterIteration, UGridF, *hermOpInner, *hermOpOuter);

      std::cout << GridLogMessage << "MPCG"
                << (fermGuess == nullptr ? "Null" : "Not Null") << std::endl;
      if (fermGuess != nullptr) {
        *fermOut = *fermGuess;
      } else {
        *fermOut = 1.0;
      }

      ZeroGuesser<FermionFieldF> iguesserDefault;
      mpcg.useGuesser(iguesserDefault);
      *temp = Zero();
      actionMatD->Mdag(*fermIn, *temp);

      mpcg(*temp, *fermOut);

      RealD nsol = norm2(*fermOut);
      actionMatD->M(*fermOut, *temp);
      RealD nMsol = norm2(*temp);
      *temp = *temp - *fermIn;

      RealD ns = norm2(*fermIn);
      RealD nr = norm2(*temp);
      RealD relres = (ns > 0.0) ? std::sqrt(nr / ns) : 0.0;

      std::cout << GridLogMessage << "MPCG: Final true residual = " << relres
                << std::endl;

      if (subGuess && fermGuess != nullptr) {
        *fermOut = *fermOut - *fermGuess;
      }
    };
  };

  // Build per-action solvers, indexed by label
  size_t nActions = inputParams.highModeActions.size();
  std::map<std::string, size_t> actionLabelMap;
  std::vector<std::shared_ptr<FermionOpD>> actionMatsD(nActions);
  std::vector<std::shared_ptr<FermionOpF>> actionMatsF(nActions);
  std::vector<RealD> actionMasses(nActions);
  std::vector<SolverFunc> lmaSolvers(nActions);
  std::vector<SolverFunc> mpcgSolvers(nActions);

  for (size_t aIdx = 0; aIdx < nActions; ++aIdx) {
    auto &actionPar = inputParams.highModeActions[aIdx];
    actionLabelMap[actionPar.label] = aIdx;
    makeAction(actionMatsD[aIdx], actionPar);
    makeAction(actionMatsF[aIdx], actionPar);
    actionMasses[aIdx] = 2.0 * actionPar.mass;

    std::cout << GridLogMessage << "Setting up action '" << actionPar.label
              << "' (index " << aIdx << ") with mass = " << actionPar.mass
              << std::endl;

    if (hasEigs) {
      lmaSolvers[aIdx] = makeLMASolver(actionMatsD[aIdx], actionMasses[aIdx],
                                       projector, eigStart, nEigs, false);
    }

    auto &mpcgPar = inputParams.mpcg;
    std::cout << GridLogMessage
              << "Setting up mixed-precision CG solver for action '"
              << actionPar.label << "'" << std::endl;
    std::cout << GridLogMessage << "  Residual: " << mpcgPar.residual
              << std::endl;

    mpcgSolvers[aIdx] =
        makeMPCGSolver(actionMatsD[aIdx], actionMatsF[aIdx], false);

    std::cout << GridLogMessage << "Solvers created for action '"
              << actionPar.label << "'" << std::endl;
  }

  // Helper to resolve action label to index
  auto resolveAction = [&actionLabelMap](const std::string &label) -> size_t {
    auto it = actionLabelMap.find(label);
    if (it == actionLabelMap.end()) {
      std::cerr << "ERROR: Unknown action label '" << label << "'" << std::endl;
      exit(1);
    }
    return it->second;
  };

  // Helper to resolve solver label to solver function
  auto resolveSolver = [&lmaSolvers, &mpcgSolvers,
                        hasEigs](const std::string &solverName,
                                 size_t aIdx) -> SolverFunc * {
    if (solverName == "lma") {
      if (!hasEigs) {
        std::cerr << "ERROR: lma solver requested but no eigenpack available"
                  << std::endl;
        exit(1);
      }
      return &lmaSolvers[aIdx];
    } else if (solverName == "mpcg") {
      return &mpcgSolvers[aIdx];
    } else {
      std::cerr << "ERROR: Unknown solver '" << solverName << "'" << std::endl;
      exit(1);
    }
    return nullptr;
  };

  for (auto &sourcePar : inputParams.sources) {
    unsigned int tStep = sourcePar.tStep;
    unsigned int t0 = sourcePar.t0;
    unsigned int nSrc = sourcePar.nSrc;

    std::cout << GridLogMessage
              << "Setting up random wall sources (color-diagonal)" << std::endl;
    std::cout << GridLogMessage << "  tStep = " << tStep << std::endl;
    std::cout << GridLogMessage << "  t0 = " << t0 << std::endl;
    std::cout << GridLogMessage << "  nSrc = " << nSrc << std::endl;

    if (t0 >= tStep) {
      std::cerr << "ERROR: t0 >= tStep" << std::endl;
      exit(1);
    }
    TimeDilutedNoise<FImpl> noise(UGrid, nSrc);
    std::string seed =
        getSeed(inputParams, sourcePar.seed);
    std::cout << GridLogMessage << "Seeding source with seed '" << seed << "'"
              << std::endl;
    rng.SeedUniqueString(seed);
    noise.generateNoise(rng);
    int nSlices = Nt / std::min(static_cast<int>(tStep), Nt);
    int nVecs = nSrc * nSlices;
    GRID_ASSERT(nVecs == 1);

    std::cout << GridLogMessage << "  Number of time slices: " << nSlices
              << std::endl;
    std::cout << GridLogMessage << "  Total number of sources: " << nVecs
              << std::endl;

    PropagatorFieldD randomWallSource(UGrid);
    for (int i = 0; i < nSrc; i++) {
      for (int j = 0; j < nSlices; j++) {
        int timeSlice = j * tStep + t0;
        int offset = i * Nt + j * tStep + t0;

        randomWallSource = noise.getProp(offset);
        std::cout << GridLogMessage << "Random wall sources setup complete"
                  << std::endl;

        // Propagator cache
        std::map<std::tuple<std::string, std::string, std::string>,
                 PropagatorFieldD>
            propCache;

        std::map<std::pair<std::string, std::string>, PropagatorFieldD>
            lmaPropCache;

        for (auto &corrPar : inputParams.corr) {
          std::cout << GridLogMessage << "Setting up meson contraction"
                    << std::endl;

          auto quarkGammaKeys =
              StagGamma::ParseSpinTaste(corrPar.quark.gammas);
          auto quarkGammaVals = StagGamma::ParseSpinTaste(
              corrPar.quark.gammas, corrPar.quark.applyG5);
          GRID_ASSERT(!quarkGammaKeys.empty());

          auto antiquarkGammaKeys =
              StagGamma::ParseSpinTaste(corrPar.antiquark.gammas);
          auto antiquarkGammaVals = StagGamma::ParseSpinTaste(
              corrPar.antiquark.gammas, corrPar.antiquark.applyG5);
          GRID_ASSERT(antiquarkGammaKeys.size() == 1);
          std::string antiquarkGammaName =
              StagGamma::GetName(antiquarkGammaKeys[0]);
          StagGamma::SpinTastePair antiquarkSpinTaste = antiquarkGammaVals[0];

          auto sinkGammaKeys =
              StagGamma::ParseSpinTaste(corrPar.sink.gammas);
          auto sinkGammaVals = StagGamma::ParseSpinTaste(
              corrPar.sink.gammas, corrPar.sink.applyG5);
          GRID_ASSERT(sinkGammaKeys.size() == quarkGammaKeys.size());

          std::map<std::string, StagGamma::SpinTastePair> solveGammas;
          for (size_t gi = 0; gi < quarkGammaKeys.size(); ++gi) {
            solveGammas.emplace(StagGamma::GetName(quarkGammaKeys[gi]),
                                quarkGammaVals[gi]);
          }
          solveGammas.emplace(antiquarkGammaName, antiquarkSpinTaste);

          size_t quarkActionIdx = resolveAction(corrPar.quarkAction);
          size_t antiquarkActionIdx = resolveAction(corrPar.antiquarkAction);

          std::cout << GridLogMessage << "Correlator: quarkAction='"
                    << corrPar.quarkAction << "' (" << corrPar.quarkSolver
                    << "), antiquarkAction='" << corrPar.antiquarkAction
                    << "' (" << corrPar.antiquarkSolver << ")" << std::endl;

          // Lambda to solve a set of gammas with a given action+solver
          auto doSolves =
              [&propCache, &lmaPropCache, &randomWallSource, &solveGammas,
               fermIn, fermOut, fermGuess, &U, UGrid, &lmaSolvers, hasEigs](
                  const std::string &actionLabel, size_t aIdx,
                  const std::string &solverType, SolverFunc &solver,
                  const std::vector<StagGamma::SpinTastePair> &gammaKeys) {
                StagGamma gamma;
                gamma.setGaugeField(U);
                PropagatorFieldD gammaProp(UGrid);

                for (const auto &gKey : gammaKeys) {
                  std::string gammaName = StagGamma::GetName(gKey);
                  auto cacheKey =
                      std::make_tuple(actionLabel, solverType, gammaName);

                  if (propCache.find(cacheKey) != propCache.end())
                    continue;

                  // If MPCG, ensure LMA solve exists for guess
                  if (solverType == "mpcg" && hasEigs) {
                    auto lmaCacheKey = std::make_tuple(
                        actionLabel, std::string("lma"), gammaName);
                    if (propCache.find(lmaCacheKey) == propCache.end()) {
                      std::cout << GridLogMessage
                                << "Pre-solving LMA for MPCG guess: "
                                << gammaName << " (action '" << actionLabel
                                << "')" << std::endl;
                      propCache.emplace(std::piecewise_construct,
                                        std::forward_as_tuple(lmaCacheKey),
                                        std::forward_as_tuple(UGrid));
                      propCache.at(lmaCacheKey) = Zero();

                      gamma.setSpinTaste(solveGammas.at(gammaName));
                      gammaProp = Zero();
                      gamma(gammaProp, randomWallSource);

                      for (int c = 0; c < 3; c++) {
                        PropToFerm<FImpl>(*fermIn, gammaProp, c);
                        *fermOut = Zero();
                        lmaSolvers[aIdx]();
                        FermToProp<FImpl>(propCache.at(lmaCacheKey), *fermOut,
                                          c);
                      }
                    }
                  }

                  std::cout << GridLogMessage << "Solving gamma: " << gammaName
                            << " (action '" << actionLabel << "', solver '"
                            << solverType << "')" << std::endl;
                  propCache.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(cacheKey),
                                    std::forward_as_tuple(UGrid));
                  propCache.at(cacheKey) = Zero();

                  *fermIn = Zero();
                  *fermOut = Zero();
                  gamma.setSpinTaste(solveGammas.at(gammaName));

                  gammaProp = Zero();
                  gamma(gammaProp, randomWallSource);

                  for (int c = 0; c < 3; c++) {
                    PropToFerm<FImpl>(*fermIn, gammaProp, c);

                    if (solverType == "mpcg" && hasEigs &&
                        fermGuess != nullptr) {
                      auto lmaCacheKey = std::make_tuple(
                          actionLabel, std::string("lma"), gammaName);
                      PropToFerm<FImpl>(*fermGuess, propCache.at(lmaCacheKey),
                                        c);
                    }
                    solver();
                    FermToProp<FImpl>(propCache.at(cacheKey), *fermOut, c);
                  }
                }
              };

          // Solve quark gammas
          SolverFunc *quarkSolver =
              resolveSolver(corrPar.quarkSolver, quarkActionIdx);
          doSolves(corrPar.quarkAction, quarkActionIdx, corrPar.quarkSolver,
                   *quarkSolver, quarkGammaKeys);

          // Solve antiquark gammas
          SolverFunc *antiquarkSolver =
              resolveSolver(corrPar.antiquarkSolver, antiquarkActionIdx);
          doSolves(corrPar.antiquarkAction, antiquarkActionIdx,
                   corrPar.antiquarkSolver, *antiquarkSolver,
                   antiquarkGammaKeys);

          // Initialize meson results
          std::vector<MesonResult> mesonResults(quarkGammaKeys.size());
          for (size_t gi = 0; gi < quarkGammaKeys.size(); ++gi) {
            std::string quarkGammaName =
                StagGamma::GetName(quarkGammaKeys[gi]);
            std::string sinkGammaName =
                StagGamma::GetName(sinkGammaKeys[gi]);

            mesonResults[gi].sourceGamma = quarkGammaName;
            mesonResults[gi].sinkGamma = sinkGammaName;
            mesonResults[gi].corr.resize(Nt, 0.0);
            mesonResults[gi].srcCorrs.resize(nVecs,
                                             std::vector<Complex>(Nt, 0.0));
            mesonResults[gi].scaling = nVecs;
          }

          // Contract
          {
            StagGamma gamma;
            gamma.setGaugeField(U);
            PropagatorFieldD gammaProp(UGrid);

            for (size_t gi = 0; gi < mesonResults.size(); ++gi) {
              std::string quarkGammaName = mesonResults[gi].sourceGamma;
              std::string sinkGammaName = mesonResults[gi].sinkGamma;
              std::cout << GridLogMessage
                        << "Contracting source gamma: " << quarkGammaName
                        << ", sink gamma: " << sinkGammaName << std::endl;
              gamma.setSpinTaste(sinkGammaVals[gi]);

              auto quarkKey = std::make_tuple(
                  corrPar.quarkAction, corrPar.quarkSolver, quarkGammaName);
              auto antiquarkKey =
                  std::make_tuple(corrPar.antiquarkAction,
                                  corrPar.antiquarkSolver, antiquarkGammaName);

              PropagatorFieldD prod(UGrid);
              gamma(gammaProp, propCache.at(quarkKey));
              prod = propCache.at(antiquarkKey) * adj(gammaProp);

              std::vector<TComplex> buf;
              LatticeComplexD slicedTrace = trace(prod);
              sliceSum(slicedTrace, buf, Tp);
              int sliceOffset = t0;
              for (int t = 0; t < Nt; ++t) {
                Complex ct = TensorRemove(buf[sliceOffset]);
                mesonResults[gi].srcCorrs[0][t] = ct;
                sliceOffset = mod(sliceOffset + 1, Nt);
              }
            }

            // Compute averaged correlators
            for (size_t gi = 0; gi < mesonResults.size(); ++gi) {
              for (int t = 0; t < Nt; ++t) {
                mesonResults[gi].corr[t] = 0.0;
                for (int si = 0; si < mesonResults[gi].scaling; si++) {
                  mesonResults[gi].corr[t] += mesonResults[gi].srcCorrs[si][t];
                }
                mesonResults[gi].corr[t] /= mesonResults[gi].scaling;
              }
            }
          }

          if (!corrPar.output.empty()) {
            std::cout << GridLogMessage << "Saving correlator to "
                      << corrPar.output << std::endl;
            saveResult(UGrid, corrPar.output, "meson", mesonResults,
                       inputParams, t0);
          }
        }
      }
    }
  }
}

NAMESPACE_END(Grid);

#endif
