#include <A2AMatrix.h>
#include <DilutedNoise.h>
#include <Eigenpack.h>
#include <Grid/Grid.h>
#include <IO.h>
#include <StagGamma.h>
#include <a2a/A2AWorker.h>
#include <functional>

using namespace std;
using namespace Grid;

template <typename T, typename FImpl>
class MesonFieldKernel : public A2AKernel<T, typename FImpl::FermionField> {
public:
  FERM_TYPE_ALIASES(FImpl, );

public:
  MesonFieldKernel(GridBase *grid) {
    _vol = 1.;
    for (auto &d : grid->GlobalDimensions()) {
      _vol *= d;
    }
  }
  virtual ~MesonFieldKernel(void) {};

  virtual void operator()(A2AMatrixSet<T> &m, const FermionField *left_e,
                          const FermionField *left_o,
                          const FermionField *right_e,
                          const FermionField *right_o) {
    MesonFunction<FImpl>(m, left_e, left_o, right_e, right_o);
  }

  virtual double flops(const unsigned int blockSizei,
                       const unsigned int blockSizej, int cbDiv = 1) {

    return _vol / cbDiv * (_worker->getFlops()) * blockSizei * blockSizej;
  }

  virtual double bytes(const unsigned int blockSizei,
                       const unsigned int blockSizej) {
    // return _vol*(12.0*sizeof(T))*blockSizei*blockSizej
    // +  _vol*(2.0*sizeof(T)*_mom.size())*blockSizei*blockSizej*_gamma.size();
    return -1.0;
  }

  virtual double kernelTime() { return _worker->_t_kernel; }
  virtual double globalSumTime() { return _worker->_t_gsum; }
  void setWorker(GridBase *grid, const std::vector<ComplexField> &mom,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 int orthogDir, LatticeGaugeField *U) {
    _worker = std::make_unique<A2AWorkerOnelink<FImpl>>(grid, mom, gammas, U,
                                                        orthogDir);
  }
  void setWorker(GridBase *grid, const std::vector<ComplexField> &mom,
                 const std::vector<StagGamma::SpinTastePair> &gammas,
                 int orthogDir) {
    _worker =
        std::make_unique<A2AWorkerLocal<FImpl>>(grid, mom, gammas, orthogDir);
  }

private:
  template <typename TFImpl, typename... Args>
  void MesonFunction(Args &&...args) {
    _worker->StagMesonField(args...);
  }

private:
  double _vol;
  std::unique_ptr<A2AWorkerBase<FImpl>> _worker;
};

template <typename FImpl, typename Pack>
class MesonFieldData
    : public A2AData<typename FImpl::FermionField, MesonFieldMetadata> {
  using Field = typename FImpl::FermionField;
  using FMat = FermionOperator<FImpl>;

public:
  MesonFieldData(FMat *action, RealD solverMass,
                 const std::vector<std::vector<Real>> &mom,
                 const std::string &outputPath, int traj)
      : _action(action), _solverMass(solverMass), _mom(mom),
        _outputPath(outputPath), _traj(traj) {}

  void setEpack(Pack &epack) {
    _epack = &epack;
    _evalMassive.resize(epack.eval.size());
    for (size_t i = 0; i < epack.eval.size(); i++)
      _evalMassive[i] = ComplexD(_solverMass, ::sqrt(epack.eval[i]));
  }

  void setLeft(std::vector<Field> &left) { _left = &left; }
  void setRight(std::vector<Field> &right) { _right = &right; }
  void setGammas(const std::vector<StagGamma::SpinTastePair> &gammas) {
    _gammas = gammas;
  }

  const std::vector<Field> &left() const override {
    return _left ? *_left : _emptyFields;
  }
  const std::vector<Field> &right() const override {
    return _right ? *_right : _emptyFields;
  }
  bool hasLowModes() const override { return _epack != nullptr; }
  std::vector<Field> &evecs() override {
    assert(_epack != nullptr);
    return _epack->evec;
  }
  const std::vector<ComplexD> &evals() const override {
    return _epack ? _evalMassive : _emptyEvals;
  }

  void swapChecker(std::vector<Field> &lowBuf, int startIdx) override {
    assert(_action != nullptr && _epack != nullptr);
    auto &ev = this->evecs();
    int count = std::min((int)lowBuf.size(), (int)ev.size() - startIdx);
    if (!_temp) {
      _temp = std::make_shared<Field>(ev.at(0).Grid());
      *_temp = Zero();
    }
    for (int i = 0; i < count; i++) {
      Field &evec = ev.at(startIdx + i);
      lowBuf[i] = evec;
      ComplexD eval_D = ComplexD(0.0, _evalMassive[startIdx + i].imag());
      int cb = evec.Checkerboard();
      int cbNeg = (cb == Even) ? Odd : Even;
      _temp->Checkerboard() = cbNeg;
      _action->Meooe(evec, *_temp);
      evec.Checkerboard() = cbNeg;
      evec = (1.0 / eval_D) * (*_temp);
    }
  }

  std::string ioname(unsigned int m, unsigned int g) const override {
    std::stringstream ss;
    ss << StagGamma::GetName(_gammas[g]) << "_";
    for (unsigned int mu = 0; mu < _mom[m].size(); ++mu) {
      ss << _mom[m][mu] << ((mu == _mom[m].size() - 1) ? "" : "_");
    }
    return ss.str();
  }

  std::string filename(unsigned int m, unsigned int g) const override {
    return _outputPath + "." + std::to_string(_traj) + "/" + ioname(m, g) +
           ".h5";
  }

  MesonFieldMetadata metadata(unsigned int m, unsigned int g) const override {
    MesonFieldMetadata md;
    for (auto pmu : _mom[m]) {
      md.momentum.push_back(pmu);
    }
    md.gamma_spin = _gammas[g].first;
    md.gamma_taste = _gammas[g].second;
    return md;
  }

private:
  FMat *_action;
  RealD _solverMass;
  std::vector<std::vector<Real>> _mom;
  std::string _outputPath;
  int _traj;
  std::vector<StagGamma::SpinTastePair> _gammas;

  Pack *_epack = nullptr;
  std::vector<Field> *_left = nullptr;
  std::vector<Field> *_right = nullptr;

  std::vector<ComplexD> _evalMassive;
  std::vector<Field> _emptyFields;
  std::vector<ComplexD> _emptyEvals;
  std::shared_ptr<Field> _temp;
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 1;

  typedef ImprovedStaggeredFermionD FermionOpD;
  typedef ImprovedStaggeredFermionF FermionOpF;
  typedef typename ImprovedStaggeredFermionD::ImplParams ImplParams;
  typedef typename ImprovedStaggeredFermionD::Impl_t FImpl;
  typedef typename ImprovedStaggeredFermionD::PropagatorField PropagatorFieldD;
  typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;
  typedef typename ImprovedStaggeredFermionF::FermionField FermionFieldF;
  typedef A2AMatrixBlockComputation<ComplexD, FermionFieldD, MesonFieldMetadata,
                                    HADRONS_A2AM_IO_TYPE>
      Computation;
  typedef MesonFieldKernel<Complex, FImpl> Kernel;

  std::string paramFile = argv[1];
  XmlReader reader(paramFile, false, "grid");

  GlobalPar inputParams;
  read(reader, "parameters", inputParams);

  auto latt = GridDefaultLatt();
  auto nsimd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  auto nsimdf = GridDefaultSimd(Nd, vComplexF::Nsimd());
  auto mpi_layout = GridDefaultMpi();
  // ========================================================================
  // SETUP: Grid communicator layouts
  // ========================================================================
  GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimd, GridDefaultMpi());
  GridRedBlackCartesian *UrbGrid =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridCartesian *UGridF = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), nsimdf, GridDefaultMpi());
  GridRedBlackCartesian *UrbGridF =
      SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);

  GridParallelRNG rng(UGrid);
  // ========================================================================
  // MODULE: MIO::LoadIldg (Load gauge configurations)
  // ========================================================================
  std::cout << GridLogMessage
            << "========================================" << std::endl;
  std::cout << GridLogMessage << "MODULE: MIO::LoadIldg" << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  LatticeGaugeFieldD U(UGrid);
  LatticeGaugeFieldD U_fat(UGrid);
  LatticeGaugeFieldD U_long(UGrid);

  FieldMetaData header;
  int traj = inputParams.trajectory;
  IldgReader IR;

  // Get lattice dimensions
  int Nt = UGrid->GlobalDimensions()[Tp];

  switch (inputParams.gauge.type) {
  case GaugePar::GaugeType::free:
    SU<Nc>::ColdConfiguration(U);
    SU<Nc>::ColdConfiguration(U_fat);
    SU<Nc>::ColdConfiguration(U_long);
    break;
  case GaugePar::GaugeType::file: {
    // Load fat links (double precision)
    std::string file_fat =
        inputParams.gauge.fatlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading fat links from " << file_fat
              << std::endl;
    IR.open(file_fat);
    IR.readConfiguration(U_fat, header);
    IR.close();

    // Load long links (double precision)
    std::string file_long =
        inputParams.gauge.longlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading long links from " << file_long
              << std::endl;
    IR.open(file_long);
    IR.readConfiguration(U_long, header);
    IR.close();

    // Load base gauge field (double precision)
    std::string file_base = inputParams.gauge.link + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading base gauge field from " << file_base
              << std::endl;
    IR.open(file_base);
    IR.readConfiguration(U, header);
    IR.close();
  } break;
  case GaugePar::GaugeType::hot:
    SU<Nc>::HotConfiguration(rng, U);
    SU<Nc>::HotConfiguration(rng, U_fat);
    SU<Nc>::HotConfiguration(rng, U_long);
    break;
  }

  // ========================================================================
  // MODULE: MUtilities::GaugeSinglePrecisionCast (Cast to single precision)
  // ========================================================================
  std::cout << GridLogMessage
            << "\n========================================" << std::endl;
  std::cout << GridLogMessage << "MODULE: MUtilities::GaugeSinglePrecisionCast"
            << std::endl;
  std::cout << GridLogMessage
            << "========================================" << std::endl;

  LatticeGaugeFieldF U_fat_f(UGridF);
  LatticeGaugeFieldF U_long_f(UGridF);

  std::cout << GridLogMessage << "Casting fat links to single precision"
            << std::endl;
  precisionChange(U_fat_f, U_fat);

  std::cout << GridLogMessage << "Casting long links to single precision"
            << std::endl;
  precisionChange(U_long_f, U_long);

  ImplParams implParams;

  auto makeAction = [&UGrid, &UrbGrid, &UGridF, &UrbGridF, &U_long, &U_fat,
                     &U_long_f, &U_fat_f, &implParams](
                        auto &action, ImprovedStaggeredPar actionPar) {
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

  bool hasEigs = inputParams.epack.type != EpackPar::EpackType::undef;
  bool hasSources = inputParams.sources.size() > 0;

  std::shared_ptr<EigenPack<FermionFieldD>> epack;

  if (hasEigs) {
    // IRL action parameters
    auto &epackPar = inputParams.epack;

    epack = std::make_shared<EigenPack<FermionFieldD>>();
    auto &actionParIRL = epackPar.action;

    // ========================================================================
    // MODULE: MSolver::StagFermionIRL (Run IRL eigensolver)
    // ========================================================================
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

    std::shared_ptr<FermionOpD> stagMatIRL;
    makeAction(stagMatIRL, actionParIRL);

    // Create operators for IRL if needed
    SchurStaggeredOperator<FermionOpD, FermionFieldD> hermOpIRL(*stagMatIRL);
    Chebyshev<FermionFieldD> Cheby(
        lanczosPar.Cheby.alpha, lanczosPar.Cheby.beta, lanczosPar.Cheby.Npoly);

    FunctionHermOp<FermionFieldD> OpCheby(Cheby, hermOpIRL);
    PlainHermOp<FermionFieldD> Op(hermOpIRL);

    ImplicitlyRestartedLanczos<FermionFieldD> IRL(OpCheby, Op, Nstop, Nk, Nm,
                                                  resid, MaxIt);

    FermionFieldD src(UrbGrid);
    int cb = epackPar.checker;

    std::cout << GridLogMessage
              << "Generating random source (checkerboard = " << epackPar.checker
              << ")" << std::endl;
    FermionFieldD gauss(UGrid);
    std::string seed = getSeed(inputParams, epackPar.seed);
    rng.SeedUniqueString(seed);
    gaussian(rng, gauss);
    pickCheckerboard(cb, src, gauss);

    epack->resize(epackPar.size, UrbGrid);

    if (epackPar.type == EpackPar::EpackType::solve) {
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
      // Load eigenpack
      std::cout << GridLogMessage << "Loading eigenpack from " << epackPar.file
                << std::endl;
      assert(!epackPar.file.empty());
      epack->read(epackPar.file, epackPar.multiFile, traj);
      epack->eval.resize(epackPar.size);
    }

    if (!epackPar.evalSave.empty()) {
      std::cout << GridLogMessage << "Saving eigenvalues to "
                << epackPar.evalSave << std::endl;
      saveResult(UGrid, epackPar.evalSave, "evals", epack->eval, inputParams);
    }

    std::cout << GridLogMessage << "Setting checkerboard of eigenvectors to "
              << (cb == Even ? "Even" : "Odd") << std::endl;
    for (auto &e : epack->evec) {
      e.Checkerboard() = cb;
    }
  }

  std::shared_ptr<FermionOpD> stagMatMassive;
  RealD solverMass;
  {
    std::shared_ptr<FermionOpF> stagMatMassiveF;
    std::shared_ptr<FermionFieldD> fermOut;
    std::shared_ptr<FermionFieldD> fermIn;
    std::shared_ptr<FermionFieldD> fermGuess;

    // Create Action objects and temporary fields for solves
    if (hasSources) {
      fermOut = std::make_shared<FermionFieldD>(UGrid);
      fermIn = std::make_shared<FermionFieldD>(UGrid);
      fermGuess = std::make_shared<FermionFieldD>(UGrid);

      makeAction(stagMatMassive, inputParams.mpcg.action);
      makeAction(stagMatMassiveF, inputParams.mpcg.action);
      solverMass = 2.0 * inputParams.mpcg.action.mass;
    }

    using SolverFunc = std::function<void()>;
    SolverFunc lmaSolver, lmaSolverSubtract;
    SolverFunc mpcgSolver, mpcgSolverSubtract;

    // Create LMA Solver lambda functions
    if (hasEigs && hasSources) {

      // Extract LMA parameters
      unsigned int eigStart =
          inputParams.lma.eigStart;      // Start from first eigenvalue
      int nEigs = inputParams.lma.nEigs; // Use all eigenvalues by default
      bool projector = inputParams.lma.projector; // Use accelerated solver mode

      if (nEigs < 1) {
        nEigs = epack->evec.size();
      }

      // Validate eigenvalue ranges
      if (eigStart > static_cast<unsigned int>(nEigs) ||
          eigStart > epack->evec.size() ||
          nEigs - eigStart > static_cast<int>(epack->evec.size()) - eigStart) {
        std::cerr << "ERROR: Requested eigs (eigStart and nEigs) out of bounds"
                  << std::endl;
        exit(1);
      }

      std::cout << GridLogMessage << "Setting up low mode projector"
                << std::endl;
      std::cout << GridLogMessage << "  eigStart = " << eigStart << std::endl;
      std::cout << GridLogMessage << "  nEigs = " << nEigs << std::endl;
      std::cout << GridLogMessage
                << "  projector = " << (projector ? "true" : "false")
                << std::endl;

      // Create temporary fields for LMA solver (heap allocated for use in
      // returned lambda)
      auto rbFerm = std::make_shared<FermionFieldD>(UrbGrid);
      auto rbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
      auto MrbFermNeg = std::make_shared<FermionFieldD>(UrbGrid);
      auto rbTemp = std::make_shared<FermionFieldD>(UrbGrid);
      auto rbTempNeg = std::make_shared<FermionFieldD>(UrbGrid);

      // Lambda to create the LMA solver function
      auto makeLMASolver = [&stagMatMassive, epack, solverMass, projector,
                            eigStart, nEigs, rbFerm, rbFermNeg, MrbFermNeg,
                            rbTemp, rbTempNeg, fermOut, fermIn](bool subGuess) {
        return [&stagMatMassive, epack, subGuess, solverMass, projector,
                eigStart, nEigs, rbFerm, rbFermNeg, MrbFermNeg, rbTemp,
                rbTempNeg, fermOut, fermIn]() {
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

          // Extract checkerboard components
          pickCheckerboard(cb, *rbFerm, *fermIn);
          pickCheckerboard(cbNeg, *rbFermNeg, *fermIn);

          // Apply M_eooe^dagger
          stagMatMassive->MeooeDag(*rbFermNeg, *MrbFermNeg);

          // Project onto low modes
          for (int k = (eigStart + nEigs - 1); k >= static_cast<int>(eigStart);
               k--) {
            const FermionFieldD &e = epack->evec[k];

            const RealD lam_DD = epack->eval[k];
            const RealD invlam_DD = 1.0 / lam_DD;
            const RealD invmag = 1.0 / (solverMass * solverMass + lam_DD);

            if (!projector) {
              // Accelerated solver mode
              const ComplexD ip =
                  TensorRemove(innerProduct(e, *rbFerm)) * invmag;
              const ComplexD ipNeg =
                  TensorRemove(innerProduct(e, *MrbFermNeg)) * invmag;
              axpy(*rbTemp, solverMass * ip + ipNeg, e, *rbTemp);
              axpy(*rbTempNeg, solverMass * ipNeg * invlam_DD - ip, e,
                   *rbTempNeg);
            } else {
              // Pure projector mode
              const ComplexD ip = TensorRemove(innerProduct(e, *rbFerm));
              const ComplexD ipNeg = TensorRemove(innerProduct(e, *MrbFermNeg));
              axpy(*rbTemp, ip, e, *rbTemp);
              axpy(*rbTempNeg, ipNeg * invlam_DD, e, *rbTempNeg);
            }
          }

          // Apply M_eooe
          stagMatMassive->Meooe(*rbTempNeg, *rbFermNeg);

          // Reconstruct full field
          setCheckerboard(*fermOut, *rbTemp);
          setCheckerboard(*fermOut, *rbFermNeg);

          *fermOut *= norm;

          if (subGuess) {
            if (projector) {
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

      // Create the normal and subtract solvers
      lmaSolver = makeLMASolver(false);
      lmaSolverSubtract = makeLMASolver(true);

      std::cout << GridLogMessage << "Low mode projector setup complete"
                << std::endl;
    }

    // Create Mixed Precision CG Solver lambda functions
    if (hasSources) {

      auto &mpcgPar = inputParams.mpcg;

      std::cout << GridLogMessage << "Setting up mixed-precision CG solver"
                << std::endl;
      std::cout << GridLogMessage
                << "  Inner action (single precision): mass = "
                << mpcgPar.action.mass << std::endl;
      std::cout << GridLogMessage
                << "  Outer action (double precision): mass = "
                << mpcgPar.action.mass << std::endl;
      std::cout << GridLogMessage << "  Residual: " << mpcgPar.residual
                << std::endl;
      std::cout << GridLogMessage
                << "  Max inner iterations: " << mpcgPar.maxInnerIteration
                << std::endl;
      std::cout << GridLogMessage
                << "  Max outer iterations: " << mpcgPar.maxOuterIteration
                << std::endl;

      // Create hermitian operators for mixed precision solve
      auto hermOpOuter =
          std::make_shared<MdagMLinearOperator<FermionOpD, FermionFieldD>>(
              *stagMatMassive);
      auto hermOpInner =
          std::make_shared<MdagMLinearOperator<FermionOpF, FermionFieldF>>(
              *stagMatMassiveF);
      auto temp = std::make_shared<FermionFieldD>(UGrid);

      std::cout << GridLogMessage << "Mixed precision CG solver created"
                << std::endl;

      // Lambda to create MPCG solver functions
      auto makeMPCGSolver = [stagMatMassive, fermOut, fermIn, fermGuess, temp,
                             hermOpInner, hermOpOuter, &UGridF,
                             &mpcgPar](bool subGuess) {
        return [stagMatMassive, subGuess, fermOut, fermIn, fermGuess, temp,
                hermOpInner, hermOpOuter, &UGridF, &mpcgPar]() {
          MixedPrecisionConjugateGradient<FermionFieldD, FermionFieldF> mpcg(
              mpcgPar.residual, mpcgPar.maxInnerIteration,
              mpcgPar.maxOuterIteration, UGridF, *hermOpInner, *hermOpOuter);

          // Compute initial guess via outer guesser
          if (fermGuess != nullptr) {
            *fermOut = *fermGuess;
          } else {
            *fermOut = 1.0;
          }

          ZeroGuesser<FermionFieldF> iguesserDefault;
          mpcg.useGuesser(iguesserDefault);
          // Create temporary for residual
          *temp = Zero();
          stagMatMassive->Mdag(*fermIn, *temp);

          // Run MPCG solver on M^dag*M*x = M^dag*source
          mpcg(*temp, *fermOut);

          RealD nsol = norm2(*fermOut);
          // Compute residual: r = M*sol - source
          stagMatMassive->M(*fermOut, *temp);
          RealD nMsol = norm2(*temp);
          *temp = *temp - *fermIn;

          // Compute relative residual
          RealD ns = norm2(*fermIn);
          RealD nr = norm2(*temp);
          RealD relres = (ns > 0.0) ? std::sqrt(nr / ns) : 0.0;

          std::cout << GridLogMessage
                    << "MPCG: Final true residual = " << relres << std::endl;

          if (subGuess && fermGuess != nullptr) {
            // For subtraction mode, compute residual vector
            *fermOut = *fermOut - *fermGuess;
          }
        };
      };

      // Create the MPCG solvers (normal and subtract)
      mpcgSolver = makeMPCGSolver(false);
      mpcgSolverSubtract = makeMPCGSolver(true);

      std::cout << GridLogMessage << "MPCG solvers created" << std::endl;
    }

    for (auto &sourcePar : inputParams.sources) {
      // Random wall source parameters (from XML) - Color-diagonal only
      unsigned int tStep = sourcePar.tStep;
      unsigned int t0 = sourcePar.t0;
      unsigned int nSrc = sourcePar.nSrc;

      std::cout << GridLogMessage
                << "Setting up random wall sources (color-diagonal)"
                << std::endl;
      std::cout << GridLogMessage << "  tStep = " << tStep << std::endl;
      std::cout << GridLogMessage << "  t0 = " << t0 << std::endl;
      std::cout << GridLogMessage << "  nSrc = " << nSrc << std::endl;

      if (t0 >= tStep) {
        std::cerr << "ERROR: t0 >= tStep" << std::endl;
        exit(1);
      }
      TimeDilutedNoise<FImpl> noise(UGrid, nSrc);
      std::string seed = getSeed(inputParams, sourcePar.seed);
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
      std::map<std::string, StagGamma::SpinTastePair> solveGammas;
      for (auto &corrPar : inputParams.corr) {

        auto quarkGammaKeys = StagGamma::ParseSpinTaste(corrPar.quark.gammas);
        auto quarkGammaVals = StagGamma::ParseSpinTaste(corrPar.quark.gammas,
                                                        corrPar.quark.applyG5);
        GRID_ASSERT(!quarkGammaKeys.empty());

        auto antiquarkGammaKeys =
            StagGamma::ParseSpinTaste(corrPar.antiquark.gammas);
        auto antiquarkGammaVals = StagGamma::ParseSpinTaste(
            corrPar.antiquark.gammas, corrPar.antiquark.applyG5);
        GRID_ASSERT(antiquarkGammaKeys.size() == 1);
        std::string antiquarkGammaName =
            StagGamma::GetName(antiquarkGammaKeys[0]);
        StagGamma::SpinTastePair antiquarkSpinTaste = antiquarkGammaVals[0];

        auto sinkGammaKeys = StagGamma::ParseSpinTaste(corrPar.sink.gammas);
        auto sinkGammaVals = StagGamma::ParseSpinTaste(corrPar.sink.gammas,
                                                       corrPar.sink.applyG5);
        GRID_ASSERT(sinkGammaKeys.size() == quarkGammaKeys.size());

        for (size_t i = 0; i < quarkGammaKeys.size(); ++i) {
          solveGammas.emplace(StagGamma::GetName(quarkGammaKeys[i]),
                              quarkGammaVals[i]);
        }
        solveGammas.emplace(antiquarkGammaName, antiquarkSpinTaste);
      }
      for (int i = 0; i < nSrc; i++) {
        for (int j = 0; j < nSlices; j++) {
          int timeSlice = j * tStep + t0;
          int offset = i * Nt + j * tStep + t0;

          randomWallSource = noise.getProp(offset);
          std::cout << GridLogMessage << "Random wall sources setup complete"
                    << std::endl;

          std::map<std::string, PropagatorFieldD> lmaProp;
          std::map<std::string, PropagatorFieldD> mpcgProp;
          for (auto &corrPar : inputParams.corr) {
            // Initialize meson results for all gamma pairs
            std::cout << GridLogMessage << "Setting up meson contraction"
                      << std::endl;

            auto antiquarkGammaKeys =
                StagGamma::ParseSpinTaste(corrPar.antiquark.gammas);
            auto quarkGammaKeys =
                StagGamma::ParseSpinTaste(corrPar.quark.gammas);
            auto sinkGammaKeys = StagGamma::ParseSpinTaste(corrPar.sink.gammas);
            std::string antiquarkGammaName =
                StagGamma::GetName(antiquarkGammaKeys[0]);

            std::vector<MesonResult> mesonResults(quarkGammaKeys.size());
            for (size_t i = 0; i < quarkGammaKeys.size(); ++i) {
              std::string quarkGammaName =
                  StagGamma::GetName(quarkGammaKeys[i]);
              std::string sinkGammaName = StagGamma::GetName(sinkGammaKeys[i]);

              mesonResults[i].sourceGamma = quarkGammaName;
              mesonResults[i].sinkGamma = sinkGammaName;
              mesonResults[i].corr.resize(Nt, 0.0);
              mesonResults[i].srcCorrs.resize(nVecs,
                                              std::vector<Complex>(Nt, 0.0));
              mesonResults[i].scaling = nVecs;
            }

            PropagatorFieldD gammaProp(UGrid);
            auto doSolves =
                [&quarkGammaKeys, &antiquarkGammaKeys, &solveGammas, &gammaProp,
                 &randomWallSource, fermIn, fermOut, fermGuess, &U,
                 UGrid](std::map<std::string, PropagatorFieldD> &propMap,
                        SolverFunc solver) {
                  StagGamma gamma;
                  gamma.setGaugeField(U);
                  // Create StagGamma operator
                  for (const auto &gammaPair : solveGammas) {
                    auto antiquarkIt = std::find_if(
                        antiquarkGammaKeys.begin(), antiquarkGammaKeys.end(),
                        [&](const auto &p) {
                          return StagGamma::GetName(p) == gammaPair.first;
                        });
                    auto quarkIt = std::find_if(
                        quarkGammaKeys.begin(), quarkGammaKeys.end(),
                        [&](const auto &p) {
                          return StagGamma::GetName(p) == gammaPair.first;
                        });
                    if ((antiquarkIt != antiquarkGammaKeys.end() ||
                         quarkIt != quarkGammaKeys.end()) &&
                        propMap.find(gammaPair.first) == propMap.end()) {
                      std::cout << GridLogMessage
                                << "Solving gamma: " << gammaPair.first
                                << std::endl;
                      propMap.emplace(gammaPair.first, UGrid);
                      propMap.at(gammaPair.first) = Zero();

                      *fermIn = Zero();
                      *fermOut = Zero();
                      *fermGuess = Zero();
                      gamma.setSpinTaste(gammaPair.second);

                      gammaProp = Zero();
                      gamma(gammaProp, randomWallSource);

                      for (int c = 0; c < 3; c++) {
                        PropToFerm<FImpl>(*fermIn, gammaProp, c);
                        solver();
                        FermToProp<FImpl>(propMap.at(gammaPair.first), *fermOut,
                                          c);
                      }
                    }
                  }
                };

            if (hasEigs) {
              std::cout << GridLogMessage << "Solving with LMA solver"
                        << std::endl;
              doSolves(lmaProp, lmaSolver);
            }
            if (!corrPar.amaOutput.empty()) {
              std::cout << GridLogMessage << "Solving with MPCG solver"
                        << std::endl;
              doSolves(mpcgProp, mpcgSolver);
            }

            auto doContractions =
                [&gammaProp, &solveGammas, &mesonResults, t0, Nt,
                 &antiquarkGammaName, &U,
                 UGrid](std::map<std::string, PropagatorFieldD> &propMap) {
                  StagGamma gamma;
                  gamma.setGaugeField(U);
                  // Accumulate meson contraction results for this source
                  for (size_t i = 0; i < mesonResults.size(); ++i) {
                    std::string quarkGammaName = mesonResults[i].sourceGamma;
                    std::string sinkGammaName = mesonResults[i].sinkGamma;
                    std::cout << GridLogMessage
                              << "Contracting source gamma: " << quarkGammaName
                              << ", sink gamma: " << sinkGammaName << std::endl;
                    gamma.setSpinTaste(solveGammas.at(sinkGammaName));

                    PropagatorFieldD prod(UGrid);
                    gamma(gammaProp, propMap.at(quarkGammaName));
                    prod = propMap.at(antiquarkGammaName) * adj(gammaProp);

                    std::vector<TComplex> buf;
                    LatticeComplexD slicedTrace = trace(prod);
                    sliceSum(slicedTrace, buf, Tp);
                    int sliceOffset = t0;
                    for (int t = 0; t < Nt; ++t) {
                      Complex ct = TensorRemove(buf[sliceOffset]);
                      mesonResults[i].srcCorrs[0][t] = ct;
                      sliceOffset = mod(sliceOffset + 1, Nt);
                    }
                  }

                  // Compute averaged correlators from all sources
                  for (size_t i = 0; i < mesonResults.size(); ++i) {
                    for (int t = 0; t < Nt; ++t) {
                      mesonResults[i].corr[t] = 0.0;
                      for (int j = 0; j < mesonResults[i].scaling; j++) {
                        mesonResults[i].corr[t] +=
                            mesonResults[i].srcCorrs[j][t];
                      }
                      mesonResults[i].corr[t] /= mesonResults[i].scaling;
                    }
                  }
                };

            if (!corrPar.lmaOutput.empty()) {
              std::cout << GridLogMessage << "Contracting LMA propagators"
                        << std::endl;
              doContractions(lmaProp);
              saveResult(UGrid, corrPar.lmaOutput, "meson", mesonResults,
                         inputParams, t0);
            }
            if (!corrPar.amaOutput.empty()) {
              std::cout << GridLogMessage
                        << "Contracting AMA (MPCG) propagators" << std::endl;
              doContractions(mpcgProp);
              saveResult(UGrid, corrPar.amaOutput, "meson", mesonResults,
                         inputParams, t0);
            }
          }
        }
      }
    }
  }
  auto &a2aPar = inputParams.a2a;

  if (hasEigs) {
    makeAction(stagMatMassive, a2aPar.action);
    RealD a2aMass = 2.0 * a2aPar.action.mass;
    int nBlock = a2aPar.block;

    std::cout << GridLogMessage
              << "Setting up all-to-all meson field construction" << std::endl;
    std::cout << GridLogMessage << "  Block size: " << nBlock << std::endl;
    std::cout << GridLogMessage << "  Output: " << a2aPar.output << std::endl;

    std::vector<StagGamma::SpinTastePair> a2aGammas, gammaComms, gammaLocal;
    std::vector<std::vector<Real>> mom;
    a2aGammas = StagGamma::ParseSpinTaste(a2aPar.spinTaste.gammas,
                                          a2aPar.spinTaste.applyG5);

    gammaComms.clear();
    gammaLocal.clear();

    StagGamma spinTaste;
    for (auto &g : a2aGammas) {
      spinTaste.setSpinTaste(g);

      if (spinTaste._spin ^ spinTaste._taste) {
        gammaComms.push_back(g);
      } else {
        gammaLocal.push_back(g);
      }
    }

    mom.clear();

    for (auto &pstr : a2aPar.mom) {
      auto p = strToVec<Real>(pstr);

      // if (p.size() != env().getNd() - 1) {
      //   HADRONS_ERROR(Size, "Momentum has " + std::to_string(p.size())
      //   +
      //                           " components instead of " +
      //                           std::to_string(env().getNd() - 1));
      // }
      mom.push_back(p);
    }
    int nmom = mom.size();
    bool allzero = true;
    if (a2aPar.mom.size() == 1) {
      for (auto p : mom[0]) {
        if (p != 0)
          allzero = false;
      }
    }
    if (allzero)
      nmom = 0;

    // TODO: Implement non-zero momentum
    // envCache(std::vector<LatticeComplexD>, _momphName, 1, nmom, UGrid);
    // envTmpLat(LatticeComplexD, "coor");
    std::shared_ptr<std::vector<LatticeComplexD>> ph =
        make_shared<std::vector<LatticeComplexD>>(0, UGrid);

    // TODO: Add TimerArray to last parameter here
    std::shared_ptr<Computation> computationLocal =
        std::make_shared<Computation>(UGrid, Tdir, mom.size(),
                                      gammaLocal.size(), nBlock);

    std::shared_ptr<Computation> computationComms =
        std::make_shared<Computation>(UGrid, Tdir, mom.size(),
                                      gammaComms.size(), nBlock);

    // TODO: Set left and right fields with method call
    std::shared_ptr<std::vector<FermionFieldD>> left, right;
    left = std::make_shared<std::vector<FermionFieldD>>(0, UGrid);
    right = std::make_shared<std::vector<FermionFieldD>>(0, UGrid);
    int N_i = left->size();
    int N_j = right->size();

    if (hasEigs) {
      if (N_j != 0 && N_i == 0) {
        N_i += 2 * epack->evec.size();
      } else if (N_i != 0 && N_j == 0) {
        N_j += 2 * epack->evec.size();
      } else {
        N_i += 2 * epack->evec.size();
        N_j += 2 * epack->evec.size();
      }
    }
    /*if (N_i < block || N_j < block)
    {
        HADRONS_ERROR(Range, "blockSize must not exceed size of input
    vector.");
    }*/

    std::cout << GridLogMessage << "Computing all-to-all meson fields"
              << std::endl;

    std::cout << GridLogMessage << "Momenta:" << std::endl;

    for (auto &p : mom) {
      std::cout << GridLogMessage << "  " << p << std::endl;
    }

    std::cout << GridLogMessage << "Spin bilinears:" << std::endl;

    for (auto &g : a2aGammas) {
      std::cout << GridLogMessage << "  " << StagGamma::GetName(g) << std::endl;
    }

    std::cout << GridLogMessage << "Meson field size: " << Nt << "*" << N_i
              << "*" << N_j << " (filesize "
              << sizeString(Nt * N_i * N_j * sizeof(HADRONS_A2AM_IO_TYPE))
              << "/momentum/bilinear)" << std::endl;

    // TODO: Implement non-zero momentum
    // auto &ph = envGet(std::vector<LatticeComplexD>, _momphName);
    // startTimer("Momentum phases");
    // for (unsigned int j = 0; j < ph.size(); ++j) {
    //   Complex i(0.0, 1.0);
    //   std::vector<Real> p;
    //   envGetTmp(LatticeComplexD, coor);
    //   ph[j] = Zero();
    //   for (unsigned int mu = 0; mu < _mom[j].size(); mu++) {
    //     LatticeCoordinate(coor, mu);
    //     ph[j] = ph[j] + (_mom[j][mu] / env().getDim(mu)) * coor;
    //   }
    //   ph[j] = exp((Real)(2 * M_PI) * i * ph[j]);
    // }
    // stopTimer("Momentum phases");

    MesonFieldData<FImpl, EigenPack<FermionFieldD>> mesonData(
        stagMatMassive.get(), a2aMass, mom, a2aPar.output, traj);
    mesonData.setLeft(*left);
    mesonData.setRight(*right);
    if (hasEigs)
      mesonData.setEpack(*epack);

    Kernel kernel(UGrid);
    int orthogDir = Tdir;

    if (gammaLocal.size() > 0) {
      mesonData.setGammas(gammaLocal);
      kernel.setWorker(UGrid, *ph, gammaLocal, orthogDir);
      computationLocal->execute(kernel, mesonData);
    }
    if (gammaComms.size() > 0) {
      mesonData.setGammas(gammaComms);
      kernel.setWorker(UGrid, *ph, gammaComms, orthogDir, &U);
      computationComms->execute(kernel, mesonData);
    }
    std::cout << GridLogMessage
              << "All-to-all meson field construction complete" << std::endl;
  }
  Grid_finalize();
}
