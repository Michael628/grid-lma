#ifndef FMGRID_MESONFIELD_KERNEL_H
#define FMGRID_MESONFIELD_KERNEL_H

#include <A2AMatrix.h>
#include <Eigenpack.h>
#include <IO.h>
#include <StagGamma.h>
#include <a2a/A2AWorker.h>

NAMESPACE_BEGIN(Grid);

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

NAMESPACE_END(Grid);

#endif
