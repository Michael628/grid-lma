#ifndef FMGRID_IO_H
#define FMGRID_IO_H

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/LocalCoherenceLanczos.h>
#include <StagGamma.h>

NAMESPACE_BEGIN(Grid);

// type aliases
#define BASIC_TYPE_ALIASES(Impl, suffix)                                       \
  typedef typename Impl::Field ScalarField##suffix;                            \
  typedef typename Impl::PropagatorField PropagatorField##suffix;              \
  typedef typename Impl::SitePropagator::scalar_object SitePropagator##suffix; \
  typedef typename Impl::ComplexField ComplexField##suffix;                    \
  typedef std::vector<SitePropagator##suffix> SlicedPropagator##suffix;        \
  typedef std::vector<                                                         \
      typename ComplexField##suffix::vector_object::scalar_object>             \
      SlicedComplex##suffix;

#define FERM_TYPE_ALIASES(FImpl, suffix)                                       \
  BASIC_TYPE_ALIASES(FImpl, suffix);                                           \
  typedef FermionOperator<FImpl> FMat##suffix;                                 \
  typedef typename FImpl::FermionField FermionField##suffix;                   \
  typedef typename FImpl::GaugeField GaugeField##suffix;                       \
  typedef typename FImpl::DoubledGaugeField DoubledGaugeField##suffix;         \
  typedef LinearOperatorBase<FermionField##suffix> FBaseOp##suffix;            \
  typedef NonHermitianLinearOperator<FMat##suffix, FermionField##suffix>       \
      FOp##suffix;                                                             \
  typedef MdagMLinearOperator<FMat##suffix, FermionField##suffix>              \
      FHermOp##suffix;                                                         \
  typedef Lattice<iSpinMatrix<typename FImpl::Simd>> SpinMatrixField##suffix;  \
  typedef Lattice<iColourVector<typename FImpl::Simd>>                         \
      ColourVectorField##suffix;                                               \
  typedef Lattice<iColourMatrix<typename FImpl::Simd>>                         \
      ColourMatrixField##suffix;                                               \
  typedef typename PropagatorField##suffix::vector_object::scalar_object       \
      SpinColourMatrixScalar##suffix;                                          \
  typedef Lattice<iSpinColourSpinColourMatrix<typename FImpl::Simd>>           \
      SpinColourSpinColourMatrixField##suffix;

#ifdef HAVE_HDF5
typedef Hdf5Reader ResultReader;
typedef Hdf5Writer ResultWriter;
#else
typedef XmlReader ResultReader;
typedef XmlWriter ResultWriter;
#endif

// ============================================================================
// MAction Module Parameter Classes
// ============================================================================
class ImprovedStaggeredPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(ImprovedStaggeredPar, double, mass, double,
                                  c1, double, c2, double, tad, std::string,
                                  boundary, std::string, twist);

  std::string parString(void) const {
    XmlWriter writer("", "");
    write(writer, "ImprovedStaggeredPar", *this);
    return writer.string();
  }
};

// ============================================================================
// MSolver Module Parameter Classes
// ============================================================================
class ImplicitlyRestartedLanczosPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(ImplicitlyRestartedLanczosPar, LanczosParams,
                                  lanczosParams);

  std::string parString(void) const {
    XmlWriter writer("", "");
    write(writer, "ImplicitlyRestartedLanczosPar", *this);
    return writer.string();
  }
};

class LowModeProjPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(LowModeProjPar, bool, projector, unsigned int,
                                  eigStart, int, nEigs);
};

class MixedPrecisionCGPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MixedPrecisionCGPar, ImprovedStaggeredPar,
                                  action, unsigned int, maxInnerIteration,
                                  unsigned int, maxOuterIteration, double,
                                  residual);
};

// ============================================================================
// MSink Module Parameter Classes
// ============================================================================
class PointPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(PointPar, std::string, mom);
};

// ============================================================================
// MSource Module Parameter Classes
// ============================================================================
class RandomWallPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(RandomWallPar, unsigned int, tStep,
                                  unsigned int, t0, unsigned int, nSrc,
                                  std::string, seed);
  RandomWallPar() : seed("noise") {}
};

// ============================================================================
// MContraction Module Parameter Classes
// ============================================================================
class ContractionPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(ContractionPar, SpinTasteParams, quark,
                                  SpinTasteParams, antiquark, SpinTasteParams,
                                  sink, std::string, lmaOutput, std::string,
                                  amaOutput);
};

class MesonResult : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonResult, std::string, sourceGamma,
                                  std::string, sinkGamma, std::vector<Complex>,
                                  corr, std::vector<std::vector<Complex>>,
                                  srcCorrs, std::vector<Integer>, timeShifts,
                                  Real, scaling);
};

class MesonFieldMetadata : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFieldMetadata, std::vector<RealF>,
                                  momentum, StagGamma::StagAlgebra, gamma_spin,
                                  StagGamma::StagAlgebra, gamma_taste);

  MesonFieldMetadata()
      : momentum{}, gamma_spin(StagGamma::StagAlgebra::undef),
        gamma_taste(StagGamma::StagAlgebra::undef) {}
};

class MesonFieldPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFieldPar, int, block,
                                  ImprovedStaggeredPar, action, std::string,
                                  output, SpinTasteParams, spinTaste,
                                  std::vector<std::string>, mom);
};

// ============================================================================
// High-level Parameter Classes
// ============================================================================
class GaugePar : Serializable {
public:
  GRID_SERIALIZABLE_ENUM(GaugeType, undef, free, 0, file, 1, hot, 2);

  GRID_SERIALIZABLE_CLASS_MEMBERS(GaugePar, std::string, link, std::string,
                                  fatlink, std::string, longlink, GaugeType,
                                  type);
};

class EpackPar : Serializable {
public:
  GRID_SERIALIZABLE_ENUM(CheckerType, undef, even, 0, odd, 1);
  GRID_SERIALIZABLE_ENUM(EpackType, undef, load, 0, solve, 1);
  GRID_SERIALIZABLE_CLASS_MEMBERS(EpackPar, ImprovedStaggeredPar, action,
                                  ImplicitlyRestartedLanczosPar, irl,
                                  std::string, evalSave, EpackType, type,
                                  unsigned int, size, std::string, file, bool,
                                  multiFile, CheckerType, checker, std::string,
                                  seed);
  EpackPar() : seed("epack"), checker(CheckerType::odd) {}
};

class GlobalPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(GlobalPar, GaugePar, gauge, EpackPar, epack,
                                  LowModeProjPar, lma, MixedPrecisionCGPar,
                                  mpcg, std::vector<ContractionPar>, corr,
                                  MesonFieldPar, a2a,
                                  std::vector<RandomWallPar>, sources,
                                  std::string, series, std::string, runSeed,
                                  unsigned int, trajectory);
};

int mkdir(const std::string dirName);
std::string dirname(const std::string &s);
void makeFileDir(const std::string filename, GridBase *g);
std::string resultFilename(const std::string stem, const GlobalPar &inputParams,
                           const std::string ext, bool includeSeries);
std::string getSeed(GlobalPar &inputParams, std::string seedSuffix = "");
template <typename T>
void saveResult(GridBase *grid, const std::string stem, const std::string name,
                const T &result, const GlobalPar &inputParams,
                const std::string ext = "h5", bool includSeries = true) {
  if (grid->IsBoss() and !stem.empty()) {
    makeFileDir(stem, grid);
    {
      ResultWriter writer(resultFilename(stem, inputParams, ext, includSeries));
      write(writer, name, result);
    }
  }
}
template <typename T>
void saveResult(GridBase *grid, const std::string stem, const std::string name,
                const T &result, const GlobalPar &inputParams,
                const int tsource, const std::string ext = "h5",
                bool includeSeries = true) {
  std::string stem_with_tsource = stem + "_t" + std::to_string(tsource);
  saveResult(grid, stem_with_tsource, name, result, inputParams, ext,
             includeSeries);
}

NAMESPACE_END(Grid);

#endif
