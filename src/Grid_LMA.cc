#include <A2AMesonField.h>
#include <Epack.h>
#include <Grid/Grid.h>
#include <HighModeCorr.h>
#include <IO.h>

using namespace std;
using namespace Grid;

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  const int Ls = 1;

  typedef ImprovedStaggeredFermionD FermionOpD;
  typedef ImprovedStaggeredFermionF FermionOpF;
  typedef typename ImprovedStaggeredFermionD::ImplParams ImplParams;
  typedef typename ImprovedStaggeredFermionD::Impl_t FImpl;
  typedef typename ImprovedStaggeredFermionD::FermionField FermionFieldD;

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

  switch (inputParams.gauge.type) {
  case GaugePar::GaugeType::free:
    SU<Nc>::ColdConfiguration(U);
    SU<Nc>::ColdConfiguration(U_fat);
    SU<Nc>::ColdConfiguration(U_long);
    break;
  case GaugePar::GaugeType::file: {
    std::string file_fat =
        inputParams.gauge.fatlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading fat links from " << file_fat
              << std::endl;
    IR.open(file_fat);
    IR.readConfiguration(U_fat, header);
    IR.close();

    std::string file_long =
        inputParams.gauge.longlink + "." + std::to_string(traj);
    std::cout << GridLogMessage << "Loading long links from " << file_long
              << std::endl;
    IR.open(file_long);
    IR.readConfiguration(U_long, header);
    IR.close();

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
  // MODULE: MUtilities::GaugeSinglePrecisionCast
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

  // ========================================================================
  // Eigenpack load/solve
  // ========================================================================
  bool hasEigs = inputParams.epack.type != EpackPar::EpackType::undef;

  std::shared_ptr<EigenPack<FermionFieldD>> epack;
  if (hasEigs) {
    epack = loadOrSolveEigenpack<FermionOpD, FermionFieldD>(
        inputParams.epack, inputParams, UGrid, UrbGrid, rng, U_fat, U_long,
        implParams, traj);
  }

  // ========================================================================
  // High-mode correlator generation
  // ========================================================================
  bool hasSources =
      inputParams.sources.size() > 0 && !inputParams.highModeActions.empty();

  if (hasSources) {
    computeHighModeCorrelators<FImpl, FermionOpD, FermionOpF>(
        inputParams, UGrid, UrbGrid, UGridF, UrbGridF, rng, U, U_fat, U_long,
        U_fat_f, U_long_f, implParams, epack);
  }

  // ========================================================================
  // A2A meson field generation
  // ========================================================================
  if (!inputParams.a2a.empty()) {
    computeA2AMesonFields<FImpl, FermionOpD>(inputParams, UGrid, UrbGrid, U,
                                             U_fat, U_long, implParams, epack,
                                             traj);
  }

  Grid_finalize();
}
