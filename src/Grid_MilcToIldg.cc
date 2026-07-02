// Load a MILC v5 gauge configuration, HISQ-smear it (fat7 + project + asqtad),
// and write the thin, fat, and long gauge fields out as ILDG configurations.
//
// Build (from build-scalar/):
//   make grid_milc_to_ildg
// Run:
//   ./grid_milc_to_ildg params.xml --grid 8.8.8.8

#include <Grid/Grid.h>
#include <Grid/qcd/utils/HighlyImprovedStaggeredFermionImpl.h>
#include <GridMilc/GridMilc.h>
#include <IO.h>

using namespace std;
using namespace Grid;

class MilcToIldgPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MilcToIldgPar, std::string, milcFile,
                                  bool, exitOnChecksumMismatch,
                                  std::string, boundary,
                                  std::string, gaugeStem,
                                  std::string, gaugeFatStem,
                                  std::string, gaugeLongStem,
                                  std::string, ensembleLabel,
                                  unsigned int, trajectory);
  MilcToIldgPar() : exitOnChecksumMismatch(false), trajectory(0) {}
};

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  typedef PeriodicGimplD Gimpl;

  std::string paramFile = argv[1];
  XmlReader reader(paramFile, false, "grid");

  MilcToIldgPar par;
  read(reader, "parameters", par);

  auto nsimd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  GridCartesian *UGrid =
      SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), nsimd, GridDefaultMpi());

  // ==========================================================================
  // Load MILC configuration
  // ==========================================================================
  std::string milcFileName = par.milcFile + "." + std::to_string(par.trajectory);
  std::cout << GridLogMessage << "========================================"
            << std::endl;
  std::cout << GridLogMessage << "Loading MILC configuration from file '"
            << milcFileName << "'" << std::endl;
  std::cout << GridLogMessage << "========================================"
            << std::endl;

  LatticeGaugeFieldD U(UGrid);
  MilcHeader header;
  MilcIO::readConfiguration(U, header, milcFileName, par.exitOnChecksumMismatch);

  // ==========================================================================
  // HISQ smear (fat7 + project + asqtad, cf. Hadrons MGauge::HISQSmear)
  // ==========================================================================
  std::cout << GridLogMessage << "\n========================================"
            << std::endl;
  std::cout << GridLogMessage << "Smearing gauge field (HISQ: fat7 + project + asqtad)"
            << std::endl;
  std::cout << GridLogMessage << "========================================"
            << std::endl;

  StaggeredImplParams stagParams;
  if (!par.boundary.empty()) {
    stagParams.boundary_phases = strToVec<Complex>(par.boundary);
  } else {
    stagParams.boundary_phases = std::vector<Complex>{
        Complex(1.), Complex(1.), Complex(1.), Complex(-1.)};
  }

  HighlyImprovedStaggeredFermionImpl<Gimpl> hisq(UGrid, stagParams);

  LatticeGaugeFieldD R(UGrid), V(UGrid), W(UGrid);
  LatticeGaugeFieldD Ufat(UGrid), Ulong(UGrid);
  hisq.rephase(R, U);
  hisq.smear(V, R);
  hisq.project(W, V);
  hisq.smear(Ufat, Ulong, W);

  // ==========================================================================
  // Write out gauge, gauge_fat, gauge_long as ILDG (cf. MIO::SaveIldg)
  // ==========================================================================
  auto saveIldg = [&](LatticeGaugeFieldD &field, const std::string &fileStem) {
    std::string fileName = fileStem + "." + std::to_string(par.trajectory);
    std::cout << GridLogMessage << "Saving ILDG configuration to file '"
              << fileName << "'" << std::endl;

    makeFileDir(fileName, field.Grid());

    std::string description =
        par.ensembleLabel.empty() ? fileName : par.ensembleLabel;

    IldgWriter writer(field.Grid()->IsBoss());
    writer.open(fileName);
    writer.writeConfiguration(field, par.trajectory, fileName, description);
    writer.close();
  };

  std::cout << GridLogMessage << "\n========================================"
            << std::endl;
  std::cout << GridLogMessage << "Writing ILDG configurations" << std::endl;
  std::cout << GridLogMessage << "========================================"
            << std::endl;

  saveIldg(U, par.gaugeStem);
  saveIldg(Ufat, par.gaugeFatStem);
  saveIldg(Ulong, par.gaugeLongStem);

  Grid_finalize();
}
