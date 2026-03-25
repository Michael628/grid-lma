#ifndef FMGRID_A2AMESONFIELD_H
#define FMGRID_A2AMESONFIELD_H

#include <Eigenpack.h>
#include <Grid/Grid.h>
#include <IO.h>
#include <MesonFieldKernel.h>
#include <StagGamma.h>

NAMESPACE_BEGIN(Grid);

template <typename FImpl, typename FermionOpD>
void computeA2AMesonFields(
    const GlobalPar &inputParams, GridCartesian *UGrid,
    GridRedBlackCartesian *UrbGrid, LatticeGaugeFieldD &U,
    LatticeGaugeFieldD &U_fat, LatticeGaugeFieldD &U_long,
    typename FermionOpD::ImplParams &implParams,
    std::shared_ptr<EigenPack<typename FImpl::FermionField>> epack, int traj) {

  using FermionFieldD = typename FImpl::FermionField;
  using Computation = A2AMatrixBlockComputation<ComplexD, FermionFieldD,
                                                MesonFieldMetadata,
                                                HADRONS_A2AM_IO_TYPE>;
  using Kernel = MesonFieldKernel<Complex, FImpl>;

  bool hasEigs = (epack != nullptr);
  int Nt = UGrid->GlobalDimensions()[Tp];

  for (size_t a2aIdx = 0; a2aIdx < inputParams.a2a.size(); ++a2aIdx) {
    auto &a2aPar = inputParams.a2a[a2aIdx];

    if (!hasEigs)
      break;

    // Create action for this a2a entry
    auto stagMatMassive = std::make_shared<FermionOpD>(
        *UGrid, *UrbGrid, 2. * a2aPar.action.mass, 2. * a2aPar.action.c1,
        2. * a2aPar.action.c2, a2aPar.action.tad, implParams);
    stagMatMassive->ImportGaugeSimple(U_long, U_fat);

    RealD a2aMass = 2.0 * a2aPar.action.mass;
    int nBlock = a2aPar.block;

    std::cout << GridLogMessage
              << "\nSetting up all-to-all meson field construction (" << a2aIdx
              << ")" << std::endl;
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

    std::shared_ptr<std::vector<LatticeComplexD>> ph =
        std::make_shared<std::vector<LatticeComplexD>>(0, UGrid);

    std::shared_ptr<Computation> computationLocal =
        std::make_shared<Computation>(UGrid, Tdir, mom.size(),
                                      gammaLocal.size(), nBlock);

    std::shared_ptr<Computation> computationComms =
        std::make_shared<Computation>(UGrid, Tdir, mom.size(),
                                      gammaComms.size(), nBlock);

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
              << "All-to-all meson field construction complete (" << a2aIdx
              << ")" << std::endl;
  }
}

NAMESPACE_END(Grid);

#endif
