#pragma once

#include <SpatialTrace.h>
#include <StagGamma.h>
#include <a2a/A2ATask.h>
#include <a2a/A2AView.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// A2ATaskFused: GEMM-based meson field contraction for staggered fermions.
//
// Fuses inner product + phase multiplication + spatial sum into a single
// batched GEMM by interleaving colour components into the K dimension:
//
//   result[γ,i,j,t] = Σ_{xyz,c} phase[γ,xyz] · conj(left_i^c[xyz])
//                                               · right_j^c[xyz]
//
// BLAS_L: M = Nphase × block × Nsimd,  K = rNxyz × Nc
// BLAS_R: N = block × Nsimd,            K = rNxyz × Nc
// GEMM:   C = L × R^T
///////////////////////////////////////////////////////////////////////////////

template <typename FImpl> class A2ATaskFused : public A2ATaskBase<FImpl> {
public:
  typedef typename FImpl::SiteSpinor vobj;
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename ComplexField::vector_object cobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;
  typedef iSinglet<scalar_type> Scalar_s;
  typedef iSinglet<vector_type> Scalar_v;
  typedef decltype(coalescedRead(Scalar_v())) calcScalar;
  typedef decltype(coalescedRead(vobj())) calcSpinor;
  typedef typename A2ATaskBase<FImpl>::ContractType ContractType;
  typedef LatticeView<cobj> ComplexView;
  typedef LatticeView<vobj> FermView;

  static constexpr int Nc = 3;

protected:
  std::vector<StagGamma::SpinTastePair> _gammas;
  std::vector<ComplexField> _phase;
  std::shared_ptr<A2AFieldView<cobj>> _phase_view;

  typedef iSinglet<Complex> ResultObj;
  SpatialTrace<ComplexField, ComplexField, ResultObj> _ST;
  bool _st_allocated{false};
  bool _left_filled{false};
  bool _right_filled{false};

public:
  // Primary constructor: computes phase fields from gammas
  A2ATaskFused(GridBase *grid, int orthogDir,
               const std::vector<StagGamma::SpinTastePair> &gammas,
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {

    int nGamma = _gammas.size();
    _phase.resize(nGamma, this->_full_grid);

    ComplexField temp(this->_full_grid);
    temp = 1.0;
    StagGamma spinTaste;

    _phase_view = std::make_shared<A2AFieldView<cobj>>();
    _phase_view->reserve(nGamma);

    for (int mu = 0; mu < nGamma; mu++) {
      spinTaste.setSpinTaste(_gammas[mu]);
      StagGamma::print(spinTaste);
      spinTaste.applyPhase(_phase[mu], temp);
    }
    _phase_view->openViews(_phase.data(), nGamma);
  }

  // Sibling constructor: shares phase views for cb pairing
  A2ATaskFused(GridBase *grid, int orthogDir, A2ATaskFused<FImpl> &other,
               const std::vector<StagGamma::SpinTastePair> &gammas = {},
               int cb = Even)
      : A2ATaskBase<FImpl>(grid, orthogDir, cb), _gammas(gammas) {
    _phase_view = other.getPhaseView();
  }

  virtual ~A2ATaskFused() {
    if (_phase_view)
      _phase_view->closeViews();
  }

  std::shared_ptr<A2AFieldView<cobj>> getPhaseView() { return _phase_view; }

  virtual int getNgamma() { return _phase_view->size(); }

  virtual double getFlops() {
    // GEMM dominates: 8 real flops per complex multiply-add
    // M = Nphase * block, N = block, K = rNxyz * Nc, batches = rNt
    return 8.0 * getNgamma();
  }

  virtual void setLeft(const FermionField *left, int size) {
    A2ATaskBase<FImpl>::setLeft(left, size);
    _left_filled = false;
  }

  virtual void setLeft(A2ATaskBase<FImpl> &other) {
    A2ATaskBase<FImpl>::setLeft(other);
    _left_filled = false;
  }

  virtual void setRight(const FermionField *right, int size) {
    A2ATaskBase<FImpl>::setRight(right, size);
    _right_filled = false;
  }

  virtual void setRight(A2ATaskBase<FImpl> &other) {
    A2ATaskBase<FImpl>::setRight(other);
    _right_filled = false;
  }

  // Stubs — not used since execute() is overridden
  virtual void vectorSumHalf(cobj *, int, int) { assert(0); }
  virtual void vectorSumFull(cobj *, int, int) { assert(0); }
  virtual void vectorSumMixed(cobj *, int, int) { assert(0); }

  virtual void execute(scalar_type *result_p) override {

    assert((this->_contract_type == ContractType::Full ||
            this->_contract_type == ContractType::BothHalf) &&
           "A2ATaskFused only supports Full and BothHalf contract types");

    bool bothHalf = (this->_contract_type == ContractType::BothHalf);
    bool cbEven = this->_cb_left == Even;
    bool oddShifts = this->_odd_shifts;

    int sizeL = this->_left_view->size();
    int sizeR = this->_right_view->size();
    int Nphase = _phase_view->size();
    int nsimd = this->_grid->Nsimd();

    int orthogDir = this->_orthog_dir;
    int Nt = this->_grid->GlobalDimensions()[orthogDir];
    int nd = this->_grid->_ndimension;
    auto rNt = this->_grid->_rdimensions[nd - 1];
    auto rNxyz = this->_grid->oSites() / rNt;
    int osites = this->_grid->oSites();

    int64_t Nleft = Nphase * sizeL;
    int64_t Nright = sizeR;
    int64_t K = rNxyz * Nc;

    // Lazy allocate BLAS buffers
    if (!_st_allocated) {
      _ST.Allocate(Nleft, Nright, this->_grid, Nc);
      _st_allocated = true;
    }

    auto *blas_L = _ST.getBlasLeftPointer();
    auto *blas_R = _ST.getBlasRightPointer();

    auto phase_v = _phase_view->getView();
    FermView *viewL_p = this->_left_view->getView();
    FermView *viewR_p = this->_right_view->getView();

    // Fill BLAS_L: phase[mmu] * conj(left_i^c), colour-interleaved into K
    if (!_left_filled) {
      GRID_TRACE("FillBlasL");

      for (int mmu = 0; mmu < Nphase; mmu++) {
        accelerator_for2d(os, osites, ii, sizeL, nsimd, {
          calcSpinor left = coalescedRead(viewL_p[ii][os]);
          calcScalar phase = coalescedRead(phase_v[mmu][os]);
          int64_t m_vec = mmu * sizeL + ii;

          for (int c = 0; c < Nc; c++) {
            calcScalar val;
            val()()() = phase()()() * conjugate(left()()(c));

            int64_t idx = m_vec + Nleft * (os * Nc + c);
            coalescedWrite(((cobj *)blas_L)[idx], val);
          }
        });
      }
      _left_filled = true;
    }

    // Fill BLAS_R: right_j^c, colour-interleaved into K
    if (!_right_filled) {
      GRID_TRACE("FillBlasR");

      accelerator_for2d(os, osites, jj, sizeR, nsimd, {
        calcSpinor right = coalescedRead(viewR_p[jj][os]);

        for (int c = 0; c < Nc; c++) {
          calcScalar val;
          val()()() = right()()(c);

          int64_t idx = jj + Nright * (os * Nc + c);
          coalescedWrite(((cobj *)blas_R)[idx], val);
        }
      });
      _right_filled = true;
    }

    // Batched GEMM: one per time slice
    {
      GRID_TRACE("GEMM");

      deviceVector<scalar_type *> Ld(rNt), Rd(rNt), Td(rNt);
      scalar_type *Lh = &_ST.BLAS_L[0];
      scalar_type *Rh = &_ST.BLAS_R[0];
      scalar_type *Th = &_ST.BLAS_T[0];

      int64_t leftStride = rNxyz * Nleft * _ST.leftWordsBatch;
      int64_t rightStride = rNxyz * Nright * _ST.rightWordsBatch;
      int64_t resultStride = _ST.nresults * _ST.resultWordsBatch;

      for (int t = 0; t < rNt; t++) {
        acceleratorPut(Ld[t], Lh + t * leftStride);
        acceleratorPut(Rd[t], Rh + t * rightStride);
        acceleratorPut(Td[t], Th + t * resultStride);
      }

      GridBLAS BLAS;
      BLAS.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_T, Nleft * _ST.leftWordsBatch,
                       Nright * _ST.rightWordsBatch, rNxyz, scalar_type(1.0),
                       Ld, Rd, scalar_type(0.0), Td);
      BLAS.synchronise();
    }

    // Export: unfold SIMD lanes, map to result_p layout
    // ExportTrace gives host data: trace[lt * nresults + i * Nright + j]
    // where i = mmu * sizeL + ii, j = jj
    // result_p layout: [gamma][t][l_index][r_index] on device
    {
      GRID_TRACE("ExtractResults");

      std::vector<ResultObj> trace;
      _ST.ExportTrace(trace);

      int nresults = Nleft * Nright;
      int lt = this->_grid->LocalDimensions()[nd - 1];
      int pc = this->_grid->_processor_coor[orthogDir];
      int localOrthogDimSize = this->_grid->_ldimensions[orthogDir];

      // Output dimensions: for BothHalf, indices are doubled
      int outSizeL = bothHalf ? 2 * sizeL : sizeL;
      int outSizeR = bothHalf ? 2 * sizeR : sizeR;
      int totalSize = Nphase * Nt * outSizeL * outSizeR;
      std::vector<scalar_type> host_result(totalSize, scalar_type(0.0));

      for (int t = 0; t < lt; t++) {
        int gt = t + pc * localOrthogDimSize;
        for (int mmu = 0; mmu < Nphase; mmu++) {
          int gammaOffset = mmu * outSizeR * outSizeL * Nt;
          for (int ii = 0; ii < sizeL; ii++) {
            for (int jj = 0; jj < sizeR; jj++) {
              int trace_idx = t * nresults + (mmu * sizeL + ii) * Nright + jj;
              scalar_type val = TensorRemove(trace[trace_idx]());

              if (!bothHalf) {
                int result_idx = gammaOffset + (jj + sizeR * (ii + sizeL * gt));
                host_result[result_idx] = val;
              } else {
                // Expand checkerboard result into (e+o)/(e-o) components
                // Layout mirrors simdSumHalf: outSizeR = 2*sizeR
                int ij_dx =
                    gammaOffset + 2 * (jj + sizeR * 2 * (ii + sizeL * gt));

                // (ee) quadrant
                host_result[ij_dx] += val;

                // (eo), (oe), (oo) quadrants with sign from cb and shifts
                if (cbEven && oddShifts) {
                  host_result[ij_dx + 1] -= val;
                  host_result[ij_dx + outSizeR + 1] -= val;
                  host_result[ij_dx + outSizeR] += val;
                } else if (cbEven) {
                  host_result[ij_dx + 1] += val;
                  host_result[ij_dx + outSizeR + 1] += val;
                  host_result[ij_dx + outSizeR] += val;
                } else if (oddShifts) {
                  host_result[ij_dx + 1] += val;
                  host_result[ij_dx + outSizeR + 1] -= val;
                  host_result[ij_dx + outSizeR] -= val;
                } else {
                  host_result[ij_dx + 1] -= val;
                  host_result[ij_dx + outSizeR + 1] += val;
                  host_result[ij_dx + outSizeR] -= val;
                }
              }
            }
          }
        }
      }

      // Accumulate onto device (result_p was zeroed by the worker)
      // Copy host to a temp device buffer, then add
      scalar_type *tmp_device = (scalar_type *)acceleratorAllocDevice(
          totalSize * sizeof(scalar_type));
      acceleratorCopyToDevice(host_result.data(), tmp_device,
                              totalSize * sizeof(scalar_type));

      // Accumulate: result_p[i] += tmp_device[i]
      accelerator_for(i, totalSize, 1,
                      { result_p[i] = result_p[i] + tmp_device[i]; });

      acceleratorFreeDevice(tmp_device);
    }
  }
};

NAMESPACE_END(Grid);
