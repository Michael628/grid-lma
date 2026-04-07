/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: SpatialTrace.h

    Copyright (C) 2025

Author: Peter Boyle <pboyle@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

NAMESPACE_BEGIN(Grid);
/*
   SpatialTrace

   Import left fields  -> nt x nxyz x (nleft x leftWords)
   Import right fields -> nt x nxyz x (nright x rightWords)

   Compute spatial trace via batched GEMM, one batch per time slice:
   For each time t: Trace[t] = sum_xyz Left[t,xyz]^dag * Right[t,xyz]
*/
template <class LeftField, class RightField, typename result_object>
class SpatialTrace {
public:
  typedef typename RightField::vector_type vector;
  typedef typename RightField::scalar_type scalar;
  typedef typename RightField::scalar_object right_scalar_object;
  typedef typename LeftField::scalar_object left_scalar_object;

  GridBase *grid;
  uint64_t rNxyz;
  uint64_t rNt;
  uint64_t nright;
  uint64_t nleft;
  uint64_t nresults;
  uint64_t rightWordsBatch;
  uint64_t leftWordsBatch;
  uint64_t resultWordsBatch;
  uint64_t resultWords;
  int64_t nsimd;
  int K_factor;

  deviceVector<scalar> BLAS_L;
  deviceVector<scalar> BLAS_R;
  deviceVector<scalar> BLAS_T;

  SpatialTrace() : K_factor(1) {};
  ~SpatialTrace() { Deallocate(); };

  void Deallocate(void) {
    grid = nullptr;
    rNxyz = 0;
    rNt = 0;
    nright = 0;
    nleft = 0;
    rightWordsBatch = 0;
    leftWordsBatch = 0;
    resultWordsBatch = 0;
    resultWords = 0;
    nresults = 0;
    K_factor = 1;
    BLAS_L.resize(0);
    BLAS_R.resize(0);
    BLAS_T.resize(0);
  }

  // K_factor multiplies the spatial K dimension (e.g. Nc for
  // colour-interleaved)
  void Allocate(int _nleft, int _nright, GridBase *_grid, int _K_factor = 1) {
    grid = _grid;
    nsimd = vector::Nsimd();
    K_factor = _K_factor;
    Coordinate ldims = grid->LocalDimensions();

    rNt = grid->_rdimensions[grid->Nd() - 1];
    rNxyz = (grid->oSites() / rNt) * K_factor;
    nleft = _nleft;
    nright = _nright;
    // Could probably do a different sizeof() call to include nsimd
    leftWordsBatch = nsimd * sizeof(left_scalar_object) / sizeof(scalar);
    rightWordsBatch = nsimd * sizeof(right_scalar_object) / sizeof(scalar);
    resultWordsBatch = nsimd * nsimd * sizeof(result_object) / sizeof(scalar);
    resultWords = sizeof(result_object) / sizeof(scalar);
    nresults =
        nleft * leftWordsBatch * nright * rightWordsBatch / resultWordsBatch;

    GRID_ASSERT(nleft * leftWordsBatch * nright * rightWordsBatch ==
                nresults * resultWordsBatch);
    // Layout: BLAS_L[rNt][rNxyz][nleft * leftWordsBatch]
    BLAS_L.resize(rNt * rNxyz * nleft * leftWordsBatch);
    // Layout: BLAS_R[rNt][rNxyz][nright * rightWordsBatch]
    BLAS_R.resize(rNt * rNxyz * nright * rightWordsBatch);
    // Layout: BLAS_T[rNt][resultWordsBatch]
    BLAS_T.resize(rNt * nresults * resultWordsBatch);
  }

  // Accessor functions cast BLAS heap objects into vectorized lists
  vector *getBlasRightPointer() {
    assert(BLAS_R.size() > 0);
    return (vector *)&BLAS_R[0];
  }

  vector *getBlasLeftPointer() {
    assert(BLAS_L.size() > 0);
    return (vector *)&BLAS_L[0];
  }

  /**** Grid mem layout already coalesced in time-major order ***/
  // Integer *getOmapPointer() {
  //   if (OMAP.size() != 0)
  //     return (Integer *)&OMAP[0];
  //
  //   int nd = grid->_ndimension;
  //
  //   Coordinate rdimensions = grid->_rdimensions;
  //
  //   Integer *map_p;
  //   OMAP.resize(rsites);
  //   map_p = &OMAP[0];
  //
  //   int64_t nxyz = rNxyz; // for capture
  //
  //   accelerator_for(os, osites, 1, {
  //     Coordinate ocoor(nd);
  //
  //     Lexicographic::CoorFromIndex(ocoor, os, rdimensions);
  //
  //     uint64_t r_t = ocoor[nd - 1];
  //
  //     Coordinate xyz_coor = ocoor;
  //     xyz_coor[nd - 1] = 0;
  //     Lexicographic::IndexFromCoor(xyz_coor, r_xyz, rdimensions);
  //
  //     uint64_t idx = r_t * nxyz + r_xyz;
  //     map_p[idx] = os;
  //   });
  //
  //   return (Integer *)&OMAP[0];
  // }

  void ExportTrace(std::vector<result_object> &trace) {
    int nd = grid->Nd();
    int nt = grid->LocalDimensions()[nd - 1];
    // Output `trace` result will have simd lanes extracted
    trace.resize(nt * nresults);

    for (int t = 0; t < trace.size(); t++) {
      trace[t] = Zero();
    }
    // Intermediate storage for coalesced BLAS output
    std::vector<scalar> temp_result(nresults * resultWordsBatch);

    Coordinate icoor(nd);
    Coordinate simd = grid->_simd_layout;
    // Now transpose each result in-place
    int M = nleft * leftWordsBatch / nsimd;
    int N = nright * rightWordsBatch / nsimd;

    for (int t = 0; t < rNt; t++) {
      acceleratorCopyFromDevice(&BLAS_T[t * nresults * resultWordsBatch],
                                &temp_result[0],
                                nresults * resultWordsBatch * sizeof(scalar));

      scalar *result = (scalar *)&trace[t * nresults];
      scalar *data = &temp_result[0];

      for (int lane = 0; lane < nsimd; lane++) {
        Lexicographic::CoorFromIndex(icoor, lane, simd);
        int simd_t = icoor[nd - 1];

        // Transpose M×N matrix in-place
        // For small matrices, use a temporary
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            // Column-major input at j*M + i becomes row-major at i*N + j
            result[simd_t * rNt * M * N + i * N + j] +=
                data[j * nsimd * M * nsimd + lane * M * nsimd + i * nsimd +
                     lane];
          }
        }
      }
    }
  }

  void Trace(std::vector<result_object> &trace_gdata,
             GridBLASOperation_t opA = GridBLAS_OP_N,
             GridBLASOperation_t opB = GridBLAS_OP_T) {
    double t_import = 0;
    double t_export = 0;
    double t_gemm = 0;
    double t_allreduce = 0;
    std::vector<result_object> trace_planes;

    // Setup batched GEMM pointers - one batch per time slice
    deviceVector<scalar *> Ld(rNt);
    deviceVector<scalar *> Rd(rNt);
    deviceVector<scalar *> Td(rNt);

    scalar *Lh = &BLAS_L[0];
    scalar *Rh = &BLAS_R[0];
    scalar *Th = &BLAS_T[0];

    {
      GRID_TRACE("Import");
      t_import -= usecond();

      // Each batch points to a different time slice
      for (int t = 0; t < rNt; t++) {
        acceleratorPut(Ld[t], Lh + t * rNxyz * nleft * leftWordsBatch);
        acceleratorPut(Rd[t], Rh + t * rNxyz * nright * rightWordsBatch);
        acceleratorPut(Td[t], Th + t * nresults * resultWordsBatch);
      }
      t_import += usecond();
    }

    GridBLAS BLAS;

    /////////////////////////////////////////
    // For each time t: T[t] = L[t] * R[t]
    // Sum over spatial xyz dimension
    /////////////////////////////////////////
    {
      GRID_TRACE("GEMM");
      t_gemm -= usecond();
      BLAS.gemmBatched(opA, opB,
                       nleft * leftWordsBatch,   // M (rows of L)
                       nright * rightWordsBatch, // N (cols of R)
                       rNxyz,                    // K (sum over spatial)
                       scalar(1.0), Ld, Rd,
                       scalar(0.0), // don't accumulate result
                       Td);
      BLAS.synchronise();
      t_gemm += usecond();
    }

    // Unfolds time from simd lanes and accumulates
    {
      GRID_TRACE("ExportTrace");
      t_export -= usecond();
      ExportTrace(trace_planes);
      t_export += usecond();
    }
    /////////////////////////////////
    // Reduce across MPI ranks
    /////////////////////////////////
    int nd = grid->Nd();
    int gt = grid->GlobalDimensions()[nd - 1];
    int lt = grid->LocalDimensions()[nd - 1];
    trace_gdata.resize(gt * nresults);

    // Initialize with zeros
    for (int t = 0; t < trace_gdata.size(); t++) {
      trace_gdata[t] = Zero();
    }

    // Fill in local time slices
    for (int t = 0; t < lt; t++) {
      int st = grid->LocalStarts()[nd - 1];
      for (int r = 0; r < nresults; r++) {
        trace_gdata[(t + st) * nresults + r] = trace_planes[t * nresults + r];
      }
    }

    {
      GRID_TRACE("GlobalSum");
      t_allreduce -= usecond();
      grid->GlobalSumVector((scalar *)&trace_gdata[0],
                            gt * nresults * resultWords);
      t_allreduce += usecond();
    }

    std::cout << GridLogPerformance << " SpatialTrace t_import  " << t_import
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_export  " << t_export
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_gemm    " << t_gemm
              << "us" << std::endl;
    std::cout << GridLogPerformance << " SpatialTrace t_reduce  " << t_allreduce
              << "us" << std::endl;
  }
};

NAMESPACE_END(Grid);
