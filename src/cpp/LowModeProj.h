/*
 * LowModeProj.hpp, part of Hadrons (https://github.com/aportelli/Hadrons)
 *
 * Copyright (C) 2015 - 2020
 *
 * Author: Antonin Portelli <antonin.portelli@me.com>
 * Author: Fionn O hOgain <fionn.o.hogain@ed.ac.uk>
 * Author: Fionn Ó hÓgáin <fionnoh@gmail.com>
 * Author: fionnoh <fionnoh@gmail.com>
 * Author: Michael Lynch <michaellynch628@gmail.com>
 *
 * Hadrons is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Hadrons is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Hadrons.  If not, see <http://www.gnu.org/licenses/>.
 *
 * See the full license in the file "LICENSE" in the top level distribution
 * directory.
 */

/*  END LEGAL */

#ifndef FMGRID_LowModeProj_hpp_
#define FMGRID_LowModeProj_hpp_

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

class LowModeProjPar : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(LowModeProjPar, bool, projector, unsigned int,
                                  eigStart, int, nEigs);
};

template <typename FImpl, typename Pack> class LowModeProj {
public:
  typedef FermionOperator<FImpl> FMat;
  typedef typename FImpl::FermionField FermionField;

  // constructor
  LowModeProj(FMat &action, Pack &epack, GridBase *grid)
      : _epack(epack), _action(action), _rbFerm(grid), _rbFermNeg(grid),
        _MrbFermNeg(grid), _rbTemp(grid), _rbTempNeg(grid) {};
  // destructor
  ~LowModeProj() = default;

  void solve(const LowModeProjPar &par, FermionField &sol,
             const FermionField &source, bool subGuess = false);

private:
  const FMat &_action;
  const Pack &_epack;
  FermionField _rbFerm;
  FermionField _rbFermNeg;
  FermionField _MrbFermNeg;
  FermionField _rbTemp;
  FermionField _rbTempNeg;
};

template <typename FImpl, typename Pack>
void LowModeProj<FImpl, Pack>::solve(const LowModeProjPar &par,
                                     FermionField &sol,
                                     const FermionField &source,
                                     bool subGuess) {
  bool projector = par.projector;

  _rbFerm = Zero();
  _rbFermNeg = Zero();
  _MrbFermNeg = Zero();

  auto eigStart = par.eigStart;
  auto nEigs = par.nEigs;

  if (nEigs < 1) {
    nEigs = _epack.evec.size();
  }

  if (eigStart > nEigs || eigStart > _epack.evec.size() ||
      nEigs - eigStart > _epack.evec.size() - eigStart) {
    std::cout << GridLogError
              << "Requested eigs (parameters eigStart and nEigs) out of bounds."
              << std::endl;
    assert(0);
  }

  int cb = _epack.evec[0].Checkerboard();
  int cbNeg = (cb == Even) ? Odd : Even;

  RealD norm = 1. / ::sqrt(norm2(_epack.evec[0]));

  _rbTemp = Zero();
  _rbTemp.Checkerboard() = cb;
  _rbTempNeg = Zero();
  _rbTempNeg.Checkerboard() = cb;

  _rbFerm.Checkerboard() = cb;
  _rbFermNeg.Checkerboard() = cbNeg;
  _MrbFermNeg.Checkerboard() = cb;
  {
    pickCheckerboard(cb, _rbFerm, source);
    pickCheckerboard(cbNeg, _rbFermNeg, source);
  }
  _action.MeooeDag(_rbFermNeg, _MrbFermNeg);

  for (int k = (eigStart + nEigs - 1); k >= int(eigStart); k--) {
    const FermionField &e = _epack.evec[k];

    const RealD mass = _epack.eval[k].real();
    const RealD lam_D = _epack.eval[k].imag();
    const RealD invlam_D = 1. / lam_D;
    const RealD invmag = 1. / (pow(mass, 2) + pow(lam_D, 2));

    if (!par.projector) {
      const ComplexD ip = TensorRemove(innerProduct(e, _rbFerm)) * invmag;
      const ComplexD ipNeg =
          TensorRemove(innerProduct(e, _MrbFermNeg)) * invmag;
      axpy(_rbTemp, mass * ip + ipNeg, e, _rbTemp);
      axpy(_rbTempNeg, mass * ipNeg * invlam_D * invlam_D - ip, e, _rbTempNeg);
    } else {
      const ComplexD ip = TensorRemove(innerProduct(e, _rbFerm));
      const ComplexD ipNeg = TensorRemove(innerProduct(e, _MrbFermNeg));
      axpy(_rbTemp, ip, e, _rbTemp);
      axpy(_rbTempNeg, ipNeg * invlam_D * invlam_D, e, _rbTempNeg);
    }
  }

  _action.Meooe(_rbTempNeg, _rbFermNeg);
  {
    setCheckerboard(sol, _rbTemp);
    setCheckerboard(sol, _rbFermNeg);
  }
  sol *= norm;
  if (subGuess) {
    if (par.projector) {
      sol = source - sol;
    } else {
      std::cout << GridLogError
                << "Subtracted solver only supported for projector=true"
                << std::endl;
      assert(0);
    }
  }
}

NAMESPACE_END(Grid);

#endif
