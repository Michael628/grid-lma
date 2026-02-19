/*
 * DilutedNoise.hpp, part of Hadrons (https://github.com/aportelli/Hadrons)
 *
 * Copyright (C) 2015 - 2020
 *
 * Author: Antonin Portelli <antonin.portelli@me.com>
 * Author: Fionn O hOgain <fionn.o.hogain@ed.ac.uk>
 * Author: Fionn Ó hÓgáin <fionnoh@gmail.com>
 * Author: Peter Boyle <paboyle@ph.ed.ac.uk>
 * Author: Vera Guelpers <Vera.Guelpers@ed.ac.uk>
 * Author: Vera Guelpers <vmg1n14@soton.ac.uk>
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
#include <Grid/Grid.h>

#ifndef FMGRID_DilutedNoise_hpp_
#define FMGRID_DilutedNoise_hpp_

NAMESPACE_BEGIN(Grid);

/******************************************************************************
 *              Abstract container for spin color diagonal noise              *
 ******************************************************************************/
template <typename FImpl> class SpinColorDiagonalNoise {
public:
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

public:
  // constructor/destructor
  SpinColorDiagonalNoise(GridCartesian *g);
  SpinColorDiagonalNoise(GridCartesian *g, const int nNoise);
  virtual ~SpinColorDiagonalNoise(void) = default;
  // access
  FermionField &getFerm(const int i);
  PropagatorField &getProp(const int i);
  int size(void) const;

  void generateNoise(GridParallelRNG &rng);
  void resize(const int nNoise);
  virtual int dilutionSize(void) const = 0;

  GridCartesian *getGrid(void) const { return grid_; }
  int fermSize(void) const { return dilutionSize(); }
  std::vector<LatticeComplex> &getNoise(void) { return noise_; }
  const std::vector<LatticeComplex> &getNoise(void) const { return noise_; }
  template <typename T = FImpl> int getNsc(void) const {
    return FImpl::Dimension;
  }

private:
  template <typename T = FImpl> void setFerm(const int i);
  virtual void setProp(const int i) = 0;

protected:
  int getNd(void) const;
  FermionField &getFerm(void) { return ferm_; }
  PropagatorField &getProp(void) { return prop_; }
  void setPropagator(LatticeComplex *eta);

protected:
  int currentProp_;

private:
  FermionField ferm_;
  GridCartesian *grid_;
  std::vector<LatticeComplex> noise_;
  PropagatorField prop_;
};

template <typename FImpl>
class TimeDilutedNoise : public SpinColorDiagonalNoise<FImpl> {
public:
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

public:
  // constructor/destructor
  TimeDilutedNoise(GridCartesian *g);
  TimeDilutedNoise(GridCartesian *g, const int nNoise);
  virtual ~TimeDilutedNoise(void) = default;
  int dilutionSize(void) const;
  Lattice<iScalar<vInteger>> &getTLat() { return tLat_; }

protected:
  Lattice<iScalar<vInteger>> tLat_;

private:
  void setProp(const int i);
};

template <typename FImpl>
class FullVolumeNoise : public SpinColorDiagonalNoise<FImpl> {
public:
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

public:
  // constructor/destructor
  FullVolumeNoise(GridCartesian *g, const int nNoise);
  virtual ~FullVolumeNoise(void) = default;
  int dilutionSize(void) const;

private:
  void setProp(const int i);
};

template <typename FImpl>
class CheckerboardNoise : public SpinColorDiagonalNoise<FImpl> {
public:
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

public:
  // constructor/destructor
  CheckerboardNoise(GridCartesian *g, const int nNoise, const int nSparse);
  virtual ~CheckerboardNoise(void) = default;
  int dilutionSize(void) const;

private:
  void setProp(const int i);
  int nSparse_, nSrc_ec_;
  LatticeInteger coor_, coorTot_;
};

template <typename FImpl>
class SparseNoise : public SpinColorDiagonalNoise<FImpl> {
public:
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

public:
  // constructor/destructor
  SparseNoise(GridCartesian *g, const int nNoise, const int nSparseL,
              const int nSparseT);
  virtual ~SparseNoise(void) = default;
  int dilutionSize(void) const;

private:
  void setProp(const int i);
  int nSparseL_, nSparseT_;
  LatticeInteger coor_;
};
/******************************************************************************
 *               SpinColorDiagonalNoise template implementation *
 ******************************************************************************/
template <typename FImpl>
SpinColorDiagonalNoise<FImpl>::SpinColorDiagonalNoise(GridCartesian *g)
    : grid_(g), ferm_(g), prop_(g) {}

template <typename FImpl>
SpinColorDiagonalNoise<FImpl>::SpinColorDiagonalNoise(GridCartesian *g,
                                                      const int nNoise)
    : SpinColorDiagonalNoise(g) {
  resize(nNoise);
}

template <typename FImpl>
template <typename T>
void SpinColorDiagonalNoise<FImpl>::setFerm(const int i) {
  int nc = FImpl::Dimension;
  std::div_t divs;
  divs = std::div(i, nc);

  PropToFerm<FImpl>(ferm_, prop_, divs.rem);
}

template <typename FImpl>
typename SpinColorDiagonalNoise<FImpl>::FermionField &
SpinColorDiagonalNoise<FImpl>::getFerm(const int i) {
  auto nsc = this->getNsc();
  std::div_t divs;
  divs = std::div(i, nsc);
  setProp(divs.quot);
  setFerm(divs.rem);
  return getFerm();
}

template <typename FImpl>
void SpinColorDiagonalNoise<FImpl>::setPropagator(LatticeComplex *eta) {
  prop_ = Zero();
  for (int i = 0; i < this->getNsc(); i++) {
    pokeColour(prop_, eta[i], i, i);
  }
}

template <typename FImpl>
typename SpinColorDiagonalNoise<FImpl>::PropagatorField &
SpinColorDiagonalNoise<FImpl>::getProp(const int i) {
  setProp(i);
  return getProp();
}

template <typename FImpl> int SpinColorDiagonalNoise<FImpl>::size(void) const {
  return noise_.size() / this->getNsc();
}

template <typename FImpl> int SpinColorDiagonalNoise<FImpl>::getNd(void) const {
  return grid_->GlobalDimensions().size();
}

template <typename FImpl>
void SpinColorDiagonalNoise<FImpl>::resize(const int nNoise) {
  noise_.resize(this->getNsc() * nNoise, grid_);
}

template <typename FImpl>
void SpinColorDiagonalNoise<FImpl>::generateNoise(GridParallelRNG &rng) {
  Complex shift(1., 1.);
  LatticeComplex eta(grid_);

  eta = Zero();

  for (int n = 0; n < noise_.size(); ++n) {
    bernoulli(rng, eta);
    eta = (2. * eta - shift) * (1. / ::sqrt(2.));
    noise_[n] = eta;
  }
}

/******************************************************************************
 *                  TimeDilutedNoise template implementation *
 ******************************************************************************/
template <typename FImpl>
TimeDilutedNoise<FImpl>::TimeDilutedNoise(GridCartesian *g, int nNoise)
    : SpinColorDiagonalNoise<FImpl>(g, nNoise), tLat_(g) {}

template <typename FImpl> int TimeDilutedNoise<FImpl>::dilutionSize() const {
  auto nt = this->getGrid()->GlobalDimensions()[Tp];
  return nt * this->getNsc() * this->size();
}

template <typename FImpl> void TimeDilutedNoise<FImpl>::setProp(const int i) {
  auto noise = this->getNoise();
  auto nd = this->getNd();
  auto nt = this->getGrid()->GlobalDimensions()[Tp];
  auto nsc = this->getNsc();

  std::vector<LatticeComplex> eta(nsc, this->getGrid());
  LatticeCoordinate(tLat_, nd - 1);

  std::div_t divs = std::div(i, nt);
  int t = divs.rem;

  for (int j = 0; j < nsc; j++)
    eta[j] = where((tLat_ == t), noise[divs.quot * nsc + j],
                   0. * noise[divs.quot * nsc + j]);

  this->setPropagator(&(eta[0]));
}

/******************************************************************************
 *                   FullVolumeNoise template implementation *
 ******************************************************************************/
template <typename FImpl>
FullVolumeNoise<FImpl>::FullVolumeNoise(GridCartesian *g, int nNoise)
    : SpinColorDiagonalNoise<FImpl>(g, nNoise) {}

template <typename FImpl> int FullVolumeNoise<FImpl>::dilutionSize() const {
  return this->getNsc() * this->size();
}

template <typename FImpl> void FullVolumeNoise<FImpl>::setProp(const int i) {
  auto noise = this->getNoise();
  this->setPropagator(&noise[i * this->getNsc()]);
}

/******************************************************************************
 *                CheckerboardNoise template implementation *
 ******************************************************************************/
template <typename FImpl>
CheckerboardNoise<FImpl>::CheckerboardNoise(GridCartesian *g, int nNoise,
                                            int nSparse)
    : SpinColorDiagonalNoise<FImpl>(g, nNoise), nSparse_(nSparse), coor_(g),
      coorTot_(g) {
  if (nNoise % nSparse_ == 0) {
    nSrc_ec_ = nNoise / nSparse_;
  } else {
    nSrc_ec_ = (nNoise - nNoise % nSparse_) / nSparse_;
  }
}

template <typename FImpl> int CheckerboardNoise<FImpl>::dilutionSize() const {
  return this->getNsc() * this->size();
}

template <typename FImpl> void CheckerboardNoise<FImpl>::setProp(const int i) {
  auto nd = this->getNd();
  auto noise = this->getNoise();
  auto nsc = this->getNsc();
  unsigned int j;

  std::vector<LatticeComplex> eta(nsc, this->getGrid());

  j = i / nSrc_ec_;

  coorTot_ = 0.;
  for (int d = 0; d < nd; ++d) {
    LatticeCoordinate(coor_, d);
    coorTot_ = coorTot_ + coor_;
  }
  coor_ = j;
  coorTot_ = coorTot_ + coor_;

  for (int k = 0; k < nsc; k++) {
    eta[k] = where(mod(coorTot_, nSparse_), 0. * noise[i * nsc + k],
                   noise[i * nsc + k]);
    eta[k] *= sqrt(1. / nSrc_ec_);
  }
  this->setPropagator(&(eta[0]));
}

/******************************************************************************
 *                SparseNoise template implementation                   *
 ******************************************************************************/
template <typename FImpl>
SparseNoise<FImpl>::SparseNoise(GridCartesian *g, int nNoise, int nSparseL,
                                int nSparseT)
    : SpinColorDiagonalNoise<FImpl>(g, nNoise), nSparseL_(nSparseL),
      nSparseT_(nSparseT), coor_(g) {}

template <typename FImpl> int SparseNoise<FImpl>::dilutionSize() const {
  auto nd = this->getNd();
  return this->getNsc() * this->size() * pow(nSparseL_, nd - 1) * nSparseT_;
}

template <typename FImpl> void SparseNoise<FImpl>::setProp(const int i) {
  auto nd = this->getNd();
  auto noise = this->getNoise();
  auto nsc = this->getNsc();

  std::vector<LatticeComplex> eta(nsc, this->getGrid());

  std::div_t divs = std::div(i, pow(nSparseL_, nd - 1) * nSparseT_);
  std::div_t subdivs;

  for (int j = 0; j < nsc; j++) {
    eta[j] = noise[divs.quot * nsc + j];
  }

  unsigned int sparseIndex = divs.rem;
  for (int d = 0; d < nd; ++d) {
    LatticeCoordinate(coor_, d);
    if (d < nd - 1)
      subdivs = std::div(sparseIndex, nSparseL_);
    else
      subdivs = std::div(sparseIndex, nSparseT_);

    for (int j = 0; j < nsc; j++) {
      eta[j] = where(coor_ == ((uint32_t)subdivs.rem), eta[j], 0. * eta[j]);
    }
    sparseIndex = subdivs.quot;
  }
  this->setPropagator(&(eta[0]));
}

NAMESPACE_END(Grid);

#endif // FMGRID_DilutedNoise_hpp_
