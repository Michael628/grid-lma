#include <Grid/Grid.h>
#include <Grid/algorithms/deflation/Deflation.h>
#include <IO.h>

#ifndef DEFAULT_ASCII_PREC
#define DEFAULT_ASCII_PREC 16
#endif

NAMESPACE_BEGIN(Grid);

struct PackRecord {
  std::string operatorXml, solverXml;
};

struct VecRecord : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(VecRecord, unsigned int, index, double, eval);
  VecRecord(void) : index(0), eval(0.) {}
};

namespace EigenPackIo {
inline void readHeader(PackRecord &record, ScidacReader &binReader) {
  std::string recordXml;

  binReader.readLimeObject(recordXml, SCIDAC_FILE_XML);
  XmlReader xmlReader(recordXml, true, "eigenPackPar");
  xmlReader.push();
  xmlReader.readCurrentSubtree(record.operatorXml);
  xmlReader.nextElement();
  xmlReader.readCurrentSubtree(record.solverXml);
}

template <typename T>
void readElement(T &evec, RealD &eval, const unsigned int index,
                 ScidacReader &binReader) {
  VecRecord vecRecord;
  bool cb = false;
  GridBase *g = evec.Grid();

  binReader.readScidacFieldRecord(evec, vecRecord);

  if (vecRecord.index != index) {
    assert(0);
  }
  for (unsigned int mu = 0; mu < g->Dimensions(); ++mu) {
    cb = cb or (g->CheckerBoarded(mu) != 0);
  }
  if (cb) {
    evec.Checkerboard() = Odd;
  }
  eval = vecRecord.eval;
}

inline void readEval(RealD &eval, const unsigned int index,
                     ScidacReader &binReader) {
  VecRecord vecRecord;

  binReader.skipPastObjectRecord(std::string(GRID_FORMAT));
  binReader.readLimeObject(vecRecord, vecRecord.SerialisableClassName(),
                           std::string(SCIDAC_RECORD_XML));
  binReader.skipPastObjectRecord(std::string(SCIDAC_PRIVATE_RECORD_XML));
  binReader.skipPastBinaryRecord();

  if (vecRecord.index != index) {
    assert(0);
  }
  eval = vecRecord.eval;
}

inline void skipElements(ScidacReader &binReader, const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    binReader.skipScidacFieldRecord();
  }
}

template <typename T>
static void readPack(std::vector<T> &evec, std::vector<RealD> &eval,
                     PackRecord &record, const std::string filename,
                     const unsigned int ki, const unsigned int kf,
                     bool multiFile) {
  ScidacReader binReader;

  if (multiFile) {
    std::string fullFilename;

    for (int k = ki; k < kf; ++k) {
      fullFilename = filename + "/v" + std::to_string(k) + ".bin";
      binReader.open(fullFilename);
      readHeader(record, binReader);
      readElement(evec[k - ki], eval[k - ki], k, binReader);
      binReader.close();
    }
  } else {
    binReader.open(filename);
    readHeader(record, binReader);
    skipElements(binReader, ki);
    for (int k = ki; k < kf; ++k) {
      readElement(evec[k - ki], eval[k - ki], k, binReader);
    }
    binReader.close();
  }
}

static void readEvals(std::vector<RealD> &eval, PackRecord &record,
                      const unsigned int ki, const unsigned int kf,
                      const std::string filename, bool multiFile) {
  ScidacReader binReader;

  if (multiFile) {
    std::string fullFilename;

    for (int k = ki; k < kf; ++k) {
      fullFilename = filename + "/v" + std::to_string(k) + ".bin";
      binReader.open(fullFilename);
      readHeader(record, binReader);
      readEval(eval[k - ki], k, binReader);
      binReader.close();
    }
  } else {
    binReader.open(filename);
    readHeader(record, binReader);
    skipElements(binReader, ki);
    for (int k = ki; k < kf; ++k) {
      readEval(eval[k - ki], k, binReader);
    }
    binReader.close();
  }
}

template <typename T>
static void readPack(std::vector<T> &evec, std::vector<RealD> &eval,
                     PackRecord &record, const std::string filename,
                     const unsigned int size, bool multiFile) {
  readPack<T>(evec, eval, record, filename, 0, size, multiFile);
}

inline void writeHeader(ScidacWriter &binWriter, PackRecord &record) {
  XmlWriter xmlWriter("", "eigenPackPar");

  xmlWriter.pushXmlString(record.operatorXml);
  xmlWriter.pushXmlString(record.solverXml);
  binWriter.writeLimeObject(1, 1, xmlWriter, "parameters", SCIDAC_FILE_XML);
}

template <typename T>
void writeElement(ScidacWriter &binWriter, T &evec, RealD &eval,
                  const unsigned int index) {
  VecRecord vecRecord;

  vecRecord.eval = eval;
  vecRecord.index = index;
  binWriter.writeScidacFieldRecord(evec, vecRecord, DEFAULT_ASCII_PREC);
}

template <typename T>
static void writePack(const std::string filename, std::vector<T> &evec,
                      std::vector<RealD> &eval, PackRecord &record,
                      const unsigned int ki, const unsigned int kf,
                      bool multiFile) {
  GridBase *grid = evec[0].Grid();
  ScidacWriter binWriter(grid->IsBoss());

  if (multiFile) {
    std::string fullFilename;

    for (int k = ki; k < kf; ++k) {
      fullFilename = filename + "/v" + std::to_string(k) + ".bin";

      makeFileDir(fullFilename, grid);
      binWriter.open(fullFilename);
      writeHeader(binWriter, record);
      writeElement(binWriter, evec[k - ki], eval[k - ki], k);
      binWriter.close();
    }
  } else {
    makeFileDir(filename, grid);
    binWriter.open(filename);
    writeHeader(binWriter, record);
    for (int k = ki; k < kf; ++k) {
      writeElement(binWriter, evec[k - ki], eval[k - ki], k);
    }
    binWriter.close();
  }
}

template <typename T>
static void writePack(const std::string filename, std::vector<T> &evec,
                      std::vector<RealD> &eval, PackRecord &record,
                      const unsigned int size, bool multiFile) {
  writePack<T>(filename, evec, eval, record, 0, size, multiFile);
}
} // namespace EigenPackIo

template <typename F> class BaseEigenPack {
public:
  typedef F Field;

public:
  std::vector<RealD> eval;
  std::vector<F> evec;
  PackRecord record;

public:
  BaseEigenPack(void) = default;
  BaseEigenPack(const size_t size, GridBase *grid) { resize(size, grid); }
  virtual ~BaseEigenPack(void) = default;
  void resize(const size_t size, GridBase *grid) {
    eval.resize(size);
    evec.resize(size, grid);
  }
};

template <typename F> class EigenPack : public BaseEigenPack<F> {
public:
  typedef F Field;

public:
  EigenPack(void) = default;
  virtual ~EigenPack(void) = default;

  EigenPack(const size_t size, GridBase *grid) { init(size, grid); }

  void init(const size_t size, GridBase *grid) { this->resize(size, grid); };

  virtual void read(const std::string fileStem, const bool multiFile,
                    const int traj = -1) {
    EigenPackIo::readPack<F>(this->evec, this->eval, this->record,
                             evecFilename(fileStem, traj, multiFile),
                             this->evec.size(), multiFile);
  }

  virtual void read(const std::string fileStem, const bool multiFile,
                    const unsigned int ki, const unsigned kf,
                    const int traj = -1) {
    EigenPackIo::readPack<F>(this->evec, this->eval, this->record,
                             evecFilename(fileStem, traj, multiFile), ki, kf,
                             multiFile);
  }

  virtual void write(const std::string fileStem, const bool multiFile,
                     const int traj = -1) {
    EigenPackIo::writePack<F>(evecFilename(fileStem, traj, multiFile),
                              this->evec, this->eval, this->record,
                              this->evec.size(), multiFile);
  }

  virtual void write(const std::string fileStem, const bool multiFile,
                     const unsigned int ki, const unsigned int kf,
                     const int traj = -1) {
    EigenPackIo::writePack<F>(evecFilename(fileStem, traj, multiFile),
                              this->evec, this->eval, this->record, ki, kf,
                              multiFile);
  }

  template <typename ColourMatrixField>
  void gaugeTransform(const ColourMatrixField &g) {
    GridBase *evGrid = this->evec[0].Grid();
    ColourMatrixField gExt(evGrid);

    sliceFill(gExt, g, 0, Odd);
    for (auto &v : this->evec) {
      v = gExt * v;
    }
  }

protected:
  std::string evecFilename(const std::string stem, const int traj,
                           const bool multiFile) {
    std::string t = (traj < 0) ? "" : ("." + std::to_string(traj));

    if (multiFile) {
      return stem + t;
    } else {
      return stem + t + ".bin";
    }
  }
};

template <typename Field> class AdaptorEigenPackMILC {
public:
  AdaptorEigenPackMILC(std::vector<Field> &_evec,
                       const std::vector<RealD> &_eval, RealD _mass = 0.0)
      : evec(_evec), mass(_mass) {
    // Store shifted M eigenvalues instead of massless M^dagM eigenvalues
    eval.resize(_eval.size(), 0.0);

    Real m = 2 * _mass;

    for (int i = 0; i < eval.size(); i++) {
      eval[i] = ComplexD(m, sqrt(_eval[i]));
    }
  }

public:
  std::vector<Field> &evec;
  Vector<ComplexD> eval;
  RealD mass;
};

template <typename FImpl>
using BaseFermionEigenPack = BaseEigenPack<typename FImpl::FermionField>;

template <typename FImpl>
using FermionEigenPack = EigenPack<typename FImpl::FermionField>;

template <typename FImpl>
using MassShiftEigenPack = AdaptorEigenPackMILC<typename FImpl::FermionField>;

NAMESPACE_END(Grid);

#undef DEFAULT_ASCII_PREC
