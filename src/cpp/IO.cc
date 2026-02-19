#include <IO.h>

NAMESPACE_BEGIN(Grid);

#define MAX_PATH_LENGTH 512u

int mkdir(const std::string dirName) {
  if (!dirName.empty() and access(dirName.c_str(), R_OK | W_OK | X_OK)) {
    mode_t mode755;
    char tmp[MAX_PATH_LENGTH];
    char *p = NULL;
    size_t len;

    mode755 = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;

    snprintf(tmp, sizeof(tmp), "%s", dirName.c_str());
    len = strlen(tmp);
    if (tmp[len - 1] == '/') {
      tmp[len - 1] = 0;
    }
    for (p = tmp + 1; *p; p++) {
      if (*p == '/') {
        *p = 0;
        ::mkdir(tmp, mode755);
        *p = '/';
      }
    }

    return ::mkdir(tmp, mode755);
  } else {
    return 0;
  }
}

std::string dirname(const std::string &s) {
  constexpr char sep = '/';
  size_t i = s.rfind(sep, s.length());

  if (i != std::string::npos) {
    return s.substr(0, i);
  } else {
    return "";
  }
}

void makeFileDir(const std::string filename, GridBase *g) {
  bool doIt = true;

  if (g) {
    doIt = g->IsBoss();
  }
  if (doIt) {
    std::string dir = dirname(filename);
    int status = mkdir(dir);

    if (status) {
      assert(0);
    }
  }
}
std::string resultFilename(const std::string stem, const GlobalPar &inputParams,
                           const std::string ext, bool includeSeries) {
  return stem + (includeSeries ? ("_" + inputParams.series) : "") + "." +
         std::to_string(inputParams.trajectory) + "." + ext;
}

std::string getSeed(GlobalPar &inputParams, std::string seedSuffix) {
  return inputParams.runSeed + (seedSuffix.empty() ? "" : "-" + seedSuffix) +
         "-" + std::to_string(inputParams.trajectory);
}

#undef MAX_PATH_LENGTH

NAMESPACE_END(Grid);
