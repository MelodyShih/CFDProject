#ifndef SOLVER_H
#define SOLVER_H

#include "operators.h"

PetscErrorCode ShellPCApply(PC pc,Vec x,Vec y);
PetscErrorCode SetupHelmholtzKSP(UserContext *user);
PetscErrorCode SetupPoissionKSP(UserContext *user);


#endif
