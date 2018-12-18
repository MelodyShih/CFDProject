#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "utils.h"

PetscErrorCode WriteSolution(UserContext *user, Vec b)
{
  PetscViewer viewer;
  Vec u,v,p;

  VecGetSubVector(b, user->isg[0], &u);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,"u.m",&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
  VecView(u,viewer);

  VecGetSubVector(b, user->isg[1], &v);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,"v.m",&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
  VecView(v,viewer);

  VecGetSubVector(b, user->isg[2], &p);
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,"p.m",&viewer);
  PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
  VecView(p,viewer);
  PetscViewerDestroy(&viewer);

  return(0);
}
