#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
// #include <petscpcmg.h>

#include "utils.h"

PetscErrorCode ShellPCApply(PC pc,Vec x,Vec y)
{
  UserContext *user;
  PCShellGetContext(pc,(void**)&user);
  Vec fu, fv, fp;
  Vec ustar, vstar, pstar;
  Vec dudx, dudy;
  Vec phi,temp;
  MatNullSpace   nullspace;


  VecGetSubVector(x, user->isg[0], &fu);
  VecGetSubVector(x, user->isg[1], &fv);
  VecGetSubVector(x, user->isg[2], &fp);

  VecGetSubVector(y, user->isg[0], &ustar);
  VecGetSubVector(y, user->isg[1], &vstar);
  VecGetSubVector(y, user->isg[2], &pstar);

  // VecView(fp, (PetscViewer) PETSC_VIEWER_DEFAULT);
  // VecView(x, (PetscViewer) PETSC_VIEWER_DEFAULT);

  int iter;

  KSPSolve(user->ksp_helmholtz, fu, ustar);
  KSPGetIterationNumber(user->ksp_helmholtz,&iter);
  user->hxvcycles += iter;
  KSPSolve(user->ksp_helmholtz, fv, vstar);
  KSPGetIterationNumber(user->ksp_helmholtz,&iter);
  user->hyvcycles += iter;

  VecDuplicate(ustar,&dudx);
  VecDuplicate(vstar,&dudy);

  MatMult(user->subA[6],ustar,dudx);
  MatMult(user->subA[7],vstar,dudy);
  // VecScale(dudx,1.0/(user->dt*user->dt));
  // VecScale(dudy,1.0/(user->dt*user->dt));

  VecAXPY(dudy,1.0,dudx);
  VecAXPY(dudy,-1.0,fp);

  MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);
  MatNullSpaceRemove(nullspace,dudy);
  MatNullSpaceDestroy(&nullspace);

  VecDuplicate(ustar,&phi);
  VecDuplicate(ustar,&temp);
  KSPSolve(user->ksp_poission, dudy, phi);
  KSPGetIterationNumber(user->ksp_poission,&iter);
  user->pvcycles += iter;

  MatMult(user->subA[2], phi, temp);
  VecAXPY(ustar,-1.0,temp);

  MatMult(user->subA[5], phi, temp);
  VecAXPY(vstar,-1.0,temp);

  MatMult(user->subA[0], phi, pstar);

  VecRestoreSubVector(y, user->isg[0], &ustar);
  VecRestoreSubVector(y, user->isg[1], &vstar);
  VecRestoreSubVector(y, user->isg[2], &pstar);

  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGSetupHelmholtz(PC pc,DM da_fine)
{
  PetscInt       nlevels,k,PETSC_UNUSED finest;
  DM             *da_list,*daclist;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  int mcoarse, mfine;
  DMDAGetInfo(da_fine,0,&mfine,0,0,0,0,0,0,0,0,0,0,0);
  nlevels = log2(mfine);
  // nlevels = 2;

  ierr = PetscMalloc(sizeof(DM)*nlevels,&da_list);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) da_list[k] = NULL;
  ierr = PetscMalloc(sizeof(DM)*nlevels,&daclist);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) daclist[k] = NULL;

  /* finest grid is nlevels - 1 */
  finest     = nlevels - 1;
  daclist[0] = da_fine;
  PetscObjectReference((PetscObject)da_fine);
  ierr = DMCoarsenHierarchy(da_fine,nlevels-1,&daclist[1]);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) {
    da_list[k] = daclist[nlevels-1-k];
    ierr       = DMDASetUniformCoordinates(da_list[k],0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  }

  ierr = PCMGSetLevels(pc,nlevels,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);

  Mat P, R;
  int coltoinsert[6];
  int rowtoinsert;
  int row, col;
  PetscScalar v[6];

  KSP ksplevel;
  PC  pclevel;
  for (k=1; k<nlevels; k++) {
    if (k < nlevels) {
      PCMGGetSmoother(pc,k,&ksplevel);
      KSPGetPC(ksplevel,&pclevel);
      KSPSetType(ksplevel,KSPRICHARDSON);
      KSPRichardsonSetScale(ksplevel,2.0/3.0);
      PCSetType(pclevel,PCJACOBI);
    }
    DMDAGetInfo(da_list[k-1],0,&mcoarse,0,0,0,0,0,0,0,0,0,0,0);
    DMDAGetInfo(da_list[k]  ,0,&mfine  ,0,0,0,0,0,0,0,0,0,0,0);

    MatCreateSeqAIJ(PETSC_COMM_WORLD, mfine*mfine, mcoarse*mcoarse, 4, NULL, &P);
    MatCreateSeqAIJ(PETSC_COMM_WORLD, mcoarse*mcoarse, mfine*mfine, 4, NULL, &R);

    for (size_t i = 0; i < mcoarse*mcoarse; i++) {
      rowtoinsert = i;
      col = i%(int)(mcoarse);
      row = i/(mcoarse);
      coltoinsert[0] = (2*row)*(mfine)   + (2*col); v[0] = 0.125;
      coltoinsert[1] = (2*row)*(mfine)   + (2*col+1); v[1] = 0.25;
      if (2*col+2 >= mfine) {
        coltoinsert[2] = (2*row)*(mfine)   + (2*col+2-mfine); v[2] = 0.125;
      }else{
        coltoinsert[2] = (2*row)*(mfine)   + (2*col+2); v[3] = 0.125;
      }
      coltoinsert[3] = (2*row+1)*(mfine) + (2*col); v[3] = 0.125;
      coltoinsert[4] = (2*row+1)*(mfine) + (2*col+1); v[4] = 0.25;
      if (2*col+2 >= mfine) {
        coltoinsert[5] = (2*row+1)*(mfine) + (2*col+2-mfine); v[5] = 0.125;
      }else{
        coltoinsert[5] = (2*row+1)*(mfine) + (2*col+2); v[5] = 0.125;
      }
      MatSetValues(R,1,&rowtoinsert,3,coltoinsert,v,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    int crow, ccol, frow, fcol;
    for (size_t i = 0; i < mfine*mfine; i++) {
      rowtoinsert = i;
      fcol = i%(int)(mfine);
      frow = i/(mfine);
      if (fcol % 2 == 1) {
        if (frow % 2 == 1) {
          crow = frow/2; ccol = fcol/2;
          coltoinsert[0]   = crow*mcoarse+ccol; v[0] = 0.75;
          if(frow/2+1 >= mcoarse) {
            crow = frow/2+1-mcoarse; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 0.25;
          }else{
            crow = frow/2+1; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 0.25;
          }
          MatSetValues(P,1,&rowtoinsert,2,coltoinsert,v,INSERT_VALUES);
        }else{
          crow = frow/2; ccol = fcol/2;
          coltoinsert[0]   = crow*mcoarse+ccol; v[0] = 0.75;

          if(frow/2-1 < 0) {
            crow = frow/2-1+mcoarse; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 0.25;
          }else{
            crow = frow/2-1; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 0.25;
          }
          MatSetValues(P,1,&rowtoinsert,2,coltoinsert,v,INSERT_VALUES);
        }
      }else{
        if (frow % 2 == 1) {
          crow = frow/2; ccol = fcol/2;
          coltoinsert[0] = crow*mcoarse+ccol; v[0] = 3.0/8.0;

          if (fcol/2-1<0) {
            crow = frow/2; ccol = fcol/2-1+mcoarse;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 3.0/8.0;
          }else{
            crow = frow/2; ccol = fcol/2-1;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 3.0/8.0;
          }

          if (frow/2+1>=mcoarse) {
            crow = frow/2+1-mcoarse; ccol = fcol/2;
            coltoinsert[2] = crow*mcoarse+ccol; v[2] = 1.0/8.0;
          }else{
            crow = frow/2+1; ccol = fcol/2;
            coltoinsert[2] = crow*mcoarse+ccol; v[2] = 1.0/8.0;
          }

          if (frow/2+1>=mcoarse) {
            if (fcol/2-1<0) {
              crow = frow/2+1-mcoarse; ccol = fcol/2-1+mcoarse;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }else{
              crow = frow/2+1-mcoarse; ccol = fcol/2-1;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }
          }else{
            if (fcol/2-1<0) {
              crow = frow/2+1; ccol = fcol/2-1+mcoarse;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }else{
              crow = frow/2+1; ccol = fcol/2-1;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }
          }
          MatSetValues(P,1,&rowtoinsert,4,coltoinsert,v,INSERT_VALUES);
        }else{
          crow = frow/2; ccol = fcol/2;
          coltoinsert[0] = crow*mcoarse+ccol; v[0] = 3.0/8.0;

          if (frow/2-1<0) {
            crow = frow/2-1+mcoarse; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 1.0/8.0;
          }else{
            crow = frow/2-1; ccol = fcol/2;
            coltoinsert[1] = crow*mcoarse+ccol; v[1] = 1.0/8.0;
          }
          if (fcol/2-1<0) {
            crow = frow/2; ccol = fcol/2-1+mcoarse;
            coltoinsert[2] = crow*mcoarse+ccol; v[2] = 3.0/8.0;
          }else{
            crow = frow/2; ccol = fcol/2-1;
            coltoinsert[2] = crow*mcoarse+ccol; v[2] = 3.0/8.0;
          }

          if (frow/2-1<0) {
            if (fcol/2-1<0) {
              crow = frow/2-1+mcoarse; ccol = fcol/2-1+mcoarse;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }else{
              crow = frow/2-1+mcoarse; ccol = fcol/2-1;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }
          }else{
            if (fcol/2-1<0) {
              crow = frow/2-1; ccol = fcol/2-1+mcoarse;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }else{
              crow = frow/2-1; ccol = fcol/2-1;
              coltoinsert[3] = crow*mcoarse+ccol; v[3] = 1.0/8.0;
            }
          }
          MatSetValues(P,1,&rowtoinsert,4,coltoinsert,v,INSERT_VALUES);
        }
      }
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PCMGSetRestriction(pc,k,R);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,k,P);CHKERRQ(ierr);

    ierr = MatDestroy(&R);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);

  }

  /* tidy up */
  for (k=0; k<nlevels; k++) {
    ierr = DMDestroy(&da_list[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(da_list);CHKERRQ(ierr);
  ierr = PetscFree(daclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupHelmholtzKSP(UserContext *user)
{
  PetscErrorCode ierr;
  PC pc;
  ierr = KSPCreate(PETSC_COMM_WORLD,&user->ksp_helmholtz);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user->ksp_helmholtz,"helmholtz_");
  ierr = KSPSetOperators(user->ksp_helmholtz,user->subA[0],user->subA[0]);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user->ksp_helmholtz);CHKERRQ(ierr);
  ierr = KSPGetPC(user->ksp_helmholtz,&pc);
  ierr = PCSetOptionsPrefix(pc,"helmholtz_");
  ierr = PCSetType(pc,PCMG);
  ierr = PCMGSetupHelmholtz(pc,user->da_u);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);
  ierr = KSPSetUp(user->ksp_helmholtz);CHKERRQ(ierr);

  return(0);
}

static PetscErrorCode PCMGSetupPoission(PC pc,DM da_fine, UserContext* user)
{
  PetscInt       nlevels,k,PETSC_UNUSED finest;
  DM             *da_list,*daclist;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  int mcoarse, mfine;
  DMDAGetInfo(da_fine,0,&mfine,0,0,0,0,0,0,0,0,0,0,0);
  nlevels = log2(mfine);
  // nlevels = 2;

  ierr = PetscMalloc(sizeof(DM)*nlevels,&da_list);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) da_list[k] = NULL;
  ierr = PetscMalloc(sizeof(DM)*nlevels,&daclist);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) daclist[k] = NULL;

  /* finest grid is nlevels - 1 */
  finest     = nlevels - 1;
  daclist[0] = da_fine;
  PetscObjectReference((PetscObject)da_fine);
  ierr = DMCoarsenHierarchy(da_fine,nlevels-1,&daclist[1]);CHKERRQ(ierr);
  for (k=0; k<nlevels; k++) {
    da_list[k] = daclist[nlevels-1-k];
    ierr       = DMDASetUniformCoordinates(da_list[k],0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  }

  ierr = PCMGSetLevels(pc,nlevels,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_NONE);CHKERRQ(ierr);
  ierr = PCMGSetNumberSmooth(pc, 2);CHKERRQ(ierr);

  Mat P, R;
  int coltoinsert[4];
  int rowtoinsert;
  int row, col;
  PetscScalar vr[4];
  PetscScalar vp[4];
  vr[0] = vr[1] = vr[2] = vr[3] = 0.25;
  vp[0] = vp[1] = vp[2] = vp[3] = 1.0;

  KSP ksplevel;
  PC  pclevel;
  Mat oplevel;

  DMCreateMatrix(da_list[0],&oplevel);
  PCMGGetSmoother(pc,0,&ksplevel);
  ConstructLaplaceOp(da_list[0], oplevel, user);
  KSPSetOperators(ksplevel,oplevel,oplevel);
  // KSPSetType(ksplevel,KSPRICHARDSON);
  // KSPRichardsonSetScale(ksplevel,2.0/3.0);

  for (k=1; k<nlevels; k++) {
    if (k < nlevels) {
      DMCreateMatrix(da_list[k],&oplevel);
      PCMGGetSmoother(pc,k,&ksplevel);
      ConstructLaplaceOp(da_list[k], oplevel, user);
      KSPSetOperators(ksplevel,oplevel,oplevel);
      KSPGetPC(ksplevel,&pclevel);
      KSPSetType(ksplevel,KSPRICHARDSON);
      KSPRichardsonSetScale(ksplevel,2.0/3.0);
      PCSetType(pclevel,PCJACOBI);
    }

    DMDAGetInfo(da_list[k-1],0,&mcoarse,0,0,0,0,0,0,0,0,0,0,0);
    DMDAGetInfo(da_list[k]  ,0,&mfine  ,0,0,0,0,0,0,0,0,0,0,0);

    MatCreateSeqAIJ(PETSC_COMM_WORLD, mfine*mfine, mcoarse*mcoarse, 4, NULL, &P);
    MatCreateSeqAIJ(PETSC_COMM_WORLD, mcoarse*mcoarse, mfine*mfine, 4, NULL, &R);

    for (size_t i = 0; i < mcoarse*mcoarse; i++) {
      rowtoinsert = i;
      col = i%(int)(mcoarse);
      row = i/(mcoarse);
      coltoinsert[0] = (2*row)*mfine   + (2*col);
      coltoinsert[1] = (2*row+1)*mfine + (2*col);
      coltoinsert[2] = (2*row)*mfine   + (2*col+1);
      coltoinsert[3] = (2*row+1)*mfine + (2*col+1);
      MatSetValues(R,1,&rowtoinsert,4,coltoinsert,vr,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    for (size_t i = 0; i < mcoarse*mcoarse; i++) {
      rowtoinsert = i;
      col = i%(int)(mcoarse);
      row = i/(mcoarse);
      coltoinsert[0] = (2*row)*mfine   + (2*col);
      coltoinsert[1] = (2*row+1)*mfine + (2*col);
      coltoinsert[2] = (2*row)*mfine   + (2*col+1);
      coltoinsert[3] = (2*row+1)*mfine + (2*col+1);
      MatSetValues(P,4,coltoinsert,1,&rowtoinsert,vp,INSERT_VALUES);
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = PCMGSetRestriction(pc,k,R);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,k,P);CHKERRQ(ierr);

    ierr = MatDestroy(&R);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);

  }

  /* tidy up */
  for (k=0; k<nlevels; k++) {
    ierr = DMDestroy(&da_list[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(da_list);CHKERRQ(ierr);
  ierr = PetscFree(daclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupPoissionKSP(UserContext *user)
{
  PetscErrorCode ierr;
  PC pc;

  ierr = KSPCreate(PETSC_COMM_WORLD,&user->ksp_poission);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user->ksp_poission,"poission_");
  ierr = KSPSetOperators(user->ksp_poission,user->L,user->L);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user->ksp_poission);CHKERRQ(ierr);

  ierr = KSPGetPC(user->ksp_poission,&pc);
  ierr = PCSetType(pc,PCMG);
  ierr = PCMGSetupPoission(pc,user->da_u,user);
  ierr = PCSetOptionsPrefix(pc,"poission_");
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
  ierr = PCSetUp(pc);
  ierr = KSPSetUp(user->ksp_poission);CHKERRQ(ierr);

  return(0);
}
