#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include "operators.h"
#include "utils.h"
#include "solver.h"


extern PetscErrorCode SetupIndexSets(UserContext *user);
extern PetscErrorCode AllocateVectors(UserContext *user, PetscInt dof);
extern PetscErrorCode ExactSolution(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetF(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt);

PetscScalar ExactVelocityX(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu)
{
  return exp(-4*mu*M_PI*M_PI*t)*sin(2*M_PI*x);
}

PetscScalar CalcFu(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  return 0.0;
}


int main(int argc,char **argv)
{
  KSP            ksp;
  UserContext    user;
  PetscErrorCode ierr;
  PetscInt       mx,my;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* da for u */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,4,4,
                      PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&user.da_u);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da_u);CHKERRQ(ierr);
  ierr = DMSetUp(user.da_u);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da_u,0,1,0,1,0,0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da_u,0,"u");CHKERRQ(ierr);
  ierr = DMDAGetInfo(user.da_u,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  user.nx = mx;
  user.ny = my;
  user.hx = 1.0/mx;
  user.hy = 1.0/my;
  user.mu = 0.1;
  user.dt = 0.1/mx;
  user.t = 0;
  printf("mx = %d, my = %d\n", mx, my);
  printf("dt = %f\n", user.dt);

  /* matrix for all dofs */
  ierr = SetupMatrix(&user);
  ierr = SetupHelmholtzKSP(&user);
  ierr = AllocateVectors(&user, 1);

  //initialize sol vector
  ierr = ExactSolution(&user, user.sol, 0);
  ierr = VecCopy(user.sol,user.sol_old);

  for (size_t i = 0; i < 1; i++) {
    ierr = SetRhs(&user,user.rhs,user.t,user.dt);
    ierr = KSPSolve(user.ksp_helmholtz,user.rhs,user.sol);CHKERRQ(ierr);
    ierr = VecCopy(user.sol,user.sol_old); CHKERRQ(ierr);
    user.t += user.dt;
  }

  PetscScalar solanal_norm, err;
  ierr = ExactSolution(&user, user.exactsol, user.t);
  ierr = VecNorm(user.exactsol,NORM_2,&solanal_norm);

  ierr = VecAXPY(user.exactsol,-1.0, user.sol);
  // VecView(user.exactsol, (PetscViewer) PETSC_VIEWER_DEFAULT);

  ierr = VecNorm(user.exactsol,NORM_2,&err);
  printf("relerr = %f\n", err/solanal_norm);

  ierr = DMDestroy(&user.da_u);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt)
{
  /* velocity part */
  MatMult(user->Luright,user->sol_old,rhs);
  SetF(user, rhs, t+0.5*dt);

  return(0);
}

PetscErrorCode SetF(UserContext *user, Vec xvec, PetscScalar t)
{
  PetscInt       row,start,end,i,j;
  PetscScalar    val,x,y;
  Vec            fu,fv;

  /* velocity part, u */
  VecGetOwnershipRange(xvec, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+1)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = user->dt*CalcFu(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(xvec, row, val, ADD_VALUES);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode AllocateVectors(UserContext *user, PetscInt dof)
{
  /* solution vector x */
  VecCreate(PETSC_COMM_WORLD, &user->sol);
  VecSetSizes(user->sol, PETSC_DECIDE, dof*user->nx*user->ny);
  VecSetType(user->sol, VECSEQ);
  VecDuplicate(user->sol,&user->sol_old);
  VecDuplicate(user->sol, &user->exactsol);
  VecDuplicate(user->sol, &user->rhs);

  return(0);
}

PetscErrorCode GetPosition(UserContext *user, PetscInt row, PetscInt *i, PetscInt *j)
{
  PetscInt n;

  /* cell number n=j*nx+i has position (i,j) in grid */
  n  = row%(user->nx*user->ny);
  *i = n%user->nx;
  *j = (n-(*i))/user->nx;

  return(0);
}

PetscErrorCode ExactSolution(UserContext *user, Vec xvec, PetscScalar t)
{
  PetscInt       row,start,end,i,j;
  PetscScalar    val,x,y;

  /* velocity part */
  VecGetOwnershipRange(xvec, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+1)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = ExactVelocityX(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(xvec, row, val, INSERT_VALUES);
  }
  PetscFunctionReturn(0);
}
