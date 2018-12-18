#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include "operators.h"
#include "utils.h"
#include "solver.h"

extern PetscErrorCode SetupIndexSets(UserContext *user);
extern PetscErrorCode AllocateVectors(UserContext *user);
extern PetscErrorCode ExactSolution(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetF(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt);

PetscScalar ExactVelocityX(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu)
{
  return 1 - 2*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));
  // return exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y);
}

PetscScalar ExactVelocityY(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu)
{
  return 1 + 2*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));
  // return -exp(-mu*t)*sin(2*M_PI*x)*cos(2*M_PI*y);
}

/* exact solution for the pressure */
PetscScalar ExactPressure(const PetscScalar x,
                          const PetscScalar y,
                          const PetscScalar t,
                          const PetscScalar mu)
{
  return -(cos(4*M_PI*(x - t)) + cos(4*M_PI*(y- t)));
  // return -0.25*exp(-2*mu*t)*(cos(4*M_PI*x) + cos(4*M_PI*y));
}

PetscScalar CalcFu(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  return 4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
         4*M_PI*sin(4*M_PI*(x - t)) - 16*mu*M_PI*M_PI*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));
  // return M_PI*exp(-2*mu*t)*sin(4*M_PI*x) - mu*exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y) + 8*mu*M_PI*M_PI*exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y);
}

PetscScalar CalcFv(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  return -4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
          4*M_PI*sin(4*M_PI*(y - t)) + 16*mu*M_PI*M_PI*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));
  // return M_PI*exp(-2*mu*t)*sin(4*M_PI*y) + mu*exp(-mu*t)*cos(2*M_PI*y)*sin(2*M_PI*x) - 8*mu*M_PI*M_PI*exp(-mu*t)*cos(2*M_PI*y)*sin(2*M_PI*x);
}

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
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
  user.mu = 1.0;
  user.dt = 1.0/mx;
  user.t = 0;
  user.gmresiter = 0;
  user.hxvcycles = 0;
  user.hyvcycles = 0;
  user.pvcycles = 0;

  printf("mx = %d, my = %d\n", mx, my);
  printf("dt = %f\n", user.dt);

  /* matrix for all dofs */
  ierr = SetupMatrix(&user);
  ierr = SetupIndexSets(&user);
  ierr = SetupHelmholtzKSP(&user);
  ierr = SetupPoissionKSP(&user);
  ierr = AllocateVectors(&user); // initialize sol, rhs

  ierr = KSPSetOperators(ksp,user.A,user.A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);
  ierr = PCSetType(pc,PCSHELL);
  ierr = PCShellSetApply(pc,ShellPCApply);
  ierr = PCShellSetContext(pc,&user);
  ierr = PCSetFromOptions(pc);CHKERRQ(ierr);

  //initialize sol vector
  ierr = ExactSolution(&user, user.sol, 0);
  ierr = VecCopy(user.sol,user.sol_old);

  int iter;
  for (size_t i = 0; i < 1; i++) {
    ierr = SetRhs(&user,user.rhs,user.t,user.dt);
    ierr = KSPSolve(ksp,user.rhs,user.sol);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&iter);
    user.gmresiter = user.gmresiter+iter;
    ierr = VecCopy(user.sol,user.sol_old); CHKERRQ(ierr);
    user.t += user.dt;
  }

  printf("#gmresiter = %d, #vcycles = %d\n", user.gmresiter,
                                             user.hxvcycles+user.hyvcycles+user.pvcycles);

  PetscScalar solanal_norm, err;
  Vec u, utrue;
  ierr = ExactSolution(&user, user.exactsol, user.t-0.5*user.dt);
  ierr = VecGetSubVector(user.exactsol, user.isg[2], &utrue);
  ierr = VecGetSubVector(user.sol, user.isg[2], &u);
  ierr = VecNorm(utrue,NORM_2,&solanal_norm);

  ierr = VecAXPY(utrue,-1.0, u);
  ierr = VecNorm(utrue,NORM_2,&err);
  printf("p relerr = %f\n", err/solanal_norm);

  ierr = ExactSolution(&user, user.exactsol, user.t);
  ierr = VecGetSubVector(user.exactsol, user.isg[1], &utrue);
  ierr = VecGetSubVector(user.sol, user.isg[1], &u);
  ierr = VecNorm(utrue,NORM_2,&solanal_norm);
  ierr = VecAXPY(utrue,-1.0, u);
  ierr = VecNorm(utrue,NORM_2,&err);
  printf("v relerr = %f\n", err/solanal_norm);

  ierr = VecGetSubVector(user.exactsol, user.isg[0], &utrue);
  ierr = VecGetSubVector(user.sol, user.isg[0], &u);
  ierr = VecNorm(utrue,NORM_2,&solanal_norm);
  ierr = VecAXPY(utrue,-1.0, u);
  ierr = VecNorm(utrue,NORM_2,&err);
  printf("u relerr = %f\n", err/solanal_norm);

  printf("t = %f\n", user.t);
  ierr = WriteSolution(&user, user.sol); CHKERRQ(ierr);


  ierr = DMDestroy(&user.da_u);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode AllocateVectors(UserContext *user)
{
  /* solution vector x */
  VecCreate(PETSC_COMM_WORLD, &user->sol);
  VecSetSizes(user->sol, PETSC_DECIDE, 3*user->nx*user->ny);
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

PetscErrorCode SetupIndexSets(UserContext *user)
{
  /* the two index sets */
  MatNestGetISs(user->A, user->isg, NULL);
  // ISView(user->isg[0],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[1],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[2],PETSC_VIEWER_STDOUT_WORLD);
  return(0);
}

PetscErrorCode ExactSolution(UserContext *user, Vec xvec, PetscScalar t)
{
  PetscInt       row,start,end,i,j;
  PetscScalar    val,x,y;
  Vec            u,v,p;

  /* velocity part */
  VecGetSubVector(xvec, user->isg[0], &u);
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+1)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = ExactVelocityX(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(u, row, val, INSERT_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[0], &u);

  /* velocity part */
  VecGetSubVector(xvec, user->isg[1], &v);
  VecGetOwnershipRange(v, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+0.5)*user->hx;
      y = (PetscReal)(j+1)*user->hy;
      val = ExactVelocityY(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(v, row, val, INSERT_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[1], &v);

  /* pressure part */
  VecGetSubVector(xvec, user->isg[2], &p);
  VecGetOwnershipRange(p, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+0.5)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = ExactPressure(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(p, row, val, INSERT_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[2], &p);

  PetscFunctionReturn(0);
}

PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt)
{
  Vec rhsu, rhsv;
  Vec solu, solv, solp;
  // Vec Gxpold, Gypold;

  VecGetSubVector(user->sol_old, user->isg[2], &solp);
  // VecDuplicate(solu, &Gxpold);
  // VecDuplicate(solv, &Gypold);
  /* velocity part */
  VecGetSubVector(rhs, user->isg[0], &rhsu);
  VecGetSubVector(user->sol_old, user->isg[0], &solu);
  MatMult(user->Luright,solu,rhsu);
  // MatMult(user->subA[2],solp,Gxpold);
  // VecAXPY(rhsu,-Gxpold)
  VecRestoreSubVector(rhs, user->isg[0], &rhsu);

  VecGetSubVector(rhs, user->isg[1], &rhsv);
  VecGetSubVector(user->sol_old, user->isg[1], &solv);
  MatMult(user->Luright,solv,rhsv);
  VecRestoreSubVector(rhs, user->isg[1], &rhsv);
  // VecView(rhs, (PetscViewer) PETSC_VIEWER_DEFAULT);

  SetF(user, rhs, t+0.5*dt);
  // VecView(rhs, (PetscViewer) PETSC_VIEWER_DEFAULT);

  return(0);
}

PetscErrorCode SetF(UserContext *user, Vec xvec, PetscScalar t)
{
  PetscInt       row,start,end,i,j;
  PetscScalar    val,x,y;
  Vec            fu,fv;

  /* velocity part, u */
  VecGetSubVector(xvec, user->isg[0], &fu);
  VecGetOwnershipRange(fu, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+1)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = user->dt*CalcFu(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(fu, row, val, ADD_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[0], &fu);

  /* velocity part, v */
  VecGetSubVector(xvec, user->isg[1], &fv);
  VecGetOwnershipRange(fv, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny){
      x = (PetscReal)(i+0.5)*user->hx;
      y = (PetscReal)(j+1)*user->hy;
      // printf("x = %f, y = %f\n", x, y);
      val = user->dt*CalcFv(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(fv, row, val, ADD_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[1], &fv);

  PetscFunctionReturn(0);
}
