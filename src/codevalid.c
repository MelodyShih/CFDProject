#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include "operators.h"
#include "utils.h"
#include "solver.h"

PetscScalar ExactVelocityX(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu)
{
  return exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y);
  // return exp(-mu*8*M_PI*M_PI*t)*sin(2*M_PI*x)*sin(2*M_PI*y);
  // return exp(-t)*sin(2*M_PI*x)*sin(2*M_PI*y);
  // return 1 - 2*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));
}

PetscScalar ExactVelocityY(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu)
{
  return -exp(-mu*t)*sin(2*M_PI*x)*cos(2*M_PI*y);
  // return 1 + 2*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));
}

PetscScalar ExacthelmholtzX(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu,
                           const PetscScalar dt)
{
  return (1 + 0.5*mu*dt*8*M_PI*M_PI)*exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y);
}

PetscScalar ExacthelmholtzY(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t,
                           const PetscScalar mu,
                           const PetscScalar dt)
{
  return (-1 - 0.5*mu*dt*8*M_PI*M_PI)*exp(-mu*t)*sin(2*M_PI*x)*cos(2*M_PI*y);
}

PetscScalar ExactdVelocityXdx(const PetscScalar x,
                              const PetscScalar y,
                              const PetscScalar t,
                              const PetscScalar mu)
{
  return 2*M_PI*exp(-mu*t)*sin(2*M_PI*x)*sin(2*M_PI*y);//-dudx
}

PetscScalar ExactdVelocityYdx(const PetscScalar x,
                              const PetscScalar y,
                              const PetscScalar t,
                              const PetscScalar mu)
{
  return -2*M_PI*exp(-mu*t)*sin(2*M_PI*x)*sin(2*M_PI*y);//-dvdy
}

/* exact solution for the pressure */
PetscScalar ExactPressure(const PetscScalar x,
                          const PetscScalar y,
                          const PetscScalar t,
                          const PetscScalar mu)
{
  return -0.25*exp(-2*mu*t)*(cos(4*M_PI*x) + cos(4*M_PI*y));
}

PetscScalar dExactPressuredx(const PetscScalar x,
                             const PetscScalar y,
                             const PetscScalar t,
                             const PetscScalar mu)
{
  return M_PI*exp(-2*mu*t)*sin(4*M_PI*(x - t));
}

PetscScalar dExactPressuredy(const PetscScalar x,
                            const PetscScalar y,
                            const PetscScalar t,
                            const PetscScalar mu)
{
  return M_PI*exp(-2*mu*t)*sin(4*M_PI*(y - t));
}
PetscScalar CalcFu(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  // return -exp(-t)*sin(2*M_PI*x)*sin(2*M_PI*y);
  // return 4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
  //        4*M_PI*sin(4*M_PI*(x - t)) - 16*mu*M_PI*M_PI*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));
  return M_PI*exp(-2*mu*t)*sin(4*M_PI*x) - mu*exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y) + 8*mu*M_PI*M_PI*exp(-mu*t)*cos(2*M_PI*x)*sin(2*M_PI*y);

}

PetscScalar CalcFv(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  // return -4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
  //           4*M_PI*sin(4*M_PI*(y - t)) + 16*mu*M_PI*M_PI*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));
  return M_PI*exp(-2*mu*t)*sin(4*M_PI*y) + mu*exp(-mu*t)*cos(2*M_PI*y)*sin(2*M_PI*x) - 8*mu*M_PI*M_PI*exp(-mu*t)*cos(2*M_PI*y)*sin(2*M_PI*x);
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

PetscErrorCode SetupIndexSets(UserContext *user)
{
  /* the two index sets */
  MatNestGetISs(user->A, user->isg, NULL);
  // ISView(user->isg[0],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[1],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[2],PETSC_VIEWER_STDOUT_WORLD);
  return(0);
}
PetscErrorCode ExactSolutionU(UserContext *user, Vec xvec, PetscScalar t)
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
  return 0;
}

PetscErrorCode SetFu(UserContext *user, Vec fu, PetscScalar t)
{
  PetscInt       row,start,end,i,j;
  PetscScalar    val,x,y;

  /* velocity part, u */
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
  return 0;
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
  user.mu = 0.0;
  user.dt = 1.0/mx;
  user.t = 0;

  printf("mx = %d, my = %d\n", mx, my);
  printf("dt = %f\n", user.dt);

  /* matrix for all dofs */
  ierr = SetupMatrix(&user);
  ierr = SetupIndexSets(&user);
  ierr = AllocateVectors(&user); // initialize sol, rhs

  Vec u, dudxtrue, dudxcal;
  PetscScalar err, dudxnorm;

  VecCreate(PETSC_COMM_WORLD, &u);
  VecSetSizes(u,PETSC_DECIDE, user.nx*user.ny);
  VecSetType(u,VECSEQ);
  VecDuplicate(u,&dudxtrue);
  VecDuplicate(u,&dudxcal);

#if 0
  /* Dx, Dy */
  // initialize velocity u
  int start,end,row,i,j;
  PetscScalar x,y,val;
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+1)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = ExactVelocityX(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(u, row, val, INSERT_VALUES);

    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = ExactdVelocityXdx(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }

  // apply Dx operator
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[6],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("dudx relerr = %3.3e\n", err/dudxnorm);

  /* I - mu*dt/2*L */
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+1)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = ExacthelmholtzX(x,y,user.t,user.mu,user.dt);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[0],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("u - 0.5*mu*L_x relerr = %3.3e\n", err/dudxnorm);

  // initialize velocity v
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+1)*user.hy;
      val = ExactVelocityY(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(u, row, val, INSERT_VALUES);

    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = ExactdVelocityYdx(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }

  // apply Dy operator
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[7],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("dvdy relerr = %3.3e\n", err/dudxnorm);

  /* I - mu*dt/2*L */
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+1)*user.hy;
      val = ExacthelmholtzY(x,y,user.t,user.mu,user.dt);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[4],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("u - 0.5*mu*L_y relerr = %3.3e\n", err/dudxnorm);

  /* Gx, Gy */
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = ExactPressure(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(u, row, val, INSERT_VALUES);

    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+1)*user.hx;
      y = (PetscReal)(j+0.5)*user.hy;
      val = dExactPressuredx(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[2],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("dpdx relerr = %3.3e\n", err/dudxnorm);

  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(&user,row,&i,&j);
    if (row < user.nx*user.ny) {
      x = (PetscReal)(i+0.5)*user.hx;
      y = (PetscReal)(j+1)*user.hy;
      val = dExactPressuredy(x,y,user.t,user.mu);
    } else {
      val = 0;
    }
    VecSetValue(dudxtrue, row, val, INSERT_VALUES);
  }
  VecNorm(dudxtrue,NORM_2,&dudxnorm);
  MatMult(user.subA[5],u,dudxcal);
  VecAXPY(dudxtrue,-1.0,dudxcal);
  VecNorm(dudxtrue,NORM_2,&err);
  printf("dpdy relerr = %3.3e\n", err/dudxnorm);
#endif
#if 0
  KSPSetOperators(ksp,user.subA[0],user.subA[0]);CHKERRQ(ierr);
  KSPSetFromOptions(ksp);CHKERRQ(ierr);
  KSPSetUp(ksp);CHKERRQ(ierr);

  ExactSolutionU(&user, u, 0);

  Vec rhs;
  VecDuplicate(u, &rhs);
  for (size_t i = 0; i < mx; i++) {
    ierr = MatMult(user.Luright, u, rhs);CHKERRQ(ierr);
    ierr = SetFu(&user, rhs, user.t+0.5*user.dt);
    ierr = KSPSolve(ksp,rhs,u);CHKERRQ(ierr);
    user.t += user.dt;
  }

  // VecView(u, (PetscViewer) PETSC_VIEWER_DEFAULT);
  ierr = ExactSolutionU(&user, dudxtrue, user.t);
  // VecView(dudxtrue, (PetscViewer) PETSC_VIEWER_DEFAULT);
  ierr = VecNorm(dudxtrue,NORM_2,&dudxnorm);
  ierr = VecAXPY(dudxtrue,-1.0, u);
  ierr = VecNorm(dudxtrue,NORM_2,&err);
  printf("u relerr = %e\n", err/dudxnorm);
  printf("%f\n", dudxnorm);
  printf("t = %f\n", user.t);
#endif

  ierr = KSPSetOperators(ksp,user.A,user.A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  //initialize sol vector
  ierr = ExactSolution(&user, user.sol, 0);
  ierr = ExactSolution(&user, user.exactsol, user.t+user.dt);
  ierr = VecCopy(user.sol,user.sol_old);


  Vec ans;
  VecDuplicate(user.sol, &ans);
  SetRhs(&user,user.rhs,user.t,user.dt);
  MatMult(user.A,user.exactsol,ans);

  ierr = VecNorm(user.rhs,NORM_2,&dudxnorm);
  ierr = VecAXPY(ans,-1.0, user.rhs);
  ierr = VecNorm(ans,NORM_2,&err);
  printf("Autrue - b relerr = %e\n", err);
  printf("%f\n", dudxnorm);
  printf("t = %f\n", user.t);


  ierr = DMDestroy(&user.da_u);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
