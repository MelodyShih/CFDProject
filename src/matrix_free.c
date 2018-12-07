/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/



/*
Added at the request of Marc Garbey.

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
// #include <petscviewermatlab.h>

typedef struct {
  DM da_u;                /* distributed array data structure */
  Mat subA[9];
  Mat A;
  Mat Luright;
  Vec u,v,p;
  Vec sol,exactsol,sol_old,rhs;
  PetscInt nx, ny;
  PetscScalar hx, hy;
  PetscScalar mu;
  PetscScalar t,dt;
  IS isg[3];

} UserContext;

extern PetscErrorCode ConstructLuOp(DM da, Mat subA, void *ctx);
extern PetscErrorCode ConstructGxOp(DM da, Mat subA, void *ctx);
extern PetscErrorCode ConstructGyOp(DM da, Mat subA, void *ctx);
extern PetscErrorCode ConstructDxOp(DM da, Mat subA, void *ctx);
extern PetscErrorCode ConstructDyOp(DM da, Mat subA, void *ctx);
extern PetscErrorCode ConstructLurightOp(DM da, Mat Luright, void *ctx);


// extern PetscErrorCode MatMult_Laplacian(Mat A,Vec x,Vec y);
extern PetscErrorCode SetupMatrix(UserContext *user);
extern PetscErrorCode SetupIndexSets(UserContext *user);
extern PetscErrorCode SetupVectors(UserContext *user);
extern PetscErrorCode ExactSolution(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetF(UserContext *user, Vec xvec, PetscScalar t);
extern PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt);

extern PetscErrorCode WriteSolution(UserContext *user, Vec b);

PetscScalar ExactVelocityX(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t)
{
  return 1 - 2*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));;
}

PetscScalar ExactVelocityY(const PetscScalar x,
                           const PetscScalar y,
                           const PetscScalar t)
{
  return 1 + 2*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));;
}

/* exact solution for the pressure */
PetscScalar ExactPressure(const PetscScalar x,
                          const PetscScalar y,
                          const PetscScalar t)
{
  return -(cos(4*M_PI*(x - t)) + cos(4*M_PI*(y- t)));
}

PetscScalar CalcFu(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  return 4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
         4*M_PI*sin(4*M_PI*(x - t)) - 16*mu*M_PI*M_PI*cos(2*M_PI*(x - t))*sin(2*M_PI*(y - t));
}

PetscScalar CalcFv(const PetscScalar x,
                   const PetscScalar y,
                   const PetscScalar t,
                   const PetscScalar mu)
{
  return -4*M_PI*(cos(2*M_PI*(x - t))*cos(2*M_PI*(y - t)) - sin(2*M_PI*(x - t))*sin(2*M_PI*(y - t))) +
          4*M_PI*sin(4*M_PI*(y - t)) + 16*mu*M_PI*M_PI*sin(2*M_PI*(x - t))*cos(2*M_PI*(y - t));
}

int main(int argc,char **argv)
{
  KSP            ksp;
  UserContext    user;
  PetscErrorCode ierr;
  PetscInt       mx,my,dof;
  Vec            b,x;
  Mat            Amat_shell;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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
  user.dt = 1.0/mx;
  user.t = 0;

  /* matrix for all dofs */
  ierr = SetupMatrix(&user);
  ierr = SetupIndexSets(&user);
  ierr = SetupVectors(&user); // initialize sol, rhs

  // ierr = MatCreateShell(PETSC_COMM_WORLD,dof*mx*my,dof*mx*my,PETSC_DECIDE,PETSC_DECIDE,&user,&Amat_shell);
  // ierr = MatShellSetOperation(Amat_shell,MATOP_MULT,(void (*)(void))MatMult_Laplacian);
  // ierr = KSPSetOperators(ksp,Amat_shell,Amat_shell);CHKERRQ(ierr);
  // ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  // ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp,user.A,user.A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  for (size_t i = 0; i < mx; i++) {
    ierr = KSPSolve(ksp,user.rhs,user.sol);CHKERRQ(ierr);
    user.t += user.dt;
    ierr = VecCopy(user.sol,user.sol_old); CHKERRQ(ierr);
    ierr = SetRhs(&user,user.rhs,user.t,user.dt);
  }

  printf("t = %f\n", user.t);
  ierr = WriteSolution(&user, user.sol); CHKERRQ(ierr);


  ierr = DMDestroy(&user.da_u);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

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

PetscErrorCode SetupVectors(UserContext *user)
{
  /* solution vector x */
  VecCreate(PETSC_COMM_WORLD, &user->sol);
  VecSetSizes(user->sol, PETSC_DECIDE, 3*user->nx*user->ny);
  VecSetType(user->sol, VECMPI);
  ExactSolution(user, user->sol, 0);

  VecDuplicate(user->sol,&user->sol_old);
  VecCopy(user->sol,user->sol_old);
  /*  VecSetRandom(user->x, NULL); */
  /*  VecView(user->x, (PetscViewer) PETSC_VIEWER_DEFAULT); */

  /* exact solution y */
  VecDuplicate(user->sol, &user->exactsol);
  printf("mx = %d, my = %d\n", user->nx, user->ny);
  ExactSolution(user, user->exactsol, 0);
  // VecView(user->exactsol, (PetscViewer) PETSC_VIEWER_DEFAULT);

  /* rhs vector b */
  VecDuplicate(user->sol, &user->rhs);
  SetRhs(user, user->rhs, 0, user->dt);
  /*VecView(s->b, (PetscViewer) PETSC_VIEWER_DEFAULT);*/
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
  Vec            u,v,p;

  /* velocity part */
  VecGetSubVector(xvec, user->isg[0], &u);
  VecGetOwnershipRange(u, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+1)*user->hx;
      y = (PetscReal)(j+0.5)*user->hy;
      val = ExactVelocityX(x,y,t);
    } else {
      val = 0;
    }
    VecSetValue(u, row, val, ADD_VALUES);
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
      val = ExactVelocityY(x,y,t);
    } else {
      val = 0;
    }
    VecSetValue(v, row, val, ADD_VALUES);
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
      val = ExactPressure(x,y,t);
    } else {
      val = 0;
    }
    VecSetValue(p, row, val, ADD_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[2], &p);

  PetscFunctionReturn(0);
}

PetscErrorCode SetRhs(UserContext *user, Vec rhs, PetscScalar t, PetscScalar dt)
{
  Vec rhsu, rhsv;
  Vec solu, solv;

  /* velocity part */
  VecGetSubVector(rhs, user->isg[0], &rhsu);
  VecGetSubVector(user->sol_old, user->isg[0], &solu);
  MatMult(user->Luright,solu,rhsu);
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

  /* velocity part */
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

  /* velocity part */
  VecGetSubVector(xvec, user->isg[1], &fv);
  VecGetOwnershipRange(fv, &start, &end);
  for (row = start; row < end; row++) {
    GetPosition(user,row,&i,&j);
    if (row < user->nx*user->ny) {
      x = (PetscReal)(i+0.5)*user->hx;
      y = (PetscReal)(j+1)*user->hy;
      val = user->dt*CalcFv(x,y,t,user->mu);
    } else {
      val = 0;
    }
    VecSetValue(fv, row, val, ADD_VALUES);
  }
  VecRestoreSubVector(xvec, user->isg[1], &fv);

  PetscFunctionReturn(0);
}


#if 0
PetscErrorCode MatMult_Laplacian(Mat A,Vec x,Vec y)
{
  UserContext *user;
  PetscFunctionBeginUser;

  MatShellGetContext(A, &user);
  MatMult(user->Lu, x, y);
  // MatMult(user->Lu, x, y);
  PetscFunctionReturn(0);
}
#endif


PetscErrorCode SetupIndexSets(UserContext *user)
{
  /* the two index sets */
  MatNestGetISs(user->A, user->isg, NULL);
  // ISView(user->isg[0],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[1],PETSC_VIEWER_STDOUT_WORLD);
  // ISView(user->isg[2],PETSC_VIEWER_STDOUT_WORLD);

  /*  ISView(isg[1],PETSC_VIEWER_STDOUT_WORLD); */
  return(0);
}

PetscErrorCode SetupMatrix(UserContext *user)
{
  PetscErrorCode ierr;

  for (size_t i = 0; i < 9; i++) {
    ierr = DMCreateMatrix(user->da_u,&user->subA[i]); CHKERRQ(ierr);
  }
  ierr = ConstructLuOp(user->da_u, user->subA[0], user);CHKERRQ(ierr);
  ierr = ConstructLuOp(user->da_u, user->subA[4], user);CHKERRQ(ierr);
  ierr = ConstructDxOp(user->da_u, user->subA[6], user);CHKERRQ(ierr);
  ierr = ConstructDyOp(user->da_u, user->subA[7], user);CHKERRQ(ierr);
  ierr = ConstructGxOp(user->da_u, user->subA[2], user);CHKERRQ(ierr);
  ierr = ConstructGyOp(user->da_u, user->subA[5], user);CHKERRQ(ierr);

  ierr = MatCreateNest(PETSC_COMM_WORLD,3,NULL,3,NULL,user->subA,&user->A);
  ierr = MatViewFromOptions(user->A,NULL,"-view_mat");CHKERRQ(ierr);
  PetscInt m, n;
  ierr = MatGetSize(user->A,&m,&n);
  printf("m = %d, n = %d\n", m, n);

  ierr = DMCreateMatrix(user->da_u,&user->Luright); CHKERRQ(ierr);
  ierr = ConstructLurightOp(user->da_u,user->Luright, user);CHKERRQ(ierr);

  return(0);
}

PetscErrorCode ConstructLuOp(DM da, Mat subA, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5], mu, dt;
  PetscReal      Hx,Hy,HxHy;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx);
  Hy        = 1.0 / (PetscReal)(my);
  HxHy      = Hx*Hy;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -0.5*mu*dt/HxHy;              col[0].i = i;   col[0].j = j-1;
      v[1] = -0.5*mu*dt/HxHy;              col[1].i = i-1; col[1].j = j;
      v[2] = 1 + 4.0*0.5*mu*dt/HxHy;       col[2].i = i;   col[2].j = j;
      v[3] = -0.5*mu*dt/HxHy;              col[3].i = i+1; col[3].j = j;
      v[4] = -0.5*mu*dt/HxHy;              col[4].i = i;   col[4].j = j+1;
      ierr = MatSetValuesStencil(subA,1,&row,5,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(subA,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(subA,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructLurightOp(DM da, Mat Luright, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5], mu, dt;
  PetscReal      Hx,Hy,HxHy;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx);
  Hy        = 1.0 / (PetscReal)(my);
  HxHy      = Hx*Hy;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = 0.5*mu*dt/HxHy;              col[0].i = i;   col[0].j = j-1;
      v[1] = 0.5*mu*dt/HxHy;              col[1].i = i-1; col[1].j = j;
      v[2] = 1 - 0.5*4*mu*dt/HxHy;        col[2].i = i;   col[2].j = j;
      v[3] = 0.5*mu*dt/HxHy;              col[3].i = i+1; col[3].j = j;
      v[4] = 0.5*mu*dt/HxHy;              col[4].i = i;   col[4].j = j+1;
      ierr = MatSetValuesStencil(Luright,1,&row,5,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(Luright,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Luright,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatViewFromOptions(Luright,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(Luright,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructDxOp(DM da, Mat subA, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      mu;
  PetscReal      dt;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx);
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -dt/Hx;              col[0].i = i;   col[0].j = j;
      v[1] =  dt/Hx;              col[1].i = i-1; col[1].j = j;
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatViewFromOptions(subA,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(subA,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructDyOp(DM da, Mat subA, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      mu;
  PetscReal      dt;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hy;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hy        = 1.0 / (PetscReal)(my);
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -dt/Hy;              col[0].i = i;   col[0].j = j;
      v[1] =  dt/Hy;              col[1].i = i;   col[1].j = j-1;
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatViewFromOptions(subA,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(subA,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructGyOp(DM da, Mat subA, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      mu;
  PetscReal      dt;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hy;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hy        = 1.0 / (PetscReal)(my);
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] =  dt/Hy;              col[0].i = i;   col[0].j = j+1;
      v[1] = -dt/Hy;              col[1].i = i;   col[1].j = j;
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatViewFromOptions(subA,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(subA,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConstructGxOp(DM da, Mat subA, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      mu;
  PetscReal      dt;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  mu = user->mu;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx);
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] =  dt/Hx;              col[0].i = i+1; col[0].j = j;
      v[1] = -dt/Hx;              col[1].i = i;   col[1].j = j;
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatViewFromOptions(subA,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(subA,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
