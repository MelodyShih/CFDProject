#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "operators.h"

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
  // MatView(user->subA[2], (PetscViewer) PETSC_VIEWER_DEFAULT);

  ierr = ConstructGyOp(user->da_u, user->subA[5], user);CHKERRQ(ierr);
  // printf("Gy = \n");
  // MatView(user->subA[0], (PetscViewer) PETSC_VIEWER_DEFAULT);
  // Mat test;
  // MatMatMult(user->subA[2],user->subA[6],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&test);
  // // MatTranspose(user->subA[0], MAT_INITIAL_MATRIX,&test);
  // MatView(test, (PetscViewer) PETSC_VIEWER_DEFAULT);
  // MatMatMult(user->subA[5],user->subA[7],MAT_REUSE_MATRIX,PETSC_DEFAULT,&test);
  // MatView(test, (PetscViewer) PETSC_VIEWER_DEFAULT);

  ierr = MatCreateNest(PETSC_COMM_WORLD,3,NULL,3,NULL,user->subA,&user->A);
  ierr = MatViewFromOptions(user->A,NULL,"-view_mat");CHKERRQ(ierr);

  ierr = DMCreateMatrix(user->da_u,&user->Luright); CHKERRQ(ierr);
  ierr = ConstructLurightOp(user->da_u,user->Luright, user);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user->da_u,&user->L); CHKERRQ(ierr);
  ierr = ConstructLaplaceOp(user->da_u,user->L,user);CHKERRQ(ierr);
  // MatView(user->L, (PetscViewer) PETSC_VIEWER_DEFAULT);


  return(0);
}

PetscErrorCode ConstructLaplaceOp(DM da, Mat L, void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5], dt;
  PetscReal      Hx,Hy,HxHy;
  MatStencil     row, col[5];

  PetscFunctionBeginUser;
  dt = user->dt;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx);
  Hy        = 1.0 / (PetscReal)(my);
  HxHy      = Hx*Hy;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -dt*dt/HxHy;              col[0].i = i;   col[0].j = j-1;
      v[1] = -dt*dt/HxHy;              col[1].i = i-1; col[1].j = j;
      v[2] = 4.0*dt*dt/HxHy;           col[2].i = i;   col[2].j = j;
      v[3] = -dt*dt/HxHy;              col[3].i = i+1; col[3].j = j;
      v[4] = -dt*dt/HxHy;              col[4].i = i;   col[4].j = j+1;
      ierr = MatSetValuesStencil(L,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(L,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(L,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
      v[2] = 1 + 0.5*4.0*mu*dt/HxHy;       col[2].i = i;   col[2].j = j;
      v[3] = -0.5*mu*dt/HxHy;              col[3].i = i+1; col[3].j = j;
      v[4] = -0.5*mu*dt/HxHy;              col[4].i = i;   col[4].j = j+1;
      ierr = MatSetValuesStencil(subA,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValuesStencil(Luright,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValuesStencil(subA,1,&row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
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
