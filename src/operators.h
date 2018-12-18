#ifndef OPERATOR_H
#define OPERATOR_H

typedef struct {
  DM da_u;                /* distributed array data structure */
  KSP ksp_helmholtz;
  KSP ksp_poission;
  Mat subA[9];
  Mat L;
  Mat A;
  Mat Luright;
  Vec ustar,vstar;
  Vec sol,exactsol,sol_old,rhs;
  PetscInt nx, ny;
  PetscScalar hx, hy;
  PetscScalar mu;
  PetscScalar t,dt;
  IS isg[3];
  PetscInt gmresiter, hxvcycles, hyvcycles, pvcycles;

} UserContext;

PetscErrorCode SetupMatrix(UserContext *user);
PetscErrorCode ConstructLaplaceOp(DM da, Mat L, void *ctx);
PetscErrorCode ConstructLuOp(DM da, Mat subA, void *ctx);
PetscErrorCode ConstructGxOp(DM da, Mat subA, void *ctx);
PetscErrorCode ConstructGyOp(DM da, Mat subA, void *ctx);
PetscErrorCode ConstructDxOp(DM da, Mat subA, void *ctx);
PetscErrorCode ConstructDyOp(DM da, Mat subA, void *ctx);
PetscErrorCode ConstructLurightOp(DM da, Mat Luright, void *ctx);



#endif
