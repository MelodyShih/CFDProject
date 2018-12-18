/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Neuman boundary conditions
        dp/dx = 0 for x = 0, x = 1.
        dp/dy = 0 for y = 0, y = 1.

     Contributed by Michael Boghosian <boghmic@iit.edu>, 2008,
         based on petsc/src/ksp/ksp/examples/tutorials/ex29.c and ex32.c

     Compare to ex66.c

     Example of Usage:
          ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause -1
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor
          mpiexec -n 4 ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view
*/

static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

extern PetscErrorCode ComputeJacobian(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef struct {
  PetscScalar uu, tt;
} UserContext;

PetscScalar CalcMu(const PetscScalar x,
                   const PetscScalar y)
{
  return 2+sin(2*M_PI*x)*cos(2*M_PI*y);
  // if (pow(x-0.5,2) + pow(y-0.5,2) < pow(0.25,2) ) {
  //   return 1.0;
  // }else{
  //   return 1.0;
  // }
  // return 1.0;
}

PetscErrorCode ConstructRestriction(DM da_coarse, Mat R)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xm,ym,xs,ys;
  PetscScalar    v[4];
  MatStencil     row[4], col;

  PetscFunctionBeginUser;
  ierr      = DMDAGetCorners(da_coarse,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  printf("%d, %d\n", xm, ym);
  // ierr =  MatSetStencil(R,2,,const PetscInt starts[],PetscInt dof)
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      col.i = i; col.j = j;
      v[0] = 0.25;              row[0].i = i;   row[0].j = j;
      v[1] = 0.25;              row[1].i = 2*i+1; row[1].j = 2*j;
      v[2] = 0.25;              row[2].i = 2*i;   row[2].j = 2*j+1;
      v[3] = 0.25;              row[3].i = 2*i+1; row[3].j = 2*j+1;
      ierr = MatSetValuesStencil(R,4,row,1,&col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(R,NULL,"-view_mat");CHKERRQ(ierr);

  MatNullSpace nullspace;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(R,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMGSetupViaCoarsen(PC pc,DM da_fine)
{
  PetscInt       nlevels,k,PETSC_UNUSED finest;
  DM             *da_list,*daclist;
  Mat            R;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  nlevels = 3;

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
  ierr = PCMGSetType(pc,PC_MG_FULL);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);

  for (k=1; k<nlevels; k++) {
    ierr = DMCreateInterpolation(da_list[k-1],da_list[k],&R,NULL);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,k,R);CHKERRQ(ierr);

    Mat P, RR;
    int mx, my;

    ierr = MatGetSize(R,&mx,&my);
    printf("mx = %d, my = %d\n", mx, my);
    MatCreateSeqAIJ(PETSC_COMM_WORLD, my, mx, 4, NULL, &P);
    MatCreateSeqAIJ(PETSC_COMM_WORLD, mx, my, 4, NULL, &RR);

    ierr = MatGetSize(P,&mx,&my);
    int coltoinsert[4];
    int rowtoinsert;
    int row, col;
    PetscScalar vp[4];
    vp[0] = vp[1] = vp[2] = vp[3] = 0.25;
    PetscScalar vr[4];
    vr[0] = vr[1] = vr[2] = vr[3] = 1.0;

    for (size_t i = 0; i < mx; i++) {
      rowtoinsert = i;
      col = i%(int) sqrt(mx);
      row = i/sqrt(mx);
      coltoinsert[0] = (2*col)*sqrt(my) + (2*row);
      coltoinsert[1] = (2*col+1)*sqrt(my) + (2*row);
      coltoinsert[2] = (2*col)*sqrt(my) + (2*row+1);
      coltoinsert[3] = (2*col+1)*sqrt(my) + (2*row+1);
      MatSetValues(P,1,&rowtoinsert,4,coltoinsert,vp,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    // ierr = MatView(P,(PetscViewer) PETSC_VIEWER_DEFAULT);

    for (size_t i = 0; i < mx; i++) {
      rowtoinsert = i;
      col = i%(int) sqrt(mx);
      row = i/sqrt(mx);
      coltoinsert[0] = (2*col)*sqrt(my) + (2*row);
      coltoinsert[1] = (2*col+1)*sqrt(my) + (2*row);
      coltoinsert[2] = (2*col)*sqrt(my) + (2*row+1);
      coltoinsert[3] = (2*col+1)*sqrt(my) + (2*row+1);
      MatSetValues(RR,4,coltoinsert,1,&rowtoinsert,vr,INSERT_VALUES);
    }

    ierr = MatAssemblyBegin(RR,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(RR,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    // ierr = MatView(RR,(PetscViewer) PETSC_VIEWER_DEFAULT);
    ierr = PCMGSetRestriction(pc,k,P);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,k,RR);CHKERRQ(ierr);

    ierr = MatDestroy(&R);CHKERRQ(ierr);
  }

  /* tidy up */
  for (k=0; k<nlevels; k++) {
    ierr = DMDestroy(&da_list[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(da_list);CHKERRQ(ierr);
  ierr = PetscFree(daclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  Vec            x,b;
  UserContext    user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,(DM)da);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b); CHKERRQ(ierr);

  user.uu     = 10.0;
  user.tt     = 10.0;

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeJacobian,&user);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);
  ierr = PCSetType(pc,PCMG);
  ierr = PCMGSetupViaCoarsen(pc,da);
  // Mat Imat;
  // PetscInt mx, my;
  // ierr = PCMGGetInterpolation(pc,1,&Imat);
  // ierr = MatGetSize(Imat,&mx,&my);
  // printf("mx = %d, my = %d\n", mx, my);
  // ierr = MatView(Imat,(PetscViewer) PETSC_VIEWER_DEFAULT);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  PetscScalar bnorm, errnorm;
  Mat A;
  Vec Ax;
  ierr = VecNorm(b,NORM_2,&bnorm);
  ierr = VecDuplicate(x,&Ax);
  ierr = KSPGetOperators(ksp,&A,NULL);
  MatMult(A,x,Ax);
  VecAXPY(b,-1.0,Ax);
  ierr = VecNorm(b,NORM_2,&errnorm);

  printf("relative err = %f\n", errnorm/bnorm);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,M,N,xm,ym,xs,ys;
  PetscScalar    Hx,Hy,pi,uu,tt,x,y;
  PetscScalar    **array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  uu   = user->uu; tt = user->tt;
  pi   = 4*atan(1.0);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);
  // printf("M = %d, N = %d\n", M, N);

  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr); /* Fine grid */
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x = ((PetscReal)i+0.5)*Hx;
      y = ((PetscReal)j+0.5)*Hy;
      // array[j][i] = exp(-100*(x*x + y*y));
      array[j][i] = -PetscCosScalar(uu*pi*((PetscReal)i+0.5)*Hx)*PetscCosScalar(tt*pi*((PetscReal)j+0.5)*Hy)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys;
  PetscScalar    v[5], Hx, Hy, HxdHy, HydHx;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;
  PetscScalar    x,y,xlow,ylow,xhigh,yhigh;

  PetscFunctionBeginUser;
  ierr  = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;

  ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      x = Hx*(PetscReal)i - 0.5*Hx;
      y = Hy*(PetscReal)j - 0.5*Hy;
      xlow = Hx*(PetscReal)i;
      xhigh = xlow + Hx;
      ylow = Hy*(PetscReal)j;
      yhigh = ylow + Hy;

      v[0] = -CalcMu(x,ylow)*HxdHy;                  col[0].i = i;   col[0].j = j-1;
      v[1] = -CalcMu(xlow,y)*HydHx;                  col[1].i = i-1; col[1].j = j;
      v[2] = (CalcMu(xlow,y) + CalcMu(xhigh,y))*HydHx +
             (CalcMu(x,ylow) + CalcMu(x,yhigh))*HydHx;  col[2].i = i;   col[2].j = j;
      v[3] = -CalcMu(xhigh,y)*HydHx;                col[3].i = i+1; col[3].j = j;
      v[4] = -CalcMu(x,yhigh)*HxdHy;                  col[4].i = i;   col[4].j = j+1;
      ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type svd -ksp_view

   test:
      suffix: 2
      nsize: 4
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view

   test:
      suffix: 3
      nsize: 2
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 5 -mg_coarse_ksp_type cg -mg_coarse_ksp_converged_reason -mg_coarse_ksp_rtol 1e-2 -mg_coarse_ksp_max_it 5 -mg_coarse_pc_type none -pc_mg_levels 2 -ksp_type pipefgmres -ksp_pipefgmres_shift 1.5

TEST*/
