/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This is a nonlinear eigenvalue problem. When p=2, it is reduced to a linear Laplace eigenvalue
   problem.

   -\nabla\cdot(|\nabla u|^{p-2} \nabla u) = k |u|^{p-2} u in (0,1)x(0,1),

   u = 0 on the entire boundary.

   The code is implemented based on DMPlex using Q1 FEM on a quadrilateral mesh. In this code, we consider p=3.

   Contributed  by Fande Kong fdkong.jd@gmail.com
*/

static char help[] = "Nonlinear inverse iteration for A(x)*x=lambda*B(x)*x.\n\n";

#include <slepceps.h>
#include <petscdmplex.h>
#include <petscds.h>

PetscErrorCode CreateSquareMesh(MPI_Comm,DM*);
PetscErrorCode SetupDiscretization(DM);
PetscErrorCode FormJacobianA(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionA(SNES,Vec,Vec,void*);
PetscErrorCode MatMult_A(Mat A,Vec x,Vec y);
PetscErrorCode FormJacobianB(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionB(SNES,Vec,Vec,void*);
PetscErrorCode MatMult_B(Mat A,Vec x,Vec y);
PetscErrorCode FormFunctionAB(SNES,Vec,Vec,Vec,void*);
PetscErrorCode BoundaryGlobalIndex(DM,const char*,IS*);

typedef struct {
  IS    bdis; /* global indices for boundary DoFs */
  SNES  snes;
} AppCtx;

int main(int argc,char **argv)
{
  DM             dm;
  MPI_Comm       comm;
  AppCtx         user;
  EPS            eps;  /* eigenproblem solver context */
  ST             st;
  EPSType        type;
  Mat            A,B,P;
  Vec            v0;
  PetscContainer container;
  PetscInt       nev,nconv,m,n,M,N;
  PetscBool      nonlin,flg=PETSC_FALSE,update;
  SNES           snes;
  PetscReal      tol,relerr;
  PetscBool      use_shell_matrix=PETSC_FALSE,test_init_sol=PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  /* Create a quadrilateral mesh on domain (0,1)x(0,1) */
  CHKERRQ(CreateSquareMesh(comm,&dm));
  /* Setup basis function */
  CHKERRQ(SetupDiscretization(dm));
  CHKERRQ(BoundaryGlobalIndex(dm,"marker",&user.bdis));
  /* Check if we are going to use shell matrices */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-use_shell_matrix",&use_shell_matrix,NULL));
  if (use_shell_matrix) {
    CHKERRQ(DMCreateMatrix(dm,&P));
    CHKERRQ(MatGetLocalSize(P,&m,&n));
    CHKERRQ(MatGetSize(P,&M,&N));
    CHKERRQ(MatCreateShell(comm,m,n,M,N,&user,&A));
    CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_A));
    CHKERRQ(MatCreateShell(comm,m,n,M,N,&user,&B));
    CHKERRQ(MatShellSetOperation(B,MATOP_MULT,(void(*)(void))MatMult_B));
  } else {
    CHKERRQ(DMCreateMatrix(dm,&A));
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  /*
     Compose callback functions and context that will be needed by the solver
  */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"formFunction",FormFunctionA));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-form_function_ab",&flg,NULL));
  if (flg) CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"formFunctionAB",FormFunctionAB));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"formJacobian",FormJacobianA));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"formFunction",FormFunctionB));
  CHKERRQ(PetscContainerCreate(comm,&container));
  CHKERRQ(PetscContainerSetPointer(container,&user));
  CHKERRQ(PetscObjectCompose((PetscObject)A,"formFunctionCtx",(PetscObject)container));
  CHKERRQ(PetscObjectCompose((PetscObject)A,"formJacobianCtx",(PetscObject)container));
  CHKERRQ(PetscObjectCompose((PetscObject)B,"formFunctionCtx",(PetscObject)container));
  CHKERRQ(PetscContainerDestroy(&container));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(comm,&eps));
  CHKERRQ(EPSSetOperators(eps,A,B));
  CHKERRQ(EPSSetProblemType(eps,EPS_GNHEP));
  /*
     Use nonlinear inverse iteration
  */
  CHKERRQ(EPSSetType(eps,EPSPOWER));
  CHKERRQ(EPSPowerSetNonlinear(eps,PETSC_TRUE));
  /*
    Attach DM to SNES
  */
  CHKERRQ(EPSPowerGetSNES(eps,&snes));
  user.snes = snes;
  CHKERRQ(SNESSetDM(snes,dm));
  CHKERRQ(EPSSetFromOptions(eps));

  /* Set a preconditioning matrix to ST */
  if (use_shell_matrix) {
    CHKERRQ(EPSGetST(eps,&st));
    CHKERRQ(STSetPreconditionerMat(st,P));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  CHKERRQ(EPSGetConverged(eps,&nconv));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_init_sol",&test_init_sol,NULL));
  if (nconv && test_init_sol) {
    PetscScalar   k;
    PetscReal     norm0;
    PetscInt      nits;

    CHKERRQ(MatCreateVecs(A,&v0,NULL));
    CHKERRQ(EPSGetEigenpair(eps,0,&k,NULL,v0,NULL));
    CHKERRQ(EPSSetInitialSpace(eps,1,&v0));
    CHKERRQ(VecDestroy(&v0));
    /* Norm of the previous residual */
    CHKERRQ(SNESGetFunctionNorm(snes,&norm0));
    /* Make the tolerance smaller than the last residual
       SNES will converge right away if the initial is setup correctly */
    CHKERRQ(SNESSetTolerances(snes,norm0*1.2,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
    CHKERRQ(EPSSolve(eps));
    /* Number of Newton iterations supposes to be zero */
    CHKERRQ(SNESGetIterationNumber(snes,&nits));
    if (nits) CHKERRQ(PetscPrintf(comm," Number of Newton iterations %" PetscInt_FMT " should be zero \n",nits));
  }

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(EPSGetTolerances(eps,&tol,NULL));
  CHKERRQ(EPSPowerGetNonlinear(eps,&nonlin));
  CHKERRQ(EPSPowerGetUpdate(eps,&update));
  CHKERRQ(PetscPrintf(comm," Solution method: %s%s\n\n",type,nonlin?(update?" (nonlinear with monolithic update)":" (nonlinear)"):""));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(comm," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* print eigenvalue and error */
  CHKERRQ(EPSGetConverged(eps,&nconv));
  if (nconv>0) {
    PetscScalar   k;
    PetscReal     na,nb;
    Vec           a,b,eigen;
    CHKERRQ(DMCreateGlobalVector(dm,&a));
    CHKERRQ(VecDuplicate(a,&b));
    CHKERRQ(VecDuplicate(a,&eigen));
    CHKERRQ(EPSGetEigenpair(eps,0,&k,NULL,eigen,NULL));
    CHKERRQ(FormFunctionA(snes,eigen,a,&user));
    CHKERRQ(FormFunctionB(snes,eigen,b,&user));
    CHKERRQ(VecAXPY(a,-k,b));
    CHKERRQ(VecNorm(a,NORM_2,&na));
    CHKERRQ(VecNorm(b,NORM_2,&nb));
    relerr = na/(nb*PetscAbsScalar(k));
    if (relerr<10*tol) CHKERRQ(PetscPrintf(comm,"k: %g, relative error below tol\n",(double)PetscRealPart(k)));
    else CHKERRQ(PetscPrintf(comm,"k: %g, relative error: %g\n",(double)PetscRealPart(k),(double)relerr));
    CHKERRQ(VecDestroy(&a));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecDestroy(&eigen));
  } else CHKERRQ(PetscPrintf(comm,"Solver did not converge\n"));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  if (use_shell_matrix) CHKERRQ(MatDestroy(&P));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(ISDestroy(&user.bdis));
  ierr = SlepcFinalize();
  return ierr;
}

/* <|u|u, v> */
static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar cof = PetscAbsScalar(u[0]);

  f0[0] = cof*u[0];
}

/* <|\nabla u| \nabla u, \nabla v> */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt    d;
  PetscScalar cof = 0;
  for (d = 0; d < dim; ++d)  cof += u_x[d]*u_x[d];

  cof = PetscSqrtScalar(cof);

  for (d = 0; d < dim; ++d) f1[d] = u_x[d]*cof;
}

/* approximate  Jacobian for   <|u|u, v> */
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0*PetscAbsScalar(u[0]);
}

/* approximate  Jacobian for   <|\nabla u| \nabla u, \nabla v> */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;

  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

PetscErrorCode SetupDiscretization(DM dm)
{
  PetscFE        fe;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  /* Create finite element */
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRQ(PetscFECreateDefault(comm,2,1,PETSC_FALSE,NULL,-1,&fe));
  CHKERRQ(PetscObjectSetName((PetscObject)fe,"u"));
  CHKERRQ(DMSetField(dm,0,NULL,(PetscObject)fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSquareMesh(MPI_Comm comm,DM *dm)
{
  PetscInt       cells[] = {8,8};
  PetscInt       dim = 2;
  DM             pdm;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRQ(DMPlexCreateBoxMesh(comm,dim,PETSC_FALSE,cells,NULL,NULL,NULL,PETSC_TRUE,dm));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMSetUp(*dm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    CHKERRQ(DMPlexDistribute(*dm,0,NULL,&pdm));
    CHKERRQ(DMDestroy(dm));
    *dm = pdm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode BoundaryGlobalIndex(DM dm,const char labelname[],IS *bdis)
{
  IS             bdpoints;
  PetscInt       nindices,*indices,numDof,offset,npoints,i,j;
  const PetscInt *bdpoints_indices;
  DMLabel        bdmarker;
  PetscSection   gsection;

  PetscFunctionBegin;
  CHKERRQ(DMGetGlobalSection(dm,&gsection));
  CHKERRQ(DMGetLabel(dm,labelname,&bdmarker));
  CHKERRQ(DMLabelGetStratumIS(bdmarker,1,&bdpoints));
  CHKERRQ(ISGetLocalSize(bdpoints,&npoints));
  CHKERRQ(ISGetIndices(bdpoints,&bdpoints_indices));
  nindices = 0;
  for (i=0;i<npoints;i++) {
    CHKERRQ(PetscSectionGetDof(gsection,bdpoints_indices[i],&numDof));
    if (numDof<=0) continue;
    nindices += numDof;
  }
  CHKERRQ(PetscCalloc1(nindices,&indices));
  nindices = 0;
  for (i=0;i<npoints;i++) {
    CHKERRQ(PetscSectionGetDof(gsection,bdpoints_indices[i],&numDof));
    if (numDof<=0) continue;
    CHKERRQ(PetscSectionGetOffset(gsection,bdpoints_indices[i],&offset));
    for (j=0;j<numDof;j++) indices[nindices++] = offset+j;
  }
  CHKERRQ(ISRestoreIndices(bdpoints,&bdpoints_indices));
  CHKERRQ(ISDestroy(&bdpoints));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm),nindices,indices,PETSC_OWN_POINTER,bdis));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormJacobian(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  Vec            Xloc;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(VecZeroEntries(Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKMEMQ;
  CHKERRQ(DMPlexSNESComputeJacobianFEM(dm,Xloc,A,B,ctx));
  if (A!=B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  CHKMEMQ;
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  PetscDS        prob;
  PetscWeakForm  wf;
  AppCtx         *userctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDS(dm,&prob));
  CHKERRQ(PetscDSGetWeakForm(prob, &wf));
  CHKERRQ(PetscWeakFormClearIndex(wf, NULL, 0, 0, 0, PETSC_WF_G3, 0));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, NULL, 0, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu));
  CHKERRQ(FormJacobian(snes,X,A,B,ctx));
  CHKERRQ(MatZeroRowsIS(B,userctx->bdis,1.0,NULL,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianB(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  PetscDS        prob;
  PetscWeakForm  wf;
  AppCtx         *userctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDS(dm,&prob));
  CHKERRQ(PetscDSGetWeakForm(prob, &wf));
  CHKERRQ(PetscWeakFormClearIndex(wf, NULL, 0, 0, 0, PETSC_WF_G3, 0));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, NULL, 0, 0, 0, 0, 0, g0_uu, 0, NULL, 0, NULL, 0, NULL));
  CHKERRQ(FormJacobian(snes,X,A,B,ctx));
  CHKERRQ(MatZeroRowsIS(B,userctx->bdis,0.0,NULL,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionAB(SNES snes,Vec x,Vec Ax,Vec Bx,void *ctx)
{
  PetscFunctionBegin;
  /*
   * In real applications, users should have a generic formFunctionAB which
   * forms Ax and Bx simultaneously for an more efficient calculation.
   * In this example, we just call FormFunctionA+FormFunctionB to mimic how
   * to use FormFunctionAB
   */
  CHKERRQ(FormFunctionA(snes,x,Ax,ctx));
  CHKERRQ(FormFunctionB(snes,x,Bx,ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  Vec            Xloc,Floc;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGetLocalVector(dm,&Floc));
  CHKERRQ(VecZeroEntries(Xloc));
  CHKERRQ(VecZeroEntries(Floc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKMEMQ;
  CHKERRQ(DMPlexSNESComputeResidualFEM(dm,Xloc,Floc,ctx));
  CHKMEMQ;
  CHKERRQ(VecZeroEntries(F));
  CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionA(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  PetscDS        prob;
  PetscWeakForm  wf;
  PetscInt       nindices,iStart,iEnd,i;
  AppCtx         *userctx = (AppCtx *)ctx;
  PetscScalar    *array,value;
  const PetscInt *indices;
  PetscInt       vecstate;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDS(dm,&prob));
  /* hook functions */
  CHKERRQ(PetscDSGetWeakForm(prob, &wf));
  CHKERRQ(PetscWeakFormClearIndex(wf, NULL, 0, 0, 0, PETSC_WF_F0, 0));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, NULL, 0, f1_u));
  CHKERRQ(FormFunction(snes,X,F,ctx));
  /* Boundary condition */
  CHKERRQ(VecLockGet(X,&vecstate));
  if (vecstate>0) CHKERRQ(VecLockReadPop(X));
  CHKERRQ(VecGetOwnershipRange(X,&iStart,&iEnd));
  CHKERRQ(VecGetArray(X,&array));
  CHKERRQ(ISGetLocalSize(userctx->bdis,&nindices));
  CHKERRQ(ISGetIndices(userctx->bdis,&indices));
  for (i=0;i<nindices;i++) {
    value = array[indices[i]-iStart] - 0.0;
    CHKERRQ(VecSetValue(F,indices[i],value,INSERT_VALUES));
  }
  CHKERRQ(ISRestoreIndices(userctx->bdis,&indices));
  CHKERRQ(VecRestoreArray(X,&array));
  if (vecstate>0) CHKERRQ(VecLockReadPush(X));
  CHKERRQ(VecAssemblyBegin(F));
  CHKERRQ(VecAssemblyEnd(F));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_A(Mat A,Vec x,Vec y)
{
  AppCtx         *userctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&userctx));
  CHKERRQ(FormFunctionA(userctx->snes,x,y,userctx));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionB(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  PetscDS        prob;
  PetscWeakForm  wf;
  PetscInt       nindices,iStart,iEnd,i;
  AppCtx         *userctx = (AppCtx *)ctx;
  PetscScalar    value;
  const PetscInt *indices;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDS(dm,&prob));
  /* hook functions */
  CHKERRQ(PetscDSGetWeakForm(prob, &wf));
  CHKERRQ(PetscWeakFormClearIndex(wf, NULL, 0, 0, 0, PETSC_WF_F1, 0));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, NULL, 0, 0, 0, 0, f0_u, 0, NULL));
  CHKERRQ(FormFunction(snes,X,F,ctx));
  /* Boundary condition */
  CHKERRQ(VecGetOwnershipRange(F,&iStart,&iEnd));
  CHKERRQ(ISGetLocalSize(userctx->bdis,&nindices));
  CHKERRQ(ISGetIndices(userctx->bdis,&indices));
  for (i=0;i<nindices;i++) {
    value = 0.0;
    CHKERRQ(VecSetValue(F,indices[i],value,INSERT_VALUES));
  }
  CHKERRQ(ISRestoreIndices(userctx->bdis,&indices));
  CHKERRQ(VecAssemblyBegin(F));
  CHKERRQ(VecAssemblyEnd(F));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_B(Mat B,Vec x,Vec y)
{
  AppCtx         *userctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(B,&userctx));
  CHKERRQ(FormFunctionB(userctx->snes,x,y,userctx));
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      requires: double
      args: -petscspace_degree 1 -petscspace_poly_tensor
      output_file: output/ex34_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -eps_power_update -form_function_ab {{0 1}}
         filter: sed -e "s/ with monolithic update//"
      test:
         suffix: 3
         args: -use_shell_matrix -eps_power_snes_mf_operator 1
      test:
         suffix: 4
         args: -use_shell_matrix -eps_power_update -init_eps_power_snes_mf_operator 1 -eps_power_snes_mf_operator 1 -form_function_ab {{0 1}}
         filter: sed -e "s/ with monolithic update//"
      test:
         suffix: 5
         args: -use_shell_matrix -eps_power_update -init_eps_power_snes_mf_operator 1 -eps_power_snes_mf_operator 1 -form_function_ab {{0 1}} -test_init_sol 1
         filter: sed -e "s/ with monolithic update//"

      test:
         suffix: 6
         args: -use_shell_matrix -eps_power_update -init_eps_power_snes_mf_operator 1 -eps_power_snes_mf_operator 1 -form_function_ab {{0 1}} -eps_monitor_all
         output_file: output/ex34_6.out
         filter: sed -e "s/\([+-].*i\)//g" -e "1,3s/[0-9]//g" -e "/[45] EPS/d"
TEST*/
