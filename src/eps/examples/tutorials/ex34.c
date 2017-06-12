/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

/*
   This is a nonlinear eigenvalue problem. When p=2, it is reduced to a linear Laplace eigenvalue
   problem.

  -\nabla\cdot(|\nabla u|^{p-2} \nabla u) = k |u|^{p-2} u in (0,1)x(0,1),

   u = 0 on the entire boundary.

   The code is implemented based on DMPlex using Q1 FEM on a quadrilateral mesh. In this code, we consider p=3.
*/

/*
  Contributed  by Fande Kong fdkong.jd@gmail.com
*/

static char help[] = "Nonlinear eigenvalue problems.\n\n";


#include <slepceps.h>
#include <petscdmplex.h>
#include <petscds.h>

PetscErrorCode CreateSqureMesh(MPI_Comm,DM*);
PetscErrorCode SetupDiscretization(DM);
PetscErrorCode FormJacobianA(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionA(SNES,Vec,Vec,void*);
PetscErrorCode FormJacobianB(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionB(SNES,Vec,Vec,void*);
PetscErrorCode BoundaryGlobalIndex(DM,const char*,IS*);

typedef struct {
IS    bdis; /* global indices for boundary DoFs */
} AppCtx;

int main(int argc,char **argv)
{
  DM             dm;
  MPI_Comm       comm;
  AppCtx         user;
  EPS            eps;  /* eigenproblem solver context */
  ST             st;   /* spectral transformation context */
  EPSType        type;
  Mat            A,B;
  PetscContainer container;
  PetscInt       nev;
  PetscBool      nonlin;
  SNES           snes;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  /* Create a quadrilateral mesh on domain (0,1)x(0,1) */
  ierr = CreateSqureMesh(comm,&dm);CHKERRQ(ierr);
  /* Setup basis function */
  ierr = SetupDiscretization(dm);CHKERRQ(ierr);
  ierr = BoundaryGlobalIndex(dm,"marker",&user.bdis);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);

  /*
     Compose callback functions and context that will be needed by the solver
  */
  ierr = PetscObjectComposeFunction((PetscObject)A,"formFunction",FormFunctionA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"formJacobian",FormJacobianA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"formFunction",FormFunctionB);CHKERRQ(ierr);
  ierr = PetscContainerCreate(comm,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,&user);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"formFunctionCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"formJacobianCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B,"formFunctionCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(comm,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
  /*
     Use nonlinear inverse iteration
  */
  ierr = EPSSetType(eps,EPSPOWER);CHKERRQ(ierr);
  ierr = EPSPowerSetNonlinear(eps,PETSC_TRUE);CHKERRQ(ierr);

  /*
     Nonlinear inverse iteration requires shift-and-invert with target=0
  */
  ierr = EPSSetTarget(eps,0.0);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);

  /*
    Attach DM to SNES
  */
  ierr = EPSPowerGetSNES(eps,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = EPSPowerGetNonlinear(eps,&nonlin);CHKERRQ(ierr);
  ierr = PetscPrintf(comm," Solution method: %s%s\n\n",type,nonlin?" (nonlinear)":"");CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* print eigenvalue and error */
  {
    PetscScalar   k;
    PetscReal     na,nb;
    Vec           a,b,eigen;
    ierr = DMCreateGlobalVector(dm,&a);CHKERRQ(ierr);
    ierr = VecDuplicate(a,&b);CHKERRQ(ierr);
    ierr = VecDuplicate(a,&eigen);CHKERRQ(ierr);
    ierr = EPSGetEigenpair(eps,0,&k,NULL,eigen,NULL);CHKERRQ(ierr);
    ierr = FormFunctionA(snes,eigen,a,&user);CHKERRQ(ierr);
    ierr = FormFunctionB(snes,eigen,b,&user);CHKERRQ(ierr);
    ierr = VecAXPY(a,-k,b);CHKERRQ(ierr);
    ierr = VecNorm(a,NORM_2,&na);CHKERRQ(ierr);
    ierr = VecNorm(b,NORM_2,&nb);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"k: %g error: %g\n",k,na/nb);CHKERRQ(ierr);
    ierr = VecDestroy(&a);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&eigen);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = ISDestroy(&user.bdis);CHKERRQ(ierr);
  ierr = SlepcFinalize();
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
  PetscDS        prob;
  PetscInt       totDim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm,2,1,PETSC_FALSE,NULL,-1,&fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "u");CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob,0,(PetscObject) fe);CHKERRQ(ierr);
  ierr = DMSetDS(dm,prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob,&totDim);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode CreateSqureMesh(MPI_Comm comm,DM *dm)
{
  PetscInt         cells[] = {8,8};
  PetscInt         dim = 2;
  DM               pdm;
  PetscMPIInt      size;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = DMPlexCreateHexBoxMesh(comm,dim,cells,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  if (size == 1) {
    PetscFunctionReturn(0);
  }
  ierr = DMPlexDistribute(*dm,0,NULL,&pdm);CHKERRQ(ierr);
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  *dm = pdm;
  PetscFunctionReturn(0);
}


PetscErrorCode BoundaryGlobalIndex(DM dm,const char labelname[],IS *bdis)
{
  IS              bdpoints;
  PetscInt        nindices,*indices,numDof,offset,npoints,i,j;
  const PetscInt  *bdpoints_indices;
  DMLabel         bdmarker;
  PetscSection    gsection;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDefaultGlobalSection(dm,&gsection);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,labelname,&bdmarker);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(bdmarker,1,&bdpoints);CHKERRQ(ierr);
  ierr = ISGetLocalSize(bdpoints,&npoints);CHKERRQ(ierr);
  ierr = ISGetIndices(bdpoints,&bdpoints_indices);CHKERRQ(ierr);
  nindices = 0;
  for (i=0;i<npoints;i++) {
    ierr = PetscSectionGetDof(gsection,bdpoints_indices[i],&numDof);CHKERRQ(ierr);
    if(numDof<=0) continue;
    nindices += numDof;
  }
  ierr = PetscCalloc1(nindices,&indices);CHKERRQ(ierr);
  nindices = 0;
  for (i=0;i<npoints;i++) {
    ierr = PetscSectionGetDof(gsection,bdpoints_indices[i],&numDof);CHKERRQ(ierr);
    if(numDof<=0) continue;
    ierr = PetscSectionGetOffset(gsection,bdpoints_indices[i],&offset);CHKERRQ(ierr);
    for (j=0;j<numDof;j++)
      indices[nindices++] = offset+j;
  }
  ierr = ISRestoreIndices(bdpoints,&bdpoints_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&bdpoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),nindices,indices,PETSC_OWN_POINTER,bdis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode FormJacobian(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  Vec            Xloc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMPlexSNESComputeJacobianFEM(dm,Xloc,A,B,ctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscDS        prob;
  AppCtx         *userctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  ierr = MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,NULL,NULL,NULL,g3_uu);CHKERRQ(ierr);
  ierr = FormJacobian(snes,X,A,B,ctx);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(B,userctx->bdis,1.0,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianB(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscDS        prob;
  AppCtx         *userctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  ierr = MatSetOption(B,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,g0_uu,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = FormJacobian(snes,X,A,B,ctx);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(B,userctx->bdis,0.0,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  Vec            Xloc,Floc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMPlexSNESComputeResidualFEM(dm,Xloc,Floc,ctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode FormFunctionA(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscDS        prob;
  PetscInt       nindices,iStart,iEnd,i;
  AppCtx         *userctx = (AppCtx *)ctx;
  PetscScalar    *array,value;
  const PetscInt *indices;
  PetscInt       vecstate;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  /* hook functions */
  ierr = PetscDSSetResidual(prob,0,NULL,f1_u);CHKERRQ(ierr);
  ierr = FormFunction(snes,X,F,ctx);CHKERRQ(ierr);
  /* Boundary condition */
  ierr = VecLockGet(X,&vecstate);CHKERRQ(ierr);
  if (vecstate>0) {
    ierr = VecLockPop(X);CHKERRQ(ierr);
  }
  ierr = VecGetOwnershipRange(X,&iStart,&iEnd);CHKERRQ(ierr);
  ierr = VecGetArray(X,&array);CHKERRQ(ierr);
  ierr = ISGetLocalSize(userctx->bdis,&nindices);CHKERRQ(ierr);
  ierr = ISGetIndices(userctx->bdis,&indices);CHKERRQ(ierr);
  for (i=0;i<nindices;i++) {
    value = array[indices[i]-iStart] - 0;
    ierr = VecSetValue(F,indices[i],value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(userctx->bdis,&indices);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&array);CHKERRQ(ierr);
  if (vecstate>0) {
    ierr = VecLockPush(X);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FormFunctionB(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscDS        prob;
  PetscInt       nindices,iStart,iEnd,i;
  AppCtx         *userctx = (AppCtx *)ctx;
  PetscScalar    value;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  /* hook functions */
  ierr = PetscDSSetResidual(prob,0,f0_u,NULL);CHKERRQ(ierr);
  ierr = FormFunction(snes,X,F,ctx);CHKERRQ(ierr);
  /* Boundary condition */
  ierr = VecGetOwnershipRange(F,&iStart,&iEnd);CHKERRQ(ierr);
  ierr = ISGetLocalSize(userctx->bdis,&nindices);CHKERRQ(ierr);
  ierr = ISGetIndices(userctx->bdis,&indices);CHKERRQ(ierr);
  for (i=0;i<nindices;i++) {
    value = 0;
    ierr = VecSetValue(F,indices[i],value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(userctx->bdis,&indices);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
