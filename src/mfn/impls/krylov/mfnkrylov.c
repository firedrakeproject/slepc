/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc matrix function solver: "krylov"

   Method: Arnoldi with Eiermann-Ernst restart

   Algorithm:

       Build Arnoldi approximations using f(H) for the Hessenberg matrix H,
       restart by discarding the Krylov basis but keeping H.

   References:

       [1] M. Eiermann and O. Ernst, "A restarted Krylov subspace method
           for the evaluation of matrix functions", SIAM J. Numer. Anal.
           44(6):2481-2504, 2006.
*/

#include <slepc/private/mfnimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode MFNSetUp_Krylov(MFN mfn)
{
  PetscInt       N;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(mfn->A,&N,NULL));
  if (mfn->ncv==PETSC_DEFAULT) mfn->ncv = PetscMin(30,N);
  if (mfn->max_it==PETSC_DEFAULT) mfn->max_it = 100;
  CHKERRQ(MFNAllocateSolution(mfn,1));
  PetscFunctionReturn(0);
}

PetscErrorCode MFNSolve_Krylov(MFN mfn,Vec b,Vec x)
{
  PetscInt          n=0,m,ld,ldh,j;
  PetscBLASInt      m_,inc=1;
  Mat               M,G=NULL,H=NULL;
  Vec               F=NULL;
  PetscScalar       *marray,*farray,*harray;
  const PetscScalar *garray;
  PetscReal         beta,betaold=0.0,nrm=1.0;
  PetscBool         breakdown;

  PetscFunctionBegin;
  m  = mfn->ncv;
  ld = m+1;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,ld,m,NULL,&M));
  CHKERRQ(MatDenseGetArray(M,&marray));

  /* set initial vector to b/||b|| */
  CHKERRQ(BVInsertVec(mfn->V,0,b));
  CHKERRQ(BVScaleColumn(mfn->V,0,1.0/mfn->bnorm));
  CHKERRQ(VecSet(x,0.0));

  /* Restart loop */
  while (mfn->reason == MFN_CONVERGED_ITERATING) {
    mfn->its++;

    /* compute Arnoldi factorization */
    CHKERRQ(BVMatArnoldi(mfn->V,mfn->transpose_solve?mfn->AT:mfn->A,M,0,&m,&beta,&breakdown));

    /* save previous Hessenberg matrix in G; allocate new storage for H and f(H) */
    if (mfn->its>1) { G = H; H = NULL; }
    ldh = n+m;
    CHKERRQ(MFN_CreateVec(ldh,&F));
    CHKERRQ(MFN_CreateDenseMat(ldh,&H));

    /* glue together the previous H and the new H obtained with Arnoldi */
    CHKERRQ(MatDenseGetArray(H,&harray));
    for (j=0;j<m;j++) {
      CHKERRQ(PetscArraycpy(harray+n+(j+n)*ldh,marray+j*ld,m));
    }
    if (mfn->its>1) {
      CHKERRQ(MatDenseGetArrayRead(G,&garray));
      for (j=0;j<n;j++) {
        CHKERRQ(PetscArraycpy(harray+j*ldh,garray+j*n,n));
      }
      CHKERRQ(MatDenseRestoreArrayRead(G,&garray));
      CHKERRQ(MatDestroy(&G));
      harray[n+(n-1)*ldh] = betaold;
    }
    CHKERRQ(MatDenseRestoreArray(H,&harray));

    if (mfn->its==1) {
      /* set symmetry flag of H from A */
      CHKERRQ(MatPropagateSymmetryOptions(mfn->A,H));
    }

    /* evaluate f(H) */
    CHKERRQ(FNEvaluateFunctionMatVec(mfn->fn,H,F));

    /* x += ||b||*V*f(H)*e_1 */
    CHKERRQ(VecGetArray(F,&farray));
    CHKERRQ(PetscBLASIntCast(m,&m_));
    nrm = BLASnrm2_(&m_,farray+n,&inc);   /* relative norm of the update ||u||/||b|| */
    CHKERRQ(MFNMonitor(mfn,mfn->its,nrm));
    for (j=0;j<m;j++) farray[j+n] *= mfn->bnorm;
    CHKERRQ(BVSetActiveColumns(mfn->V,0,m));
    CHKERRQ(BVMultVec(mfn->V,1.0,1.0,x,farray+n));
    CHKERRQ(VecRestoreArray(F,&farray));

    /* check convergence */
    if (mfn->its >= mfn->max_it) mfn->reason = MFN_DIVERGED_ITS;
    if (mfn->its>1) {
      if (m<mfn->ncv || breakdown || beta==0.0 || nrm<mfn->tol) mfn->reason = MFN_CONVERGED_TOL;
    }

    /* restart with vector v_{m+1} */
    if (mfn->reason == MFN_CONVERGED_ITERATING) {
      CHKERRQ(BVCopyColumn(mfn->V,m,0));
      n += m;
      betaold = beta;
    }
  }

  CHKERRQ(MatDestroy(&H));
  CHKERRQ(MatDestroy(&G));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(MatDenseRestoreArray(M,&marray));
  CHKERRQ(MatDestroy(&M));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode MFNCreate_Krylov(MFN mfn)
{
  PetscFunctionBegin;
  mfn->ops->solve          = MFNSolve_Krylov;
  mfn->ops->setup          = MFNSetUp_Krylov;
  PetscFunctionReturn(0);
}
