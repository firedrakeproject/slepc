/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc matrix equation solver: "krylov"

   Method: Arnoldi with Eiermann-Ernst restart

   Algorithm:

       Project the equation onto the Arnoldi basis and solve the compressed
       equation the Hessenberg matrix H, restart by discarding the Krylov
       basis but keeping H.

   References:

       [1] Y. Saad, "Numerical solution of large Lyapunov equations", in
           Signal processing, scattering and operator theory, and numerical
           methods, vol. 5 of Progr. Systems Control Theory, pages 503-511,
           1990.

       [2] D. Kressner, "Memory-efficient Krylov subspace techniques for
           solving large-scale Lyapunov equations", in 2008 IEEE Int. Conf.
           Computer-Aided Control Systems, pages 613-618, 2008.
*/

#include <slepc/private/lmeimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode LMESetUp_Krylov(LME lme)
{
  PetscInt       N;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(lme->A,&N,NULL));
  if (lme->ncv==PETSC_DEFAULT) lme->ncv = PetscMin(30,N);
  if (lme->max_it==PETSC_DEFAULT) lme->max_it = 100;
  CHKERRQ(LMEAllocateSolution(lme,1));
  PetscFunctionReturn(0);
}

PetscErrorCode LMESolve_Krylov_Lyapunov_Vec(LME lme,Vec b,PetscBool fixed,PetscInt rrank,BV C1,BV *X1,PetscInt *col,PetscBool *fail,PetscInt *totalits)
{
  PetscInt       n=0,m,ldh,ldg=0,i,j,rank=0,lrank,pass,nouter=0,its;
  PetscReal      bnorm,beta,errest;
  PetscBool      breakdown;
  PetscScalar    *Harray,*G=NULL,*Gnew=NULL,*L,*U,*r,*Qarray,sone=1.0,zero=0.0;
  PetscBLASInt   n_,m_,rk_;
  Mat            Q,H;

  PetscFunctionBegin;
  *fail = PETSC_FALSE;
  its = 0;
  m  = lme->ncv;
  ldh = m+1;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,ldh,m,NULL,&H));
  CHKERRQ(MatDenseGetArray(H,&Harray));

  CHKERRQ(VecNorm(b,NORM_2,&bnorm));
  PetscCheck(bnorm,PetscObjectComm((PetscObject)lme),PETSC_ERR_ARG_WRONG,"Cannot process a zero vector in the right-hand side");

  for (pass=0;pass<2;pass++) {

    /* set initial vector to b/||b|| */
    CHKERRQ(BVInsertVec(lme->V,0,b));
    CHKERRQ(BVScaleColumn(lme->V,0,1.0/bnorm));

    /* Restart loop */
    while ((pass==0 && !*fail) || (pass==1 && its+1<nouter)) {
      its++;

      /* compute Arnoldi factorization */
      CHKERRQ(BVMatArnoldi(lme->V,lme->A,H,0,&m,&beta,&breakdown));
      CHKERRQ(BVSetActiveColumns(lme->V,0,m));

      if (pass==0) {
        /* glue together the previous H and the new H obtained with Arnoldi */
        ldg = n+m+1;
        CHKERRQ(PetscCalloc1(ldg*(n+m),&Gnew));
        for (j=0;j<m;j++) CHKERRQ(PetscArraycpy(Gnew+n+(j+n)*ldg,Harray+j*ldh,m));
        Gnew[n+m+(n+m-1)*ldg] = beta;
        if (G) {
          for (j=0;j<n;j++) CHKERRQ(PetscArraycpy(Gnew+j*ldg,G+j*(n+1),n+1));
          CHKERRQ(PetscFree(G));
        }
        G = Gnew;
        n += m;
      } else {
        /* update Z = Z + V(:,1:m)*Q    with   Q=U(blk,:)*P(1:nrk,:)'  */
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,m,*col+rank,NULL,&Q));
        CHKERRQ(MatDenseGetArray(Q,&Qarray));
        CHKERRQ(PetscBLASIntCast(m,&m_));
        CHKERRQ(PetscBLASIntCast(n,&n_));
        CHKERRQ(PetscBLASIntCast(rank,&rk_));
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m_,&rk_,&rk_,&sone,U+its*m,&n_,L,&n_,&zero,Qarray+(*col)*m,&m_));
        CHKERRQ(MatDenseRestoreArray(Q,&Qarray));
        CHKERRQ(BVSetActiveColumns(*X1,*col,*col+rank));
        CHKERRQ(BVMult(*X1,1.0,1.0,lme->V,Q));
        CHKERRQ(MatDestroy(&Q));
      }

      if (pass==0) {
        /* solve compressed Lyapunov equation */
        CHKERRQ(PetscCalloc1(n,&r));
        CHKERRQ(PetscCalloc1(n*n,&L));
        r[0] = bnorm;
        errest = PetscAbsScalar(G[n+(n-1)*ldg]);
        CHKERRQ(LMEDenseHessLyapunovChol(lme,n,G,ldg,1,r,n,L,n,&errest));
        CHKERRQ(LMEMonitor(lme,*totalits+its,errest));
        CHKERRQ(PetscFree(r));

        /* check convergence */
        if (errest<lme->tol) {
          lme->errest += errest;
          CHKERRQ(PetscMalloc1(n*n,&U));
          /* transpose L */
          for (j=0;j<n;j++) {
            for (i=j+1;i<n;i++) {
              L[i+j*n] = PetscConj(L[j+i*n]);
              L[j+i*n] = 0.0;
            }
          }
          CHKERRQ(LMEDenseRankSVD(lme,n,L,n,U,n,&lrank));
          CHKERRQ(PetscInfo(lme,"Rank of the Cholesky factor = %" PetscInt_FMT "\n",lrank));
          nouter = its;
          its = -1;
          if (!fixed) {  /* X1 was not set by user, allocate it with rank columns */
            rank = lrank;
            if (*col) CHKERRQ(BVResize(*X1,*col+rank,PETSC_TRUE));
            else CHKERRQ(BVDuplicateResize(C1,rank,X1));
          } else rank = PetscMin(lrank,rrank);
          CHKERRQ(PetscFree(G));
          break;
        } else {
          CHKERRQ(PetscFree(L));
          if (*totalits+its>=lme->max_it) *fail = PETSC_TRUE;
        }
      }

      /* restart with vector v_{m+1} */
      if (!*fail) CHKERRQ(BVCopyColumn(lme->V,m,0));
    }
  }

  *col += rank;
  *totalits += its+1;
  CHKERRQ(MatDenseRestoreArray(H,&Harray));
  CHKERRQ(MatDestroy(&H));
  if (L) CHKERRQ(PetscFree(L));
  if (U) CHKERRQ(PetscFree(U));
  PetscFunctionReturn(0);
}

PetscErrorCode LMESolve_Krylov_Lyapunov(LME lme)
{
  PetscBool      fail,fixed = lme->X? PETSC_TRUE: PETSC_FALSE;
  PetscInt       i,k,rank=0,col=0;
  Vec            b;
  BV             X1=NULL,C1;
  Mat            X1m,X1t,C1m;

  PetscFunctionBegin;
  CHKERRQ(MatLRCGetMats(lme->C,NULL,&C1m,NULL,NULL));
  CHKERRQ(BVCreateFromMat(C1m,&C1));
  CHKERRQ(BVSetFromOptions(C1));
  CHKERRQ(BVGetActiveColumns(C1,NULL,&k));
  if (fixed) {
    CHKERRQ(MatLRCGetMats(lme->X,NULL,&X1m,NULL,NULL));
    CHKERRQ(BVCreateFromMat(X1m,&X1));
    CHKERRQ(BVSetFromOptions(X1));
    CHKERRQ(BVGetActiveColumns(X1,NULL,&rank));
    rank = rank/k;
  }
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(C1,i,&b));
    CHKERRQ(LMESolve_Krylov_Lyapunov_Vec(lme,b,fixed,rank,C1,&X1,&col,&fail,&lme->its));
    CHKERRQ(BVRestoreColumn(C1,i,&b));
    if (fail) {
      lme->reason = LME_DIVERGED_ITS;
      break;
    }
  }
  if (lme->reason==LME_CONVERGED_ITERATING) lme->reason = LME_CONVERGED_TOL;
  CHKERRQ(BVCreateMat(X1,&X1t));
  if (fixed) CHKERRQ(MatCopy(X1t,X1m,SAME_NONZERO_PATTERN));
  else CHKERRQ(MatCreateLRC(NULL,X1t,NULL,NULL,&lme->X));
  CHKERRQ(MatDestroy(&X1t));
  CHKERRQ(BVDestroy(&C1));
  CHKERRQ(BVDestroy(&X1));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode LMECreate_Krylov(LME lme)
{
  PetscFunctionBegin;
  lme->ops->solve[LME_LYAPUNOV]      = LMESolve_Krylov_Lyapunov;
  lme->ops->setup                    = LMESetUp_Krylov;
  PetscFunctionReturn(0);
}
