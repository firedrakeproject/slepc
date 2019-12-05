/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "lyapii"

   Method: Lyapunov inverse iteration

   Algorithm:

       Lyapunov inverse iteration using LME solvers

   References:

       [1] H.C. Elman and M. Wu, "Lyapunov inverse iteration for computing a
           few rightmost eigenvalues of large generalized eigenvalue problems",
           SIAM J. Matrix Anal. Appl. 34(4):1685-1707, 2013.
*/

#include <slepc/private/epsimpl.h>          /*I "slepceps.h" I*/
#include <slepc/private/lmeimpl.h>          /*I "slepclme.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  LME lme;
  DS  ds;
} EPS_LYAPII;

PetscErrorCode EPSSetUp_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  SlepcSC        sc;
  PetscBool      istrivial,issinv;

  PetscFunctionBegin;
//  if (eps->ncv) {
//    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
//  } else eps->ncv = eps->nev;
//  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(1000*eps->nev,100*eps->n);
  if (!eps->which) eps->which=EPS_LARGEST_REAL;
  if (eps->which!=EPS_LARGEST_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Balancing not supported in this solver");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSINVERT,&issinv);CHKERRQ(ierr);
  if (!issinv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Must use STSINVERT spectral transformation");

  if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
  ierr = LMESetProblemType(ctx->lme,LME_LYAPUNOV);CHKERRQ(ierr);
  ierr = LMESetErrorIfNotConverged(ctx->lme,PETSC_TRUE);CHKERRQ(ierr);

  ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,3);CHKERRQ(ierr);
  ierr = EPSGetDS(eps,&eps->ds);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->ncv*eps->ncv);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(eps->ds,&sc);CHKERRQ(ierr);
  sc->comparison = SlepcCompareSmallestMagnitude;
  PetscFunctionReturn(0);
}

static PetscErrorCode EV2x2(PetscScalar *M,PetscInt ld,PetscScalar *wr,PetscScalar *wi,PetscScalar *vec)
{
#if defined(PETSC_MISSING_LAPACK_GEEV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEEV/GESVD - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscScalar    work[10];
  PetscBLASInt   two=2,info,lwork=10,ld_;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      rwork[4];
#endif
#if defined(PETSC_HAVE_ESSL)
  PetscInt       i;
  PetscScalar    sdummy;
  PetscBLASInt   idummy,io=1;
  PetscScalar    wri[4];
#endif

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(ld,&ld_);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_ESSL)
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&two,M,&ld_,wr,wi,NULL,&ld_,vec,&ld_,work,&lwork,&info));
    SlepcCheckLapackInfo("geev",info);
#else
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_("N","V",&two,M,&ld_,wr,NULL,&ld_,vec,&ld_,work,&lwork,rwork,&info));
    SlepcCheckLapackInfo("geev",info);
#endif
#else
    PetscStackCallBLAS("LAPACKgeev",LAPACKgeev_(&io,M,&ld_,wri,vec,ld_,&idummy,&ld_,work,&lwork));
#if !defined(PETSC_USE_COMPLEX)
    for (i=0;i<2;i++) {
      wr[i] = wri[2*i];
      wi[i] = wri[2*i+1];
    }
#else
    for (i=0;i<2;i++) wr[i] = wri[i];
#endif
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscInt       off,f,c,i,j,ldds,ldS,rk,nloc,nv,idx;
  Vec            v,w,u;
  Mat            S,C,Ux[2],Y,Y1,R,U,W,X;
  PetscScalar    theta,*eigr,*eigi,*array,er,ei,*uu,*pV,*s,*si,*xx,*aa,*bb,*pS,pM[4],vec[4];
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      norm;
#endif

  PetscFunctionBegin;
  ierr = DSCreate(PetscObjectComm((PetscObject)eps),&ctx->ds);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->ds);CHKERRQ(ierr);
  ierr = DSSetType(ctx->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSAllocate(ctx->ds,eps->ncv);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(eps->ds,&ldds);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(ctx->ds,&ldS);CHKERRQ(ierr);

  /* Operator for the Lyapunov equation */
  ierr = STGetOperator(eps->st,&S);CHKERRQ(ierr);
  ierr = LMESetCoefficients(ctx->lme,S,NULL,NULL,NULL);CHKERRQ(ierr);

  /* Right-hand side */
  ierr = MatCreateDense(PetscObjectComm((PetscObject)eps),PETSC_DECIDE,PETSC_DECIDE,eps->n,1,NULL,&Ux[0]);CHKERRQ(ierr);
  ierr = MatCreateDense(PetscObjectComm((PetscObject)eps),PETSC_DECIDE,PETSC_DECIDE,eps->n,2,NULL,&Ux[1]);CHKERRQ(ierr);
  ierr = BVGetSizes(eps->V,&nloc,NULL,&nv);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
  ierr = PetscMalloc4(nv,&s,nv,&si,nv*nv,&eigr,nv*nv,&eigi);CHKERRQ(ierr);

  /* Initialize first column */
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  ierr = BVGetColumn(eps->V,0,&v);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&u);CHKERRQ(ierr);
  ierr = MatMult(S,v,u);CHKERRQ(ierr);
  ierr = BVRestoreColumn(eps->V,0,&v);CHKERRQ(ierr);
  ierr = VecScale(u,PetscSqrtReal(2.0));
  ierr = VecGetArray(u,&uu);CHKERRQ(ierr);
  ierr = MatDenseGetColumn(Ux[0],0,&array);CHKERRQ(ierr);
  ierr = PetscMemcpy(array,uu,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&uu);CHKERRQ(ierr);
  ierr = MatDenseRestoreColumn(Ux[0],&array);CHKERRQ(ierr);
  idx = 0;
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Matrix for placing the solution (from the BV context) */
    ierr = BVGetArray(eps->V,&pV);CHKERRQ(ierr);
    ierr = PetscMemzero(pV,nv*eps->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatCreateDense(PetscObjectComm((PetscObject)eps),PETSC_DECIDE,PETSC_DECIDE,eps->n,nv,pV,&Y1);CHKERRQ(ierr);
    ierr = MatCreateLRC(NULL,Y1,NULL,NULL,&Y);CHKERRQ(ierr);
    ierr = MatDestroy(&Y1);CHKERRQ(ierr);
    ierr = LMESetSolution(ctx->lme,Y);CHKERRQ(ierr);
    ierr = MatDestroy(&Y);CHKERRQ(ierr);

    /* Solve the Lyapunov equation Y = lyap(S,2*S*Z*S') */
    ierr = MatCreateLRC(NULL,Ux[idx],NULL,NULL,&C);CHKERRQ(ierr);
    ierr = LMESetRHS(ctx->lme,C);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = LMESolve(ctx->lme);CHKERRQ(ierr);
    ierr = BVRestoreArray(eps->V,&pV);CHKERRQ(ierr);

    /* SVD of the solution */
    ierr = DSSetDimensions(ctx->ds,nv,nv,0,0);CHKERRQ(ierr);
    ierr = DSGetMat(ctx->ds,DS_MAT_A,&R);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,0,nv);CHKERRQ(ierr);
    ierr = BVOrthogonalize(eps->V,R);CHKERRQ(ierr);
    ierr = DSRestoreMat(ctx->ds,DS_MAT_A,&R);CHKERRQ(ierr);
    ierr = DSSetState(ctx->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(ctx->ds,s,si);CHKERRQ(ierr);

    /* Determine rank */
    rk = nv;
    for (i=1;i<nv;i++) if (PetscAbsScalar(s[i]/s[0])<PETSC_SQRT_MACHINE_EPSILON) {rk=i; break;}
    rk = PetscMin(rk,eps->mpd);
    ierr = DSGetMat(ctx->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,0,rk);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(eps->V,0,rk);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    /*Rank reduction */
    ierr = DSSetDimensions(ctx->ds,rk,rk,0,0);CHKERRQ(ierr);
    ierr = DSGetMat(ctx->ds,DS_MAT_A,&W);CHKERRQ(ierr);
    ierr = BVMatProject(eps->V,S,eps->V,W);CHKERRQ(ierr);
    ierr = MatDenseGetArray(W,&pS);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_A,&aa);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_B,&bb);CHKERRQ(ierr);
    /* A = kron(I,S)+kron(S,I) B = -2*kron(S,S) */
    ierr = PetscMemzero(aa,ldds*rk*rk*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(bb,ldds*rk*rk*sizeof(PetscScalar));CHKERRQ(ierr);
    for (f=0;f<rk;f++) {
      off = f*rk+f*rk*ldds;
      for (i=0;i<rk;i++) for (j=0;j<rk;j++) aa[off+i+j*ldds] = pS[i+j*rk];
      for (c=0;c<rk;c++) {
        off = f*rk+c*rk*ldds;
        theta = pS[f+c*rk];
        for (i=0;i<rk;i++) aa[off+i+i*ldds] += theta;
        for (i=0;i<rk;i++) for (j=0;j<rk;j++) bb[off+i+j*ldds] = -2*theta*pS[i+j*rk];
      }
    }
    ierr = DSRestoreArray(eps->ds,DS_MAT_A,&aa);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_B,&bb);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(W,&pS);CHKERRQ(ierr);
    ierr = MatDestroy(&W);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,rk*rk,rk*rk,0,0);CHKERRQ(ierr);
    ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(eps->ds,eigr,eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eigr,eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSVectors(eps->ds,DS_MAT_X,0,NULL);CHKERRQ(ierr);
    ierr = DSGetArray(eps->ds,DS_MAT_X,&xx);CHKERRQ(ierr);
    ierr = DSGetArray(ctx->ds,DS_MAT_A,&aa);CHKERRQ(ierr);
    for (i=0;i<rk;i++) {
      ierr = PetscMemcpy(aa+i*ldS,xx+i*rk,rk*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(ctx->ds,DS_MAT_A,&aa);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_X,&xx);CHKERRQ(ierr);
    ierr = DSSetDimensions(ctx->ds,rk,rk,0,0);CHKERRQ(ierr);
    ierr = DSSetState(ctx->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(ctx->ds,s,si);CHKERRQ(ierr);
    rk = 2;
    if (PetscAbsScalar(s[1]/s[0])<PETSC_SQRT_MACHINE_EPSILON) rk=1;
    ierr = DSGetMat(ctx->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,0,rk);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    /* Prepare right-hand side */
    idx = (rk==2)?1:0;
    for (i=0;i<rk;i++) {
      ierr = BVGetColumn(eps->V,i,&v);CHKERRQ(ierr);
      ierr = MatMult(S,v,u);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,i,&v);CHKERRQ(ierr);
      ierr = VecScale(u,PetscSqrtReal(2.0));
      ierr = VecGetArray(u,&uu);CHKERRQ(ierr);
      ierr = MatDenseGetColumn(Ux[idx],i,&array);CHKERRQ(ierr);
      ierr = PetscMemcpy(array,uu,nloc*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(u,&uu);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(Ux[idx],&array);CHKERRQ(ierr);
    }
    /* Eigenpair approximation */
    ierr = BVGetColumn(eps->V,0,&v);CHKERRQ(ierr);
    ierr = MatMult(S,v,u);CHKERRQ(ierr);
    ierr = VecDot(u,v,pM);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,0,&v);CHKERRQ(ierr);
    if (rk>1) {
      ierr = BVGetColumn(eps->V,1,&w);CHKERRQ(ierr);
      ierr = VecDot(u,w,pM+1);CHKERRQ(ierr);
      ierr = MatMult(S,w,u);CHKERRQ(ierr);
      ierr = VecDot(u,w,pM+3);CHKERRQ(ierr);
      ierr = BVGetColumn(eps->V,0,&v);CHKERRQ(ierr);
      ierr = VecDot(u,v,pM+2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,0,&v);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,1,&w);CHKERRQ(ierr);
      ierr = EV2x2(pM,2,eps->eigr,eps->eigi,vec);CHKERRQ(ierr);
      ierr = MatCreateDense(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,2,2,vec,&X);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(eps->V,0,rk);CHKERRQ(ierr);
      ierr = BVMultInPlace(eps->V,X,0,2);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      norm = eps->eigr[0]*eps->eigr[0]+eps->eigi[0]*eps->eigi[0];
      er = eps->eigr[0]/norm; ei = (PetscImaginaryPart(eps->eigr[0]) + eps->eigi[0])/-norm;
#else
      er =1.0/eps->eigr[0]; ei = 0.0;
#endif
    } else {eps->eigr[0] = pM[0]; er = 1.0/eps->eigr[0]; ei = 0.0;}
    ierr = BVGetColumn(eps->V,0,&v);CHKERRQ(ierr);
    if (eps->eigi[0]!=0.0) {
      ierr = BVGetColumn(eps->V,1,&w);CHKERRQ(ierr);
    } else w = NULL;
    ierr = EPSComputeResidualNorm_Private(eps,PETSC_FALSE,er,ei,v,w,eps->work,&eps->errest[0]);
    ierr = BVRestoreColumn(eps->V,0,&v);CHKERRQ(ierr);
    if (w) {
      ierr = BVRestoreColumn(eps->V,1,&w);CHKERRQ(ierr);
    }
    if (eps->errest[0]<eps->tol) {
      eps->nconv++;
      if (ei!=0.0) eps->nconv++;
    }

    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,eps->nconv+1);CHKERRQ(ierr);
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,eps->nconv,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = MatDestroy(&Y1);CHKERRQ(ierr);
  ierr = MatDestroy(&Ux[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&Ux[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = DSDestroy(&ctx->ds);CHKERRQ(ierr);
  ierr = PetscFree4(s,si,eigr,eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_LyapII(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS Lyapunov Inverse Iteration Options");CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
  ierr = LMESetFromOptions(ctx->lme);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIISetLME_LyapII(EPS eps,LME lme)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)lme);CHKERRQ(ierr);
  ierr = LMEDestroy(&ctx->lme);CHKERRQ(ierr);
  ctx->lme = lme;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme);CHKERRQ(ierr);
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIISetLME - Associate a linear matrix equation solver object (LME) to the
   eigenvalue solver.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIIGetLME()
@*/
PetscErrorCode EPSLyapIISetLME(EPS eps,LME lme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(lme,LME_CLASSID,2);
  PetscCheckSameComm(eps,1,lme,2);
  ierr = PetscTryMethod(eps,"EPSLyapIISetLME_C",(EPS,LME),(eps,lme));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLyapIIGetLME_LyapII(EPS eps,LME *lme)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) {
    ierr = LMECreate(PetscObjectComm((PetscObject)eps),&ctx->lme);CHKERRQ(ierr);
    ierr = LMESetOptionsPrefix(ctx->lme,((PetscObject)eps)->prefix);CHKERRQ(ierr);
    ierr = LMEAppendOptionsPrefix(ctx->lme,"eps_lyapii_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->lme,(PetscObject)eps,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->lme);CHKERRQ(ierr);
  }
  *lme = ctx->lme;
  PetscFunctionReturn(0);
}

/*@
   EPSLyapIIGetLME - Retrieve the linear matrix equation solver object (LME)
   associated with the eigenvalue solver.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  lme - the linear matrix equation solver object

   Level: advanced

.seealso: EPSLyapIISetLME()
@*/
PetscErrorCode EPSLyapIIGetLME(EPS eps,LME *lme)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(lme,2);
  ierr = PetscUseMethod(eps,"EPSLyapIIGetLME_C",(EPS,LME*),(eps,lme));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_LyapII(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->lme) { ierr = EPSLyapIIGetLME(eps,&ctx->lme);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = LMEView(ctx->lme,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  if (!ctx->lme) { ierr = LMEReset(ctx->lme);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_LyapII(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LYAPII     *ctx = (EPS_LYAPII*)eps->data;

  PetscFunctionBegin;
  ierr = LMEDestroy(&ctx->lme);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetDefaultST_LyapII(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EPSCreate_LyapII(EPS eps)
{
  EPS_LYAPII     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->solve          = EPSSolve_LyapII;
  eps->ops->setup          = EPSSetUp_LyapII;
  eps->ops->setfromoptions = EPSSetFromOptions_LyapII;
  eps->ops->reset          = EPSReset_LyapII;
  eps->ops->destroy        = EPSDestroy_LyapII;
  eps->ops->view           = EPSView_LyapII;
  eps->ops->setdefaultst   = EPSSetDefaultST_LyapII;
  eps->ops->backtransform  = EPSBackTransform_Default;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIISetLME_C",EPSLyapIISetLME_LyapII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLyapIIGetLME_C",EPSLyapIIGetLME_LyapII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

