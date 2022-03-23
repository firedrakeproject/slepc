/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc polynomial eigensolver: "stoar"

   Method: S-TOAR with spectrum slicing for symmetric quadratic eigenproblems

   Algorithm:

       Symmetric Two-Level Orthogonal Arnoldi.

   References:

       [1] C. Campos and J.E. Roman, "Inertia-based spectrum slicing
           for symmetric quadratic eigenvalue problems", Numer. Linear
           Algebra Appl. 27(4):e2293, 2020.
*/

#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/
#include "../src/pep/impls/krylov/pepkrylov.h"
#include <slepcblaslapack.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-slice-qep,\n"
  "   author = \"C. Campos and J. E. Roman\",\n"
  "   title = \"Inertia-based spectrum slicing for symmetric quadratic eigenvalue problems\",\n"
  "   journal = \"Numer. Linear Algebra Appl.\",\n"
  "   volume = \"27\",\n"
  "   number = \"4\",\n"
  "   pages = \"e2293\",\n"
  "   year = \"2020,\"\n"
  "   doi = \"https://doi.org/10.1002/nla.2293\"\n"
  "}\n";

#define SLICE_PTOL PETSC_SQRT_MACHINE_EPSILON

static PetscErrorCode PEPQSliceResetSR(PEP pep)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr=ctx->sr;
  PEP_shift      s;
  PetscInt       i;

  PetscFunctionBegin;
  if (sr) {
    /* Reviewing list of shifts to free memory */
    s = sr->s0;
    if (s) {
      while (s->neighb[1]) {
        s = s->neighb[1];
        CHKERRQ(PetscFree(s->neighb[0]));
      }
      CHKERRQ(PetscFree(s));
    }
    CHKERRQ(PetscFree(sr->S));
    for (i=0;i<pep->nconv;i++) CHKERRQ(PetscFree(sr->qinfo[i].q));
    CHKERRQ(PetscFree(sr->qinfo));
    for (i=0;i<3;i++) CHKERRQ(VecDestroy(&sr->v[i]));
    CHKERRQ(EPSDestroy(&sr->eps));
    CHKERRQ(PetscFree(sr));
  }
  ctx->sr = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PEPReset_STOAR_QSlice(PEP pep)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  CHKERRQ(PEPQSliceResetSR(pep));
  CHKERRQ(PetscFree(ctx->inertias));
  CHKERRQ(PetscFree(ctx->shifts));
  PetscFunctionReturn(0);
}

/*
  PEPQSliceAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors.
*/
static PetscErrorCode PEPQSliceAllocateSolution(PEP pep)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PetscInt       k;
  PetscLogDouble cnt;
  BVType         type;
  Vec            t;
  PEP_SR         sr = ctx->sr;

  PetscFunctionBegin;
  /* allocate space for eigenvalues and friends */
  k = PetscMax(1,sr->numEigs);
  CHKERRQ(PetscFree4(sr->eigr,sr->eigi,sr->errest,sr->perm));
  CHKERRQ(PetscCalloc4(k,&sr->eigr,k,&sr->eigi,k,&sr->errest,k,&sr->perm));
  CHKERRQ(PetscFree(sr->qinfo));
  CHKERRQ(PetscCalloc1(k,&sr->qinfo));
  cnt = 2*k*sizeof(PetscScalar) + 2*k*sizeof(PetscReal) + k*sizeof(PetscInt);
  CHKERRQ(PetscLogObjectMemory((PetscObject)pep,cnt));

  /* allocate sr->V and transfer options from pep->V */
  CHKERRQ(BVDestroy(&sr->V));
  CHKERRQ(BVCreate(PetscObjectComm((PetscObject)pep),&sr->V));
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)sr->V));
  if (!pep->V) CHKERRQ(PEPGetBV(pep,&pep->V));
  if (!((PetscObject)(pep->V))->type_name) CHKERRQ(BVSetType(sr->V,BVSVEC));
  else {
    CHKERRQ(BVGetType(pep->V,&type));
    CHKERRQ(BVSetType(sr->V,type));
  }
  CHKERRQ(STMatCreateVecsEmpty(pep->st,&t,NULL));
  CHKERRQ(BVSetSizesFromVec(sr->V,t,k+1));
  CHKERRQ(VecDestroy(&t));
  sr->ld = k;
  CHKERRQ(PetscFree(sr->S));
  CHKERRQ(PetscMalloc1((k+1)*sr->ld*(pep->nmat-1),&sr->S));
  PetscFunctionReturn(0);
}

/* Convergence test to compute positive Ritz values */
static PetscErrorCode ConvergedPositive(EPS eps,PetscScalar eigr,PetscScalar eigi,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscFunctionBegin;
  *errest = (PetscRealPart(eigr)>0.0)?0.0:res;
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQSliceMatGetInertia(PEP pep,PetscReal shift,PetscInt *inertia,PetscInt *zeros)
{
  KSP            ksp,kspr;
  PC             pc;
  Mat            F;
  PetscBool      flg;

  PetscFunctionBegin;
  if (!pep->solvematcoeffs) CHKERRQ(PetscMalloc1(pep->nmat,&pep->solvematcoeffs));
  if (shift==PETSC_MAX_REAL) { /* Inertia of matrix A[2] */
    pep->solvematcoeffs[0] = 0.0; pep->solvematcoeffs[1] = 0.0; pep->solvematcoeffs[2] = 1.0;
  } else CHKERRQ(PEPEvaluateBasis(pep,shift,0,pep->solvematcoeffs,NULL));
  CHKERRQ(STMatSetUp(pep->st,pep->sfactor,pep->solvematcoeffs));
  CHKERRQ(STGetKSP(pep->st,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCREDUNDANT,&flg));
  if (flg) {
    CHKERRQ(PCRedundantGetKSP(pc,&kspr));
    CHKERRQ(KSPGetPC(kspr,&pc));
  }
  CHKERRQ(PCFactorGetMatrix(pc,&F));
  CHKERRQ(MatGetInertia(F,inertia,zeros,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPQSliceGetInertia(PEP pep,PetscReal shift,PetscInt *inertia,PetscInt *zeros,PetscInt correction)
{
  KSP            ksp;
  Mat            P;
  PetscReal      nzshift=0.0,dot;
  PetscRandom    rand;
  PetscInt       nconv;
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr=ctx->sr;

  PetscFunctionBegin;
  if (shift >= PETSC_MAX_REAL) { /* Right-open interval */
    *inertia = 0;
  } else if (shift <= PETSC_MIN_REAL) {
    *inertia = 0;
    if (zeros) *zeros = 0;
  } else {
    /* If the shift is zero, perturb it to a very small positive value.
       The goal is that the nonzero pattern is the same in all cases and reuse
       the symbolic factorizations */
    nzshift = (shift==0.0)? 10.0/PETSC_MAX_REAL: shift;
    CHKERRQ(PEPQSliceMatGetInertia(pep,nzshift,inertia,zeros));
    CHKERRQ(STSetShift(pep->st,nzshift));
  }
  if (!correction) {
    if (shift >= PETSC_MAX_REAL) *inertia = 2*pep->n;
    else if (shift>PETSC_MIN_REAL) {
      CHKERRQ(STGetKSP(pep->st,&ksp));
      CHKERRQ(KSPGetOperators(ksp,&P,NULL));
      if (*inertia!=pep->n && !sr->v[0]) {
        CHKERRQ(MatCreateVecs(P,&sr->v[0],NULL));
        CHKERRQ(VecDuplicate(sr->v[0],&sr->v[1]));
        CHKERRQ(VecDuplicate(sr->v[0],&sr->v[2]));
        CHKERRQ(BVGetRandomContext(pep->V,&rand));
        CHKERRQ(VecSetRandom(sr->v[0],rand));
      }
      if (*inertia<pep->n && *inertia>0) {
        if (!sr->eps) {
          CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)pep),&sr->eps));
          CHKERRQ(EPSSetProblemType(sr->eps,EPS_HEP));
          CHKERRQ(EPSSetWhichEigenpairs(sr->eps,EPS_LARGEST_REAL));
        }
        CHKERRQ(EPSSetConvergenceTestFunction(sr->eps,ConvergedPositive,NULL,NULL));
        CHKERRQ(EPSSetOperators(sr->eps,P,NULL));
        CHKERRQ(EPSSolve(sr->eps));
        CHKERRQ(EPSGetConverged(sr->eps,&nconv));
        PetscCheck(nconv,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"Inertia computation fails in %g",(double)nzshift);
        CHKERRQ(EPSGetEigenpair(sr->eps,0,NULL,NULL,sr->v[0],sr->v[1]));
      }
      if (*inertia!=pep->n) {
        CHKERRQ(MatMult(pep->A[1],sr->v[0],sr->v[1]));
        CHKERRQ(MatMult(pep->A[2],sr->v[0],sr->v[2]));
        CHKERRQ(VecAXPY(sr->v[1],2*nzshift,sr->v[2]));
        CHKERRQ(VecDotRealPart(sr->v[1],sr->v[0],&dot));
        if (dot>0.0) *inertia = 2*pep->n-*inertia;
      }
    }
  } else if (correction<0) *inertia = 2*pep->n-*inertia;
  PetscFunctionReturn(0);
}

/*
   Check eigenvalue type - used only in non-hyperbolic problems.
   All computed eigenvalues must have the same definite type (positive or negative).
   If ini=TRUE the type is available in omega, otherwise we compute an eigenvalue
   closest to shift and determine its type.
 */
static PetscErrorCode PEPQSliceCheckEigenvalueType(PEP pep,PetscReal shift,PetscReal omega,PetscBool ini)
{
  PEP            pep2;
  ST             st;
  PetscInt       nconv;
  PetscScalar    lambda;
  PetscReal      dot;
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr=ctx->sr;

  PetscFunctionBegin;
  if (!ini) {
    PetscCheck(-(omega/(shift*ctx->alpha+ctx->beta))*sr->type>=0,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"Different positive/negative type detected in eigenvalue %g",(double)shift);
  } else {
    CHKERRQ(PEPCreate(PetscObjectComm((PetscObject)pep),&pep2));
    CHKERRQ(PEPSetOptionsPrefix(pep2,((PetscObject)pep)->prefix));
    CHKERRQ(PEPAppendOptionsPrefix(pep2,"pep_eigenvalue_type_"));
    CHKERRQ(PEPSetTolerances(pep2,PETSC_DEFAULT,pep->max_it/4));
    CHKERRQ(PEPSetType(pep2,PEPTOAR));
    CHKERRQ(PEPSetOperators(pep2,pep->nmat,pep->A));
    CHKERRQ(PEPSetWhichEigenpairs(pep2,PEP_TARGET_MAGNITUDE));
    CHKERRQ(PEPGetRG(pep2,&pep2->rg));
    CHKERRQ(RGSetType(pep2->rg,RGINTERVAL));
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(RGIntervalSetEndpoints(pep2->rg,pep->inta,pep->intb,-PETSC_SQRT_MACHINE_EPSILON,PETSC_SQRT_MACHINE_EPSILON));
#else
    CHKERRQ(RGIntervalSetEndpoints(pep2->rg,pep->inta,pep->intb,0.0,0.0));
#endif
    pep2->target = shift;
    st = pep2->st;
    pep2->st = pep->st;
    CHKERRQ(PEPSolve(pep2));
    CHKERRQ(PEPGetConverged(pep2,&nconv));
    if (nconv) {
      CHKERRQ(PEPGetEigenpair(pep2,0,&lambda,NULL,pep2->work[0],NULL));
      CHKERRQ(MatMult(pep->A[1],pep2->work[0],pep2->work[1]));
      CHKERRQ(MatMult(pep->A[2],pep2->work[0],pep2->work[2]));
      CHKERRQ(VecAXPY(pep2->work[1],2.0*lambda*pep->sfactor,pep2->work[2]));
      CHKERRQ(VecDotRealPart(pep2->work[1],pep2->work[0],&dot));
      CHKERRQ(PetscInfo(pep,"lambda=%g, %s type\n",(double)PetscRealPart(lambda),(dot>0.0)?"positive":"negative"));
      if (!sr->type) sr->type = (dot>0.0)?1:-1;
      else PetscCheck(sr->type*dot>=0.0,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"Different positive/negative type detected in eigenvalue %g",(double)PetscRealPart(lambda));
    }
    pep2->st = st;
    CHKERRQ(PEPDestroy(&pep2));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PEPQSliceDiscriminant(PEP pep,Vec u,Vec w,PetscReal *d,PetscReal *smas,PetscReal *smenos)
{
  PetscReal      ap,bp,cp,dis;

  PetscFunctionBegin;
  CHKERRQ(MatMult(pep->A[0],u,w));
  CHKERRQ(VecDotRealPart(w,u,&cp));
  CHKERRQ(MatMult(pep->A[1],u,w));
  CHKERRQ(VecDotRealPart(w,u,&bp));
  CHKERRQ(MatMult(pep->A[2],u,w));
  CHKERRQ(VecDotRealPart(w,u,&ap));
  dis = bp*bp-4*ap*cp;
  if (dis>=0.0 && smas) {
    if (ap>0) *smas = (-bp+PetscSqrtReal(dis))/(2*ap);
    else if (ap<0) *smas = (-bp-PetscSqrtReal(dis))/(2*ap);
    else {
      if (bp >0) *smas = -cp/bp;
      else *smas = PETSC_MAX_REAL;
    }
  }
  if (dis>=0.0 && smenos) {
    if (ap>0) *smenos = (-bp-PetscSqrtReal(dis))/(2*ap);
    else if (ap<0) *smenos = (-bp+PetscSqrtReal(dis))/(2*ap);
    else {
      if (bp<0) *smenos = -cp/bp;
      else *smenos = PETSC_MAX_REAL;
    }
  }
  if (d) *d = dis;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PEPQSliceEvaluateQEP(PEP pep,PetscScalar x,Mat M,MatStructure str)
{
  PetscFunctionBegin;
  CHKERRQ(MatCopy(pep->A[0],M,SAME_NONZERO_PATTERN));
  CHKERRQ(MatAXPY(M,x,pep->A[1],str));
  CHKERRQ(MatAXPY(M,x*x,pep->A[2],str));
  PetscFunctionReturn(0);
}

/*@
   PEPCheckDefiniteQEP - Determines if a symmetric/Hermitian quadratic eigenvalue problem
   is definite or not.

   Logically Collective on pep

   Input Parameter:
.  pep  - eigensolver context

   Output Parameters:
+  xi - first computed parameter
.  mu - second computed parameter
.  definite - flag indicating that the problem is definite
-  hyperbolic - flag indicating that the problem is hyperbolic

   Notes:
   This function is intended for quadratic eigenvalue problems, Q(lambda)=A*lambda^2+B*lambda+C,
   with symmetric (or Hermitian) coefficient matrices A,B,C.

   On output, the flag 'definite' may have the values -1 (meaning that the QEP is not
   definite), 1 (if the problem is definite), or 0 if the algorithm was not able to
   determine whether the problem is definite or not.

   If definite=1, the output flag 'hyperbolic' informs in a similar way about whether the
   problem is hyperbolic or not.

   If definite=1, the computed values xi and mu satisfy Q(xi)<0 and Q(mu)>0, as
   obtained via the method proposed in [Niendorf and Voss, LAA 2010]. Furthermore, if
   hyperbolic=1 then only xi is computed.

   Level: advanced

.seealso: PEPSetProblemType()
@*/
PetscErrorCode PEPCheckDefiniteQEP(PEP pep,PetscReal *xi,PetscReal *mu,PetscInt *definite,PetscInt *hyperbolic)
{
  PetscRandom    rand;
  Vec            u,w;
  PetscReal      d=0.0,s=0.0,sp,mut=0.0,omg=0.0,omgp;
  PetscInt       k,its=10,hyp=0,check=0,nconv,inertia,n;
  Mat            M=NULL;
  MatStructure   str;
  EPS            eps;
  PetscBool      transform,ptypehyp;

  PetscFunctionBegin;
  PetscCheck(pep->problem_type==PEP_HERMITIAN || pep->problem_type==PEP_HYPERBOLIC,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Only available for Hermitian (or hyperbolic) problems");
  ptypehyp = (pep->problem_type==PEP_HYPERBOLIC)? PETSC_TRUE: PETSC_FALSE;
  if (!pep->st) CHKERRQ(PEPGetST(pep,&pep->st));
  CHKERRQ(PEPSetDefaultST(pep));
  CHKERRQ(STSetMatrices(pep->st,pep->nmat,pep->A));
  CHKERRQ(MatGetSize(pep->A[0],&n,NULL));
  CHKERRQ(STGetTransform(pep->st,&transform));
  CHKERRQ(STSetTransform(pep->st,PETSC_FALSE));
  CHKERRQ(STSetUp(pep->st));
  CHKERRQ(MatCreateVecs(pep->A[0],&u,&w));
  CHKERRQ(PEPGetBV(pep,&pep->V));
  CHKERRQ(BVGetRandomContext(pep->V,&rand));
  CHKERRQ(VecSetRandom(u,rand));
  CHKERRQ(VecNormalize(u,NULL));
  CHKERRQ(PEPQSliceDiscriminant(pep,u,w,&d,&s,NULL));
  if (d<0.0) check = -1;
  if (!check) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)pep),&eps));
    CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
    CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));
    CHKERRQ(EPSSetTolerances(eps,PetscSqrtReal(PETSC_SQRT_MACHINE_EPSILON),PETSC_DECIDE));
    CHKERRQ(MatDuplicate(pep->A[0],MAT_DO_NOT_COPY_VALUES,&M));
    CHKERRQ(STGetMatStructure(pep->st,&str));
  }
  for (k=0;k<its&&!check;k++) {
    CHKERRQ(PEPQSliceEvaluateQEP(pep,s,M,str));
    CHKERRQ(EPSSetOperators(eps,M,NULL));
    CHKERRQ(EPSSolve(eps));
    CHKERRQ(EPSGetConverged(eps,&nconv));
    if (!nconv) break;
    CHKERRQ(EPSGetEigenpair(eps,0,NULL,NULL,u,w));
    sp = s;
    CHKERRQ(PEPQSliceDiscriminant(pep,u,w,&d,&s,&omg));
    if (d<0.0) {check = -1; break;}
    if (PetscAbsReal((s-sp)/s)<100*PETSC_MACHINE_EPSILON) break;
    if (s>sp) {hyp = -1;}
    mut = 2*s-sp;
    CHKERRQ(PEPQSliceMatGetInertia(pep,mut,&inertia,NULL));
    if (inertia == n) {check = 1; break;}
  }
  for (;k<its&&!check;k++) {
    mut = (s-omg)/2;
    CHKERRQ(PEPQSliceMatGetInertia(pep,mut,&inertia,NULL));
    if (inertia == n) {check = 1; break;}
    if (PetscAbsReal((s-omg)/omg)<100*PETSC_MACHINE_EPSILON) break;
    CHKERRQ(PEPQSliceEvaluateQEP(pep,omg,M,str));
    CHKERRQ(EPSSetOperators(eps,M,NULL));
    CHKERRQ(EPSSolve(eps));
    CHKERRQ(EPSGetConverged(eps,&nconv));
    if (!nconv) break;
    CHKERRQ(EPSGetEigenpair(eps,0,NULL,NULL,u,w));
    omgp = omg;
    CHKERRQ(PEPQSliceDiscriminant(pep,u,w,&d,NULL,&omg));
    if (d<0.0) {check = -1; break;}
    if (omg<omgp) hyp = -1;
  }
  if (check==1) *xi = mut;
  PetscCheck(hyp!=-1 || !ptypehyp,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Problem does not satisfy hyperbolic test; consider removing the hyperbolicity flag");
  if (check==1 && hyp==0) {
    CHKERRQ(PEPQSliceMatGetInertia(pep,PETSC_MAX_REAL,&inertia,NULL));
    if (inertia == 0) hyp = 1;
    else hyp = -1;
  }
  if (check==1 && hyp!=1) {
    check = 0;
    CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
    for (;k<its&&!check;k++) {
      CHKERRQ(PEPQSliceEvaluateQEP(pep,s,M,str));
      CHKERRQ(EPSSetOperators(eps,M,NULL));
      CHKERRQ(EPSSolve(eps));
      CHKERRQ(EPSGetConverged(eps,&nconv));
      if (!nconv) break;
      CHKERRQ(EPSGetEigenpair(eps,0,NULL,NULL,u,w));
      sp = s;
      CHKERRQ(PEPQSliceDiscriminant(pep,u,w,&d,&s,&omg));
      if (d<0.0) {check = -1; break;}
      if (PetscAbsReal((s-sp)/s)<100*PETSC_MACHINE_EPSILON) break;
      mut = 2*s-sp;
      CHKERRQ(PEPQSliceMatGetInertia(pep,mut,&inertia,NULL));
      if (inertia == 0) {check = 1; break;}
    }
    for (;k<its&&!check;k++) {
      mut = (s-omg)/2;
      CHKERRQ(PEPQSliceMatGetInertia(pep,mut,&inertia,NULL));
      if (inertia == 0) {check = 1; break;}
      if (PetscAbsReal((s-omg)/omg)<100*PETSC_MACHINE_EPSILON) break;
      CHKERRQ(PEPQSliceEvaluateQEP(pep,omg,M,str));
      CHKERRQ(EPSSetOperators(eps,M,NULL));
      CHKERRQ(EPSSolve(eps));
      CHKERRQ(EPSGetConverged(eps,&nconv));
      if (!nconv) break;
      CHKERRQ(EPSGetEigenpair(eps,0,NULL,NULL,u,w));
      CHKERRQ(PEPQSliceDiscriminant(pep,u,w,&d,NULL,&omg));
      if (d<0.0) {check = -1; break;}
    }
  }
  if (check==1) *mu = mut;
  *definite = check;
  *hyperbolic = hyp;
  if (M) CHKERRQ(MatDestroy(&M));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(STSetTransform(pep->st,transform));
  PetscFunctionReturn(0);
}

/*
   Dummy backtransform operation
 */
static PetscErrorCode PEPBackTransform_Skip(PEP pep)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetUp_STOAR_QSlice(PEP pep)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr;
  PetscInt       ld,i,zeros=0;
  SlepcSC        sc;
  PetscReal      r;

  PetscFunctionBegin;
  PEPCheckSinvertCayley(pep);
  PetscCheck(pep->inta<pep->intb,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues unless you provide a computational interval with PEPSetInterval()");
  PetscCheck(pep->intb<PETSC_MAX_REAL || pep->inta>PETSC_MIN_REAL,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
  PEPCheckUnsupportedCondition(pep,PEP_FEATURE_STOPPING,PETSC_TRUE," (with spectrum slicing)");
  if (pep->tol==PETSC_DEFAULT) {
#if defined(PETSC_USE_REAL_SINGLE)
    pep->tol = SLEPC_DEFAULT_TOL;
#else
    /* use tighter tolerance */
    pep->tol = SLEPC_DEFAULT_TOL*1e-2;
#endif
  }
  if (ctx->nev==1) ctx->nev = PetscMin(20,pep->n);  /* nev not set, use default value */
  PetscCheck(pep->n<=10 || ctx->nev>=10,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"nev cannot be less than 10 in spectrum slicing runs");
  pep->ops->backtransform = PEPBackTransform_Skip;
  if (pep->max_it==PETSC_DEFAULT) pep->max_it = 100;

  /* create spectrum slicing context and initialize it */
  CHKERRQ(PEPQSliceResetSR(pep));
  CHKERRQ(PetscNewLog(pep,&sr));
  ctx->sr   = sr;
  sr->itsKs = 0;
  sr->nleap = 0;
  sr->sPres = NULL;

  if (pep->solvematcoeffs) CHKERRQ(PetscFree(pep->solvematcoeffs));
  CHKERRQ(PetscMalloc1(pep->nmat,&pep->solvematcoeffs));
  if (!pep->st) CHKERRQ(PEPGetST(pep,&pep->st));
  CHKERRQ(STSetTransform(pep->st,PETSC_FALSE));
  CHKERRQ(STSetUp(pep->st));

  ctx->hyperbolic = (pep->problem_type==PEP_HYPERBOLIC)? PETSC_TRUE: PETSC_FALSE;

  /* check presence of ends and finding direction */
  if (pep->inta > PETSC_MIN_REAL || pep->intb >= PETSC_MAX_REAL) {
    sr->int0 = pep->inta;
    sr->int1 = pep->intb;
    sr->dir = 1;
    if (pep->intb >= PETSC_MAX_REAL) { /* Right-open interval */
      sr->hasEnd = PETSC_FALSE;
    } else sr->hasEnd = PETSC_TRUE;
  } else {
    sr->int0 = pep->intb;
    sr->int1 = pep->inta;
    sr->dir = -1;
    sr->hasEnd = PetscNot(pep->inta <= PETSC_MIN_REAL);
  }

  /* compute inertia0 */
  CHKERRQ(PEPQSliceGetInertia(pep,sr->int0,&sr->inertia0,ctx->detect?&zeros:NULL,ctx->hyperbolic?0:1));
  PetscCheck(!zeros || (sr->int0!=pep->inta && sr->int0!=pep->intb),((PetscObject)pep)->comm,PETSC_ERR_USER,"Found singular matrix for the transformed problem in the interval endpoint");
  if (!ctx->hyperbolic && ctx->checket) CHKERRQ(PEPQSliceCheckEigenvalueType(pep,sr->int0,0.0,PETSC_TRUE));

  /* compute inertia1 */
  CHKERRQ(PEPQSliceGetInertia(pep,sr->int1,&sr->inertia1,ctx->detect?&zeros:NULL,ctx->hyperbolic?0:1));
  PetscCheck(!zeros,((PetscObject)pep)->comm,PETSC_ERR_USER,"Found singular matrix for the transformed problem in an interval endpoint defined by user");
  if (!ctx->hyperbolic && ctx->checket && sr->hasEnd) {
    CHKERRQ(PEPQSliceCheckEigenvalueType(pep,sr->int1,0.0,PETSC_TRUE));
    PetscCheck(sr->type || sr->inertia1==sr->inertia0,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"No information of eigenvalue type in interval");
    PetscCheck(!sr->type || sr->inertia1!=sr->inertia0,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"Different positive/negative type detected");
    if (sr->dir*(sr->inertia1-sr->inertia0)<0) {
      sr->intcorr = -1;
      sr->inertia0 = 2*pep->n-sr->inertia0;
      sr->inertia1 = 2*pep->n-sr->inertia1;
    } else sr->intcorr = 1;
  } else {
    if (sr->inertia0<=pep->n && sr->inertia1<=pep->n) sr->intcorr = 1;
    else if (sr->inertia0>=pep->n && sr->inertia1>=pep->n) sr->intcorr = -1;
  }

  if (sr->hasEnd) {
    sr->dir = -sr->dir; r = sr->int0; sr->int0 = sr->int1; sr->int1 = r;
    i = sr->inertia0; sr->inertia0 = sr->inertia1; sr->inertia1 = i;
  }

  /* number of eigenvalues in interval */
  sr->numEigs = (sr->dir)*(sr->inertia1 - sr->inertia0);
  CHKERRQ(PetscInfo(pep,"QSlice setup: allocating for %" PetscInt_FMT " eigenvalues in [%g,%g]\n",sr->numEigs,(double)pep->inta,(double)pep->intb));
  if (sr->numEigs) {
    CHKERRQ(PEPQSliceAllocateSolution(pep));
    CHKERRQ(PEPSetDimensions_Default(pep,ctx->nev,&ctx->ncv,&ctx->mpd));
    pep->nev = ctx->nev; pep->ncv = ctx->ncv; pep->mpd = ctx->mpd;
    ld   = ctx->ncv+2;
    CHKERRQ(DSSetType(pep->ds,DSGHIEP));
    CHKERRQ(DSSetCompact(pep->ds,PETSC_TRUE));
    CHKERRQ(DSSetExtraRow(pep->ds,PETSC_TRUE));
    CHKERRQ(DSAllocate(pep->ds,ld));
    CHKERRQ(DSGetSlepcSC(pep->ds,&sc));
    sc->rg            = NULL;
    sc->comparison    = SlepcCompareLargestMagnitude;
    sc->comparisonctx = NULL;
    sc->map           = NULL;
    sc->mapobj        = NULL;
  } else {pep->ncv = 0; pep->nev = 0; pep->mpd = 0;}
  PetscFunctionReturn(0);
}

/*
   Fills the fields of a shift structure
*/
static PetscErrorCode PEPCreateShift(PEP pep,PetscReal val,PEP_shift neighb0,PEP_shift neighb1)
{
  PEP_shift      s,*pending2;
  PetscInt       i;
  PEP_SR         sr;
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  CHKERRQ(PetscNewLog(pep,&s));
  s->value = val;
  s->neighb[0] = neighb0;
  if (neighb0) neighb0->neighb[1] = s;
  s->neighb[1] = neighb1;
  if (neighb1) neighb1->neighb[0] = s;
  s->comp[0] = PETSC_FALSE;
  s->comp[1] = PETSC_FALSE;
  s->index = -1;
  s->neigs = 0;
  s->nconv[0] = s->nconv[1] = 0;
  s->nsch[0] = s->nsch[1]=0;
  /* Inserts in the stack of pending shifts */
  /* If needed, the array is resized */
  if (sr->nPend >= sr->maxPend) {
    sr->maxPend *= 2;
    CHKERRQ(PetscMalloc1(sr->maxPend,&pending2));
    CHKERRQ(PetscLogObjectMemory((PetscObject)pep,sr->maxPend*sizeof(PEP_shift*)));
    for (i=0;i<sr->nPend;i++) pending2[i] = sr->pending[i];
    CHKERRQ(PetscFree(sr->pending));
    sr->pending = pending2;
  }
  sr->pending[sr->nPend++]=s;
  PetscFunctionReturn(0);
}

/* Provides next shift to be computed */
static PetscErrorCode PEPExtractShift(PEP pep)
{
  PetscInt       iner,zeros=0;
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr;
  PetscReal      newShift,aux;
  PEP_shift      sPres;

  PetscFunctionBegin;
  sr = ctx->sr;
  if (sr->nPend > 0) {
    if (sr->dirch) {
      aux = sr->int1; sr->int1 = sr->int0; sr->int0 = aux;
      iner = sr->inertia1; sr->inertia1 = sr->inertia0; sr->inertia0 = iner;
      sr->dir *= -1;
      CHKERRQ(PetscFree(sr->s0->neighb[1]));
      CHKERRQ(PetscFree(sr->s0));
      sr->nPend--;
      CHKERRQ(PEPCreateShift(pep,sr->int0,NULL,NULL));
      sr->sPrev = NULL;
      sr->sPres = sr->pending[--sr->nPend];
      pep->target = sr->sPres->value;
      sr->s0 = sr->sPres;
      pep->reason = PEP_CONVERGED_ITERATING;
    } else {
      sr->sPrev = sr->sPres;
      sr->sPres = sr->pending[--sr->nPend];
    }
    sPres = sr->sPres;
    CHKERRQ(PEPQSliceGetInertia(pep,sPres->value,&iner,ctx->detect?&zeros:NULL,sr->intcorr));
    if (zeros) {
      newShift = sPres->value*(1.0+SLICE_PTOL);
      if (sr->dir*(sPres->neighb[0] && newShift-sPres->neighb[0]->value) < 0) newShift = (sPres->value+sPres->neighb[0]->value)/2;
      else if (sPres->neighb[1] && sr->dir*(sPres->neighb[1]->value-newShift) < 0) newShift = (sPres->value+sPres->neighb[1]->value)/2;
      CHKERRQ(PEPQSliceGetInertia(pep,newShift,&iner,&zeros,sr->intcorr));
      PetscCheck(!zeros,((PetscObject)pep)->comm,PETSC_ERR_CONV_FAILED,"Inertia computation fails in %g",(double)newShift);
      sPres->value = newShift;
    }
    sr->sPres->inertia = iner;
    pep->target = sr->sPres->value;
    pep->reason = PEP_CONVERGED_ITERATING;
    pep->its = 0;
  } else sr->sPres = NULL;
  PetscFunctionReturn(0);
}

/*
  Obtains value of subsequent shift
*/
static PetscErrorCode PEPGetNewShiftValue(PEP pep,PetscInt side,PetscReal *newS)
{
  PetscReal lambda,d_prev;
  PetscInt  i,idxP;
  PEP_SR    sr;
  PEP_shift sPres,s;
  PEP_STOAR *ctx=(PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  sPres = sr->sPres;
  if (sPres->neighb[side]) {
  /* Completing a previous interval */
    if (!sPres->neighb[side]->neighb[side] && sPres->neighb[side]->nconv[side]==0) { /* One of the ends might be too far from eigenvalues */
      if (side) *newS = (sPres->value + PetscRealPart(sr->eigr[sr->perm[sr->indexEig-1]]))/2;
      else *newS = (sPres->value + PetscRealPart(sr->eigr[sr->perm[0]]))/2;
    } else *newS=(sPres->value + sPres->neighb[side]->value)/2;
  } else { /* (Only for side=1). Creating a new interval. */
    if (sPres->neigs==0) {/* No value has been accepted*/
      if (sPres->neighb[0]) {
        /* Multiplying by 10 the previous distance */
        *newS = sPres->value + 10*(sr->dir)*PetscAbsReal(sPres->value - sPres->neighb[0]->value);
        sr->nleap++;
        /* Stops when the interval is open and no values are found in the last 5 shifts (there might be infinite eigenvalues) */
        PetscCheck(sr->hasEnd || sr->nleap<=5,PetscObjectComm((PetscObject)pep),PETSC_ERR_CONV_FAILED,"Unable to compute the wanted eigenvalues with open interval");
      } else { /* First shift */
        if (pep->nconv != 0) {
          /* Unaccepted values give information for next shift */
          idxP=0;/* Number of values left from shift */
          for (i=0;i<pep->nconv;i++) {
            lambda = PetscRealPart(pep->eigr[i]);
            if ((sr->dir)*(lambda - sPres->value) <0) idxP++;
            else break;
          }
          /* Avoiding subtraction of eigenvalues (might be the same).*/
          if (idxP>0) {
            d_prev = PetscAbsReal(sPres->value - PetscRealPart(pep->eigr[0]))/(idxP+0.3);
          } else {
            d_prev = PetscAbsReal(sPres->value - PetscRealPart(pep->eigr[pep->nconv-1]))/(pep->nconv+0.3);
          }
          *newS = sPres->value + ((sr->dir)*d_prev*pep->nev)/2;
          sr->dirch = PETSC_FALSE;
        } else { /* No values found, no information for next shift */
          PetscCheck(!sr->dirch,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"First shift renders no information");
          sr->dirch = PETSC_TRUE;
          *newS = sr->int1;
        }
      }
    } else { /* Accepted values found */
      sr->dirch = PETSC_FALSE;
      sr->nleap = 0;
      /* Average distance of values in previous subinterval */
      s = sPres->neighb[0];
      while (s && PetscAbs(s->inertia - sPres->inertia)==0) {
        s = s->neighb[0];/* Looking for previous shifts with eigenvalues within */
      }
      if (s) {
        d_prev = PetscAbsReal((sPres->value - s->value)/(sPres->inertia - s->inertia));
      } else { /* First shift. Average distance obtained with values in this shift */
        /* first shift might be too far from first wanted eigenvalue (no values found outside the interval)*/
        if ((sr->dir)*(PetscRealPart(sr->eigr[0])-sPres->value)>0 && PetscAbsReal((PetscRealPart(sr->eigr[sr->indexEig-1]) - PetscRealPart(sr->eigr[0]))/PetscRealPart(sr->eigr[0])) > PetscSqrtReal(pep->tol)) {
          d_prev =  PetscAbsReal((PetscRealPart(sr->eigr[sr->indexEig-1]) - PetscRealPart(sr->eigr[0])))/(sPres->neigs+0.3);
        } else {
          d_prev = PetscAbsReal(PetscRealPart(sr->eigr[sr->indexEig-1]) - sPres->value)/(sPres->neigs+0.3);
        }
      }
      /* Average distance is used for next shift by adding it to value on the right or to shift */
      if ((sr->dir)*(PetscRealPart(sr->eigr[sPres->index + sPres->neigs -1]) - sPres->value)>0) {
        *newS = PetscRealPart(sr->eigr[sPres->index + sPres->neigs -1])+ ((sr->dir)*d_prev*(pep->nev))/2;
      } else { /* Last accepted value is on the left of shift. Adding to shift */
        *newS = sPres->value + ((sr->dir)*d_prev*(pep->nev))/2;
      }
    }
    /* End of interval can not be surpassed */
    if ((sr->dir)*(sr->int1 - *newS) < 0) *newS = sr->int1;
  }/* of neighb[side]==null */
  PetscFunctionReturn(0);
}

/*
  Function for sorting an array of real values
*/
static PetscErrorCode sortRealEigenvalues(PetscScalar *r,PetscInt *perm,PetscInt nr,PetscBool prev,PetscInt dir)
{
  PetscReal re;
  PetscInt  i,j,tmp;

  PetscFunctionBegin;
  if (!prev) for (i=0;i<nr;i++) perm[i] = i;
  /* Insertion sort */
  for (i=1;i<nr;i++) {
    re = PetscRealPart(r[perm[i]]);
    j = i-1;
    while (j>=0 && dir*(re - PetscRealPart(r[perm[j]])) <= 0) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
    }
  }
  PetscFunctionReturn(0);
}

/* Stores the pairs obtained since the last shift in the global arrays */
static PetscErrorCode PEPStoreEigenpairs(PEP pep)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PetscReal      lambda,err,*errest;
  PetscInt       i,*aux,count=0,ndef,ld,nconv=pep->nconv,d=pep->nmat-1,idx;
  PetscBool      iscayley,divide=PETSC_FALSE;
  PEP_SR         sr = ctx->sr;
  PEP_shift      sPres;
  Vec            w,vomega;
  Mat            MS;
  BV             tV;
  PetscScalar    *S,*eigr,*tS,*omega;

  PetscFunctionBegin;
  sPres = sr->sPres;
  sPres->index = sr->indexEig;

  if (nconv>sr->ndef0+sr->ndef1) {
    /* Back-transform */
    CHKERRQ(STBackTransform(pep->st,nconv,pep->eigr,pep->eigi));
    for (i=0;i<nconv;i++) {
#if defined(PETSC_USE_COMPLEX)
      if (PetscImaginaryPart(pep->eigr[i])) pep->eigr[i] = sr->int0-sr->dir;
#else
      if (pep->eigi[i]) pep->eigr[i] = sr->int0-sr->dir;
#endif
    }
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STCAYLEY,&iscayley));
    /* Sort eigenvalues */
    CHKERRQ(sortRealEigenvalues(pep->eigr,pep->perm,nconv,PETSC_FALSE,sr->dir));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nconv,&vomega));
    CHKERRQ(BVGetSignature(ctx->V,vomega));
    CHKERRQ(VecGetArray(vomega,&omega));
    CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
    CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
    CHKERRQ(MatDenseGetArray(MS,&S));
    /* Values stored in global array */
    CHKERRQ(PetscCalloc4(nconv,&eigr,nconv,&errest,nconv*nconv*d,&tS,nconv,&aux));
    ndef = sr->ndef0+sr->ndef1;
    for (i=0;i<nconv;i++) {
      lambda = PetscRealPart(pep->eigr[pep->perm[i]]);
      err = pep->errest[pep->perm[i]];
      if ((sr->dir)*(lambda - sPres->ext[0]) > 0 && (sr->dir)*(sPres->ext[1] - lambda) > 0) {/* Valid value */
        PetscCheck(sr->indexEig+count-ndef<sr->numEigs,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Unexpected error in Spectrum Slicing");
        CHKERRQ(PEPQSliceCheckEigenvalueType(pep,lambda,PetscRealPart(omega[pep->perm[i]]),PETSC_FALSE));
        eigr[count] = lambda;
        errest[count] = err;
        if (((sr->dir)*(sPres->value - lambda) > 0) && ((sr->dir)*(lambda - sPres->ext[0]) > 0)) sPres->nconv[0]++;
        if (((sr->dir)*(lambda - sPres->value) > 0) && ((sr->dir)*(sPres->ext[1] - lambda) > 0)) sPres->nconv[1]++;
        CHKERRQ(PetscArraycpy(tS+count*(d*nconv),S+pep->perm[i]*(d*ld),nconv));
        CHKERRQ(PetscArraycpy(tS+count*(d*nconv)+nconv,S+pep->perm[i]*(d*ld)+ld,nconv));
        count++;
      }
    }
    CHKERRQ(VecRestoreArray(vomega,&omega));
    CHKERRQ(VecDestroy(&vomega));
    for (i=0;i<count;i++) {
      CHKERRQ(PetscArraycpy(S+i*(d*ld),tS+i*nconv*d,nconv));
      CHKERRQ(PetscArraycpy(S+i*(d*ld)+ld,tS+i*nconv*d+nconv,nconv));
    }
    CHKERRQ(MatDenseRestoreArray(MS,&S));
    CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    CHKERRQ(BVSetActiveColumns(ctx->V,0,count));
    CHKERRQ(BVTensorCompress(ctx->V,count));
    if (sr->sPres->nconv[0] && sr->sPres->nconv[1]) {
      divide = PETSC_TRUE;
      CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
      CHKERRQ(MatDenseGetArray(MS,&S));
      CHKERRQ(PetscArrayzero(tS,nconv*nconv*d));
      for (i=0;i<count;i++) {
        CHKERRQ(PetscArraycpy(tS+i*nconv*d,S+i*(d*ld),count));
        CHKERRQ(PetscArraycpy(tS+i*nconv*d+nconv,S+i*(d*ld)+ld,count));
      }
      CHKERRQ(MatDenseRestoreArray(MS,&S));
      CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
      CHKERRQ(BVSetActiveColumns(pep->V,0,count));
      CHKERRQ(BVDuplicateResize(pep->V,count,&tV));
      CHKERRQ(BVCopy(pep->V,tV));
    }
    if (sr->sPres->nconv[0]) {
      if (divide) {
        CHKERRQ(BVSetActiveColumns(ctx->V,0,sr->sPres->nconv[0]));
        CHKERRQ(BVTensorCompress(ctx->V,sr->sPres->nconv[0]));
      }
      for (i=0;i<sr->ndef0;i++) aux[i] = sr->idxDef0[i];
      for (i=sr->ndef0;i<sr->sPres->nconv[0];i++) aux[i] = sr->indexEig+i-sr->ndef0;
      CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
      CHKERRQ(MatDenseGetArray(MS,&S));
      for (i=0;i<sr->sPres->nconv[0];i++) {
        sr->eigr[aux[i]] = eigr[i];
        sr->errest[aux[i]] = errest[i];
        CHKERRQ(BVGetColumn(pep->V,i,&w));
        CHKERRQ(BVInsertVec(sr->V,aux[i],w));
        CHKERRQ(BVRestoreColumn(pep->V,i,&w));
        idx = sr->ld*d*aux[i];
        CHKERRQ(PetscArrayzero(sr->S+idx,sr->ld*d));
        CHKERRQ(PetscArraycpy(sr->S+idx,S+i*(ld*d),sr->sPres->nconv[0]));
        CHKERRQ(PetscArraycpy(sr->S+idx+sr->ld,S+i*(ld*d)+ld,sr->sPres->nconv[0]));
        CHKERRQ(PetscFree(sr->qinfo[aux[i]].q));
        CHKERRQ(PetscMalloc1(sr->sPres->nconv[0],&sr->qinfo[aux[i]].q));
        CHKERRQ(PetscArraycpy(sr->qinfo[aux[i]].q,aux,sr->sPres->nconv[0]));
        sr->qinfo[aux[i]].nq = sr->sPres->nconv[0];
      }
      CHKERRQ(MatDenseRestoreArray(MS,&S));
      CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    }

    if (sr->sPres->nconv[1]) {
      if (divide) {
        CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
        CHKERRQ(MatDenseGetArray(MS,&S));
        for (i=0;i<sr->sPres->nconv[1];i++) {
          CHKERRQ(PetscArraycpy(S+i*(d*ld),tS+(sr->sPres->nconv[0]+i)*nconv*d,count));
          CHKERRQ(PetscArraycpy(S+i*(d*ld)+ld,tS+(sr->sPres->nconv[0]+i)*nconv*d+nconv,count));
        }
        CHKERRQ(MatDenseRestoreArray(MS,&S));
        CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
        CHKERRQ(BVSetActiveColumns(pep->V,0,count));
        CHKERRQ(BVCopy(tV,pep->V));
        CHKERRQ(BVSetActiveColumns(ctx->V,0,sr->sPres->nconv[1]));
        CHKERRQ(BVTensorCompress(ctx->V,sr->sPres->nconv[1]));
      }
      for (i=0;i<sr->ndef1;i++) aux[i] = sr->idxDef1[i];
      for (i=sr->ndef1;i<sr->sPres->nconv[1];i++) aux[i] = sr->indexEig+sr->sPres->nconv[0]-sr->ndef0+i-sr->ndef1;
      CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
      CHKERRQ(MatDenseGetArray(MS,&S));
      for (i=0;i<sr->sPres->nconv[1];i++) {
        sr->eigr[aux[i]] = eigr[sr->sPres->nconv[0]+i];
        sr->errest[aux[i]] = errest[sr->sPres->nconv[0]+i];
        CHKERRQ(BVGetColumn(pep->V,i,&w));
        CHKERRQ(BVInsertVec(sr->V,aux[i],w));
        CHKERRQ(BVRestoreColumn(pep->V,i,&w));
        idx = sr->ld*d*aux[i];
        CHKERRQ(PetscArrayzero(sr->S+idx,sr->ld*d));
        CHKERRQ(PetscArraycpy(sr->S+idx,S+i*(ld*d),sr->sPres->nconv[1]));
        CHKERRQ(PetscArraycpy(sr->S+idx+sr->ld,S+i*(ld*d)+ld,sr->sPres->nconv[1]));
        CHKERRQ(PetscFree(sr->qinfo[aux[i]].q));
        CHKERRQ(PetscMalloc1(sr->sPres->nconv[1],&sr->qinfo[aux[i]].q));
        CHKERRQ(PetscArraycpy(sr->qinfo[aux[i]].q,aux,sr->sPres->nconv[1]));
        sr->qinfo[aux[i]].nq = sr->sPres->nconv[1];
      }
      CHKERRQ(MatDenseRestoreArray(MS,&S));
      CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    }
    sPres->neigs = count-sr->ndef0-sr->ndef1;
    sr->indexEig += sPres->neigs;
    sPres->nconv[0]-= sr->ndef0;
    sPres->nconv[1]-= sr->ndef1;
    CHKERRQ(PetscFree4(eigr,errest,tS,aux));
  } else {
    sPres->neigs = 0;
    sPres->nconv[0]= 0;
    sPres->nconv[1]= 0;
  }
  /* Global ordering array updating */
  CHKERRQ(sortRealEigenvalues(sr->eigr,sr->perm,sr->indexEig,PETSC_FALSE,sr->dir));
  /* Check for completion */
  sPres->comp[0] = PetscNot(sPres->nconv[0] < sPres->nsch[0]);
  sPres->comp[1] = PetscNot(sPres->nconv[1] < sPres->nsch[1]);
  PetscCheck(sPres->nconv[0]<=sPres->nsch[0] && sPres->nconv[1]<=sPres->nsch[1],PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia");
  if (divide) CHKERRQ(BVDestroy(&tV));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLookForDeflation(PEP pep)
{
  PetscReal val;
  PetscInt  i,count0=0,count1=0;
  PEP_shift sPres;
  PetscInt  ini,fin;
  PEP_SR    sr;
  PEP_STOAR *ctx=(PEP_STOAR*)pep->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  sPres = sr->sPres;

  if (sPres->neighb[0]) ini = (sr->dir)*(sPres->neighb[0]->inertia - sr->inertia0);
  else ini = 0;
  fin = sr->indexEig;
  /* Selection of ends for searching new values */
  if (!sPres->neighb[0]) sPres->ext[0] = sr->int0;/* First shift */
  else sPres->ext[0] = sPres->neighb[0]->value;
  if (!sPres->neighb[1]) {
    if (sr->hasEnd) sPres->ext[1] = sr->int1;
    else sPres->ext[1] = (sr->dir > 0)?PETSC_MAX_REAL:PETSC_MIN_REAL;
  } else sPres->ext[1] = sPres->neighb[1]->value;
  /* Selection of values between right and left ends */
  for (i=ini;i<fin;i++) {
    val=PetscRealPart(sr->eigr[sr->perm[i]]);
    /* Values to the right of left shift */
    if ((sr->dir)*(val - sPres->ext[1]) < 0) {
      if ((sr->dir)*(val - sPres->value) < 0) count0++;
      else count1++;
    } else break;
  }
  /* The number of values on each side are found */
  if (sPres->neighb[0]) {
    sPres->nsch[0] = (sr->dir)*(sPres->inertia - sPres->neighb[0]->inertia)-count0;
    PetscCheck(sPres->nsch[0]>=0,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia");
  } else sPres->nsch[0] = 0;

  if (sPres->neighb[1]) {
    sPres->nsch[1] = (sr->dir)*(sPres->neighb[1]->inertia - sPres->inertia) - count1;
    PetscCheck(sPres->nsch[1]>=0,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia");
  } else sPres->nsch[1] = (sr->dir)*(sr->inertia1 - sPres->inertia);

  /* Completing vector of indexes for deflation */
  for (i=0;i<count0;i++) sr->idxDef0[i] = sr->perm[ini+i];
  sr->ndef0 = count0;
  for (i=0;i<count1;i++) sr->idxDef1[i] = sr->perm[ini+count0+i];
  sr->ndef1 = count1;
  PetscFunctionReturn(0);
}

/*
  Compute a run of Lanczos iterations
*/
static PetscErrorCode PEPSTOARrun_QSlice(PEP pep,PetscReal *a,PetscReal *b,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscBool *symmlost,Vec *t_)
{
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       i,j,m=*M,l,lock;
  PetscInt       lds,d,ld,offq,nqt,ldds;
  Vec            v=t_[0],t=t_[1],q=t_[2];
  PetscReal      norm,sym=0.0,fro=0.0,*f;
  PetscScalar    *y,*S,sigma;
  PetscBLASInt   j_,one=1;
  PetscBool      lindep;
  Mat            MS;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(*M,&y));
  CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
  CHKERRQ(BVTensorGetDegree(ctx->V,&d));
  CHKERRQ(BVGetActiveColumns(pep->V,&lock,&nqt));
  lds = d*ld;
  offq = ld;
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldds));

  *breakdown = PETSC_FALSE; /* ----- */
  CHKERRQ(STGetShift(pep->st,&sigma));
  CHKERRQ(DSGetDimensions(pep->ds,NULL,&l,NULL,NULL));
  CHKERRQ(BVSetActiveColumns(ctx->V,0,m));
  CHKERRQ(BVSetActiveColumns(pep->V,0,nqt));
  for (j=k;j<m;j++) {
    /* apply operator */
    CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
    CHKERRQ(MatDenseGetArray(MS,&S));
    CHKERRQ(BVGetColumn(pep->V,nqt,&t));
    CHKERRQ(BVMultVec(pep->V,1.0,0.0,v,S+j*lds));
    CHKERRQ(MatMult(pep->A[1],v,q));
    CHKERRQ(MatMult(pep->A[2],v,t));
    CHKERRQ(VecAXPY(q,sigma*pep->sfactor,t));
    CHKERRQ(VecScale(q,pep->sfactor));
    CHKERRQ(BVMultVec(pep->V,1.0,0.0,v,S+offq+j*lds));
    CHKERRQ(MatMult(pep->A[2],v,t));
    CHKERRQ(VecAXPY(q,pep->sfactor*pep->sfactor,t));
    CHKERRQ(STMatSolve(pep->st,q,t));
    CHKERRQ(VecScale(t,-1.0));
    CHKERRQ(BVRestoreColumn(pep->V,nqt,&t));

    /* orthogonalize */
    CHKERRQ(BVOrthogonalizeColumn(pep->V,nqt,S+(j+1)*lds,&norm,&lindep));
    if (!lindep) {
      *(S+(j+1)*lds+nqt) = norm;
      CHKERRQ(BVScaleColumn(pep->V,nqt,1.0/norm));
      nqt++;
    }
    for (i=0;i<nqt;i++) *(S+(j+1)*lds+offq+i) = *(S+j*lds+i)+sigma*(*(S+(j+1)*lds+i));
    CHKERRQ(BVSetActiveColumns(pep->V,0,nqt));
    CHKERRQ(MatDenseRestoreArray(MS,&S));
    CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));

    /* level-2 orthogonalization */
    CHKERRQ(BVOrthogonalizeColumn(ctx->V,j+1,y,&norm,&lindep));
    a[j] = PetscRealPart(y[j]);
    omega[j+1] = (norm > 0)?1.0:-1.0;
    CHKERRQ(BVScaleColumn(ctx->V,j+1,1.0/norm));
    b[j] = PetscAbsReal(norm);

    /* check symmetry */
    CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_T,&f));
    if (j==k) {
      for (i=l;i<j-1;i++) y[i] = PetscAbsScalar(y[i])-PetscAbsReal(f[2*ldds+i]);
      for (i=0;i<l;i++) y[i] = 0.0;
    }
    CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_T,&f));
    if (j>0) y[j-1] = PetscAbsScalar(y[j-1])-PetscAbsReal(b[j-1]);
    CHKERRQ(PetscBLASIntCast(j,&j_));
    sym = SlepcAbs(BLASnrm2_(&j_,y,&one),sym);
    fro = SlepcAbs(fro,SlepcAbs(a[j],b[j]));
    if (j>0) fro = SlepcAbs(fro,b[j-1]);
    if (sym/fro>PetscMax(PETSC_SQRT_MACHINE_EPSILON,10*pep->tol)) {
      *symmlost = PETSC_TRUE;
      *M=j;
      break;
    }
  }
  CHKERRQ(BVSetActiveColumns(pep->V,lock,nqt));
  CHKERRQ(BVSetActiveColumns(ctx->V,0,*M));
  CHKERRQ(PetscFree(y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPSTOAR_QSlice(PEP pep,Mat B)
{
  PEP_STOAR      *ctx = (PEP_STOAR*)pep->data;
  PetscInt       j,k,l,nv=0,ld,ldds,t,nq=0,idx;
  PetscInt       nconv=0,deg=pep->nmat-1,count0=0,count1=0;
  PetscScalar    *om,sigma,*back,*S,*pQ;
  PetscReal      beta,norm=1.0,*omega,*a,*b,eta,lambda;
  PetscBool      breakdown,symmlost=PETSC_FALSE,sinv,falselock=PETSC_TRUE;
  Mat            MS,MQ;
  Vec            v,vomega;
  PEP_SR         sr;
  BVOrthogType   otype;
  BVOrthogBlockType obtype;

  PetscFunctionBegin;
  /* Resize if needed for deflating vectors  */
  sr = ctx->sr;
  sigma = sr->sPres->value;
  k = sr->ndef0+sr->ndef1;
  pep->ncv = ctx->ncv+k;
  pep->nev = ctx->nev+k;
  CHKERRQ(PEPAllocateSolution(pep,3));
  CHKERRQ(BVDestroy(&ctx->V));
  CHKERRQ(BVCreateTensor(pep->V,pep->nmat-1,&ctx->V));
  CHKERRQ(BVGetOrthogonalization(pep->V,&otype,NULL,&eta,&obtype));
  CHKERRQ(BVSetOrthogonalization(ctx->V,otype,BV_ORTHOG_REFINE_ALWAYS,eta,obtype));
  CHKERRQ(DSAllocate(pep->ds,pep->ncv+2));
  CHKERRQ(PetscMalloc1(pep->ncv,&back));
  CHKERRQ(DSGetLeadingDimension(pep->ds,&ldds));
  CHKERRQ(BVSetMatrix(ctx->V,B,PETSC_TRUE));
  PetscCheck(ctx->lock,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"A locking variant is needed for spectrum slicing");
  /* undocumented option to use a cheaper locking instead of the true locking */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pep_stoar_falselocking",&falselock,NULL));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv));
  CHKERRQ(RGPushScale(pep->rg,sinv?pep->sfactor:1.0/pep->sfactor));
  CHKERRQ(STScaleShift(pep->st,sinv?pep->sfactor:1.0/pep->sfactor));

  /* Get the starting Arnoldi vector */
  CHKERRQ(BVSetActiveColumns(pep->V,0,1));
  CHKERRQ(BVTensorBuildFirstColumn(ctx->V,pep->nini));
  CHKERRQ(BVSetActiveColumns(ctx->V,0,1));
  if (k) {
    /* Insert deflated vectors */
    CHKERRQ(BVSetActiveColumns(pep->V,0,0));
    idx = sr->ndef0?sr->idxDef0[0]:sr->idxDef1[0];
    for (j=0;j<k;j++) {
      CHKERRQ(BVGetColumn(pep->V,j,&v));
      CHKERRQ(BVCopyVec(sr->V,sr->qinfo[idx].q[j],v));
      CHKERRQ(BVRestoreColumn(pep->V,j,&v));
    }
    /* Update innerproduct matrix */
    CHKERRQ(BVSetActiveColumns(ctx->V,0,0));
    CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
    CHKERRQ(BVSetActiveColumns(pep->V,0,k));
    CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));

    CHKERRQ(BVGetSizes(pep->V,NULL,NULL,&ld));
    CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
    CHKERRQ(MatDenseGetArray(MS,&S));
    for (j=0;j<sr->ndef0;j++) {
      CHKERRQ(PetscArrayzero(S+j*ld*deg,ld*deg));
      CHKERRQ(PetscArraycpy(S+j*ld*deg,sr->S+sr->idxDef0[j]*sr->ld*deg,k));
      CHKERRQ(PetscArraycpy(S+j*ld*deg+ld,sr->S+sr->idxDef0[j]*sr->ld*deg+sr->ld,k));
      pep->eigr[j] = sr->eigr[sr->idxDef0[j]];
      pep->errest[j] = sr->errest[sr->idxDef0[j]];
    }
    for (j=0;j<sr->ndef1;j++) {
      CHKERRQ(PetscArrayzero(S+(j+sr->ndef0)*ld*deg,ld*deg));
      CHKERRQ(PetscArraycpy(S+(j+sr->ndef0)*ld*deg,sr->S+sr->idxDef1[j]*sr->ld*deg,k));
      CHKERRQ(PetscArraycpy(S+(j+sr->ndef0)*ld*deg+ld,sr->S+sr->idxDef1[j]*sr->ld*deg+sr->ld,k));
      pep->eigr[j+sr->ndef0] = sr->eigr[sr->idxDef1[j]];
      pep->errest[j+sr->ndef0] = sr->errest[sr->idxDef1[j]];
    }
    CHKERRQ(MatDenseRestoreArray(MS,&S));
    CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    CHKERRQ(BVSetActiveColumns(ctx->V,0,k+1));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+1,&vomega));
    CHKERRQ(VecGetArray(vomega,&om));
    for (j=0;j<k;j++) {
      CHKERRQ(BVOrthogonalizeColumn(ctx->V,j,NULL,&norm,NULL));
      CHKERRQ(BVScaleColumn(ctx->V,j,1/norm));
      om[j] = (norm>=0.0)?1.0:-1.0;
    }
    CHKERRQ(BVTensorGetFactors(ctx->V,NULL,&MS));
    CHKERRQ(MatDenseGetArray(MS,&S));
    for (j=0;j<deg;j++) {
      CHKERRQ(BVSetRandomColumn(pep->V,k+j));
      CHKERRQ(BVOrthogonalizeColumn(pep->V,k+j,S+k*ld*deg+j*ld,&norm,NULL));
      CHKERRQ(BVScaleColumn(pep->V,k+j,1.0/norm));
      S[k*ld*deg+j*ld+k+j] = norm;
    }
    CHKERRQ(MatDenseRestoreArray(MS,&S));
    CHKERRQ(BVSetActiveColumns(pep->V,0,k+deg));
    CHKERRQ(BVTensorRestoreFactors(ctx->V,NULL,&MS));
    CHKERRQ(BVOrthogonalizeColumn(ctx->V,k,NULL,&norm,NULL));
    CHKERRQ(BVScaleColumn(ctx->V,k,1.0/norm));
    om[k] = (norm>=0.0)?1.0:-1.0;
    CHKERRQ(VecRestoreArray(vomega,&om));
    CHKERRQ(BVSetSignature(ctx->V,vomega));
    CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_T,&a));
    CHKERRQ(VecGetArray(vomega,&om));
    for (j=0;j<k;j++) a[j] = PetscRealPart(om[j]/(pep->eigr[j]-sigma));
    CHKERRQ(VecRestoreArray(vomega,&om));
    CHKERRQ(VecDestroy(&vomega));
    CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_T,&a));
    CHKERRQ(DSGetArray(pep->ds,DS_MAT_Q,&pQ));
    CHKERRQ(PetscArrayzero(pQ,ldds*k));
    for (j=0;j<k;j++) pQ[j+j*ldds] = 1.0;
    CHKERRQ(DSRestoreArray(pep->ds,DS_MAT_Q,&pQ));
  }
  CHKERRQ(BVSetActiveColumns(ctx->V,0,k+1));
  CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_D,&omega));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+1,&vomega));
  CHKERRQ(BVGetSignature(ctx->V,vomega));
  CHKERRQ(VecGetArray(vomega,&om));
  for (j=0;j<k+1;j++) omega[j] = PetscRealPart(om[j]);
  CHKERRQ(VecRestoreArray(vomega,&om));
  CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega));
  CHKERRQ(VecDestroy(&vomega));

  CHKERRQ(PetscInfo(pep,"Start STOAR: sigma=%g in [%g,%g], for deflation: left=%" PetscInt_FMT " right=%" PetscInt_FMT ", searching: left=%" PetscInt_FMT " right=%" PetscInt_FMT "\n",(double)sr->sPres->value,(double)(sr->sPres->neighb[0]?sr->sPres->neighb[0]->value:sr->int0),(double)(sr->sPres->neighb[1]?sr->sPres->neighb[1]->value:sr->int1),sr->ndef0,sr->ndef1,sr->sPres->nsch[0],sr->sPres->nsch[1]));

  /* Restart loop */
  l = 0;
  pep->nconv = k;
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;
    CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_T,&a));
    b = a+ldds;
    CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_D,&omega));

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(pep->nconv+pep->mpd,pep->ncv);
    CHKERRQ(PEPSTOARrun_QSlice(pep,a,b,omega,pep->nconv+l,&nv,&breakdown,&symmlost,pep->work));
    beta = b[nv-1];
    if (symmlost && nv==pep->nconv+l) {
      pep->reason = PEP_DIVERGED_SYMMETRY_LOST;
      pep->nconv = nconv;
      CHKERRQ(PetscInfo(pep,"Symmetry lost in STOAR sigma=%g nconv=%" PetscInt_FMT "\n",(double)sr->sPres->value,nconv));
      if (falselock || !ctx->lock) {
        CHKERRQ(BVSetActiveColumns(ctx->V,0,pep->nconv));
        CHKERRQ(BVTensorCompress(ctx->V,0));
      }
      break;
    }
    CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_T,&a));
    CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega));
    CHKERRQ(DSSetDimensions(pep->ds,nv,pep->nconv,pep->nconv+l));
    if (l==0) CHKERRQ(DSSetState(pep->ds,DS_STATE_INTERMEDIATE));
    else CHKERRQ(DSSetState(pep->ds,DS_STATE_RAW));

    /* Solve projected problem */
    CHKERRQ(DSSolve(pep->ds,pep->eigr,pep->eigi));
    CHKERRQ(DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(pep->ds));
    CHKERRQ(DSSynchronize(pep->ds,pep->eigr,pep->eigi));

    /* Check convergence */
    /* CHKERRQ(PEPSTOARpreKConvergence(pep,nv,&norm,pep->work));*/
    norm = 1.0;
    CHKERRQ(DSGetDimensions(pep->ds,NULL,NULL,NULL,&t));
    CHKERRQ(PEPKrylovConvergence(pep,PETSC_FALSE,pep->nconv,t-pep->nconv,PetscAbsReal(beta)*norm,&k));
    CHKERRQ((*pep->stopping)(pep,pep->its,pep->max_it,k,pep->nev,&pep->reason,pep->stoppingctx));
    for (j=0;j<k;j++) back[j] = pep->eigr[j];
    CHKERRQ(STBackTransform(pep->st,k,back,pep->eigi));
    count0=count1=0;
    for (j=0;j<k;j++) {
      lambda = PetscRealPart(back[j]);
      if (((sr->dir)*(sr->sPres->value - lambda) > 0) && ((sr->dir)*(lambda - sr->sPres->ext[0]) > 0)) count0++;
      if (((sr->dir)*(lambda - sr->sPres->value) > 0) && ((sr->dir)*(sr->sPres->ext[1] - lambda) > 0)) count1++;
    }
    if ((count0-sr->ndef0 >= sr->sPres->nsch[0]) && (count1-sr->ndef1 >= sr->sPres->nsch[1])) pep->reason = PEP_CONVERGED_TOL;
    /* Update l */
    if (pep->reason != PEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      l = PetscMin(l,t);
      CHKERRQ(DSGetTruncateSize(pep->ds,k,t,&l));
      if (!breakdown) {
        /* Prepare the Rayleigh quotient for restart */
        CHKERRQ(DSTruncate(pep->ds,k+l,PETSC_FALSE));
      }
    }
    nconv = k;
    if (!ctx->lock && pep->reason == PEP_CONVERGED_ITERATING && !breakdown) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */
    if (l) CHKERRQ(PetscInfo(pep,"Preparing to restart keeping l=%" PetscInt_FMT " vectors\n",l));

    /* Update S */
    CHKERRQ(DSGetMat(pep->ds,DS_MAT_Q,&MQ));
    CHKERRQ(BVMultInPlace(ctx->V,MQ,pep->nconv,k+l));
    CHKERRQ(MatDestroy(&MQ));

    /* Copy last column of S */
    CHKERRQ(BVCopyColumn(ctx->V,nv,k+l));
    CHKERRQ(DSGetArrayReal(pep->ds,DS_MAT_D,&omega));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,k+l,&vomega));
    CHKERRQ(VecGetArray(vomega,&om));
    for (j=0;j<k+l;j++) om[j] = omega[j];
    CHKERRQ(VecRestoreArray(vomega,&om));
    CHKERRQ(BVSetActiveColumns(ctx->V,0,k+l));
    CHKERRQ(BVSetSignature(ctx->V,vomega));
    CHKERRQ(VecDestroy(&vomega));
    CHKERRQ(DSRestoreArrayReal(pep->ds,DS_MAT_D,&omega));

    if (breakdown && pep->reason == PEP_CONVERGED_ITERATING) {
      /* stop if breakdown */
      CHKERRQ(PetscInfo(pep,"Breakdown TOAR method (it=%" PetscInt_FMT " norm=%g)\n",pep->its,(double)beta));
      pep->reason = PEP_DIVERGED_BREAKDOWN;
    }
    if (pep->reason != PEP_CONVERGED_ITERATING) l--;
    CHKERRQ(BVGetActiveColumns(pep->V,NULL,&nq));
    if (k+l+deg<=nq) {
      CHKERRQ(BVSetActiveColumns(ctx->V,pep->nconv,k+l+1));
      if (!falselock && ctx->lock) CHKERRQ(BVTensorCompress(ctx->V,k-pep->nconv));
      else CHKERRQ(BVTensorCompress(ctx->V,0));
    }
    pep->nconv = k;
    CHKERRQ(PEPMonitor(pep,pep->its,nconv,pep->eigr,pep->eigi,pep->errest,nv));
  }
  sr->itsKs += pep->its;
  if (pep->nconv>0) {
    CHKERRQ(BVSetActiveColumns(ctx->V,0,pep->nconv));
    CHKERRQ(BVGetActiveColumns(pep->V,NULL,&nq));
    CHKERRQ(BVSetActiveColumns(pep->V,0,nq));
    if (nq>pep->nconv) {
      CHKERRQ(BVTensorCompress(ctx->V,pep->nconv));
      CHKERRQ(BVSetActiveColumns(pep->V,0,pep->nconv));
    }
    for (j=0;j<pep->nconv;j++) {
      pep->eigr[j] *= pep->sfactor;
      pep->eigi[j] *= pep->sfactor;
    }
  }
  CHKERRQ(PetscInfo(pep,"Finished STOAR: nconv=%" PetscInt_FMT " (deflated=%" PetscInt_FMT ", left=%" PetscInt_FMT ", right=%" PetscInt_FMT ")\n",pep->nconv,sr->ndef0+sr->ndef1,count0-sr->ndef0,count1-sr->ndef1));
  CHKERRQ(STScaleShift(pep->st,sinv?1.0/pep->sfactor:pep->sfactor));
  CHKERRQ(RGPopScale(pep->rg));

  PetscCheck(pep->reason!=PEP_DIVERGED_SYMMETRY_LOST || nconv>=sr->ndef0+sr->ndef1,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Symmetry lost at sigma=%g",(double)sr->sPres->value);
  if (pep->reason == PEP_DIVERGED_SYMMETRY_LOST && nconv==sr->ndef0+sr->ndef1) {
    PetscCheck(++sr->symmlost<=10,PetscObjectComm((PetscObject)pep),PETSC_ERR_PLIB,"Symmetry lost at sigma=%g",(double)sr->sPres->value);
  } else sr->symmlost = 0;

  CHKERRQ(DSTruncate(pep->ds,pep->nconv,PETSC_TRUE));
  CHKERRQ(PetscFree(back));
  PetscFunctionReturn(0);
}

#define SWAP(a,b,t) {t=a;a=b;b=t;}

static PetscErrorCode PEPQSliceGetInertias(PEP pep,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR          sr=ctx->sr;
  PetscInt        i=0,j,tmpi;
  PetscReal       v,tmpr;
  PEP_shift       s;

  PetscFunctionBegin;
  PetscCheck(pep->state,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONGSTATE,"Must call PEPSetUp() first");
  PetscCheck(sr,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see PEPSetInterval()");
  if (!sr->s0) {  /* PEPSolve not called yet */
    *n = 2;
  } else {
    *n = 1;
    s = sr->s0;
    while (s) {
      (*n)++;
      s = s->neighb[1];
    }
  }
  CHKERRQ(PetscMalloc1(*n,shifts));
  CHKERRQ(PetscMalloc1(*n,inertias));
  if (!sr->s0) {  /* PEPSolve not called yet */
    (*shifts)[0]   = sr->int0;
    (*shifts)[1]   = sr->int1;
    (*inertias)[0] = sr->inertia0;
    (*inertias)[1] = sr->inertia1;
  } else {
    s = sr->s0;
    while (s) {
      (*shifts)[i]     = s->value;
      (*inertias)[i++] = s->inertia;
      s = s->neighb[1];
    }
    (*shifts)[i]   = sr->int1;
    (*inertias)[i] = sr->inertia1;
  }
  /* remove possible duplicate in last position */
  if ((*shifts)[(*n)-1]==(*shifts)[(*n)-2]) (*n)--;
  /* sort result */
  for (i=0;i<*n;i++) {
    v = (*shifts)[i];
    for (j=i+1;j<*n;j++) {
      if (v > (*shifts)[j]) {
        SWAP((*shifts)[i],(*shifts)[j],tmpr);
        SWAP((*inertias)[i],(*inertias)[j],tmpi);
        v = (*shifts)[i];
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_STOAR_QSlice(PEP pep)
{
  PetscInt       i,j,ti,deg=pep->nmat-1;
  PetscReal      newS;
  PEP_STOAR      *ctx=(PEP_STOAR*)pep->data;
  PEP_SR         sr=ctx->sr;
  Mat            S,B;
  PetscScalar    *pS;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));

  /* Only with eigenvalues present in the interval ...*/
  if (sr->numEigs==0) {
    pep->reason = PEP_CONVERGED_TOL;
    PetscFunctionReturn(0);
  }

  /* Inner product matrix */
  CHKERRQ(PEPSTOARSetUpInnerMatrix(pep,&B));

  /* Array of pending shifts */
  sr->maxPend = 100; /* Initial size */
  sr->nPend = 0;
  CHKERRQ(PetscMalloc1(sr->maxPend,&sr->pending));
  CHKERRQ(PetscLogObjectMemory((PetscObject)pep,sr->maxPend*sizeof(PEP_shift*)));
  CHKERRQ(PEPCreateShift(pep,sr->int0,NULL,NULL));
  /* extract first shift */
  sr->sPrev = NULL;
  sr->sPres = sr->pending[--sr->nPend];
  sr->sPres->inertia = sr->inertia0;
  pep->target = sr->sPres->value;
  sr->s0 = sr->sPres;
  sr->indexEig = 0;

  for (i=0;i<sr->numEigs;i++) {
    sr->eigr[i]   = 0.0;
    sr->eigi[i]   = 0.0;
    sr->errest[i] = 0.0;
    sr->perm[i]   = i;
  }
  /* Vectors for deflation */
  CHKERRQ(PetscMalloc2(sr->numEigs,&sr->idxDef0,sr->numEigs,&sr->idxDef1));
  CHKERRQ(PetscLogObjectMemory((PetscObject)pep,2*sr->numEigs*sizeof(PetscInt)));
  sr->indexEig = 0;
  while (sr->sPres) {
    /* Search for deflation */
    CHKERRQ(PEPLookForDeflation(pep));
    /* KrylovSchur */
    CHKERRQ(PEPSTOAR_QSlice(pep,B));

    CHKERRQ(PEPStoreEigenpairs(pep));
    /* Select new shift */
    if (!sr->sPres->comp[1]) {
      CHKERRQ(PEPGetNewShiftValue(pep,1,&newS));
      CHKERRQ(PEPCreateShift(pep,newS,sr->sPres,sr->sPres->neighb[1]));
    }
    if (!sr->sPres->comp[0]) {
      /* Completing earlier interval */
      CHKERRQ(PEPGetNewShiftValue(pep,0,&newS));
      CHKERRQ(PEPCreateShift(pep,newS,sr->sPres->neighb[0],sr->sPres));
    }
    /* Preparing for a new search of values */
    CHKERRQ(PEPExtractShift(pep));
  }

  /* Updating pep values prior to exit */
  CHKERRQ(PetscFree2(sr->idxDef0,sr->idxDef1));
  CHKERRQ(PetscFree(sr->pending));
  pep->nconv  = sr->indexEig;
  pep->reason = PEP_CONVERGED_TOL;
  pep->its    = sr->itsKs;
  pep->nev    = sr->indexEig;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,pep->nconv,pep->nconv,NULL,&S));
  CHKERRQ(MatDenseGetArray(S,&pS));
  for (i=0;i<pep->nconv;i++) {
    for (j=0;j<sr->qinfo[i].nq;j++) pS[i*pep->nconv+sr->qinfo[i].q[j]] = *(sr->S+i*sr->ld*deg+j);
  }
  CHKERRQ(MatDenseRestoreArray(S,&pS));
  CHKERRQ(BVSetActiveColumns(sr->V,0,pep->nconv));
  CHKERRQ(BVMultInPlace(sr->V,S,0,pep->nconv));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(BVDestroy(&pep->V));
  pep->V = sr->V;
  CHKERRQ(PetscFree4(pep->eigr,pep->eigi,pep->errest,pep->perm));
  pep->eigr   = sr->eigr;
  pep->eigi   = sr->eigi;
  pep->perm   = sr->perm;
  pep->errest = sr->errest;
  if (sr->dir<0) {
    for (i=0;i<pep->nconv/2;i++) {
      ti = sr->perm[i]; sr->perm[i] = sr->perm[pep->nconv-1-i]; sr->perm[pep->nconv-1-i] = ti;
    }
  }
  CHKERRQ(PetscFree(ctx->inertias));
  CHKERRQ(PetscFree(ctx->shifts));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PEPQSliceGetInertias(pep,&ctx->nshifts,&ctx->shifts,&ctx->inertias));
  PetscFunctionReturn(0);
}
