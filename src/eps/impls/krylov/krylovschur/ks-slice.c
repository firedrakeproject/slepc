/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with spectrum slicing for symmetric eigenproblems

   References:

       [1] R.G. Grimes et al., "A shifted block Lanczos algorithm for
           solving sparse symmetric generalized eigenproblems", SIAM J.
           Matrix Anal. Appl. 15(1):228-272, 1994.

       [2] C. Campos and J.E. Roman, "Spectrum slicing strategies based
           on restarted Lanczos methods", Numer. Algor. 60(2):279-295,
           2012.
*/

#include <slepc/private/epsimpl.h>
#include "krylovschur.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@Article{slepc-slice,\n"
  "   author = \"C. Campos and J. E. Roman\",\n"
  "   title = \"Strategies for spectrum slicing based on restarted {Lanczos} methods\",\n"
  "   journal = \"Numer. Algorithms\",\n"
  "   volume = \"60\",\n"
  "   number = \"2\",\n"
  "   pages = \"279--295\",\n"
  "   year = \"2012,\"\n"
  "   doi = \"https://doi.org/10.1007/s11075-012-9564-z\"\n"
  "}\n";

#define SLICE_PTOL PETSC_SQRT_MACHINE_EPSILON

static PetscErrorCode EPSSliceResetSR(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR          sr=ctx->sr;
  EPS_shift       s;

  PetscFunctionBegin;
  if (sr) {
    if (ctx->npart>1) {
      CHKERRQ(BVDestroy(&sr->V));
      CHKERRQ(PetscFree4(sr->eigr,sr->eigi,sr->errest,sr->perm));
    }
    /* Reviewing list of shifts to free memory */
    s = sr->s0;
    if (s) {
      while (s->neighb[1]) {
        s = s->neighb[1];
        CHKERRQ(PetscFree(s->neighb[0]));
      }
      CHKERRQ(PetscFree(s));
    }
    CHKERRQ(PetscFree(sr));
  }
  ctx->sr = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_KrylovSchur_Slice(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (!ctx->global) PetscFunctionReturn(0);
  /* Reset auxiliary EPS */
  CHKERRQ(EPSSliceResetSR(ctx->eps));
  CHKERRQ(EPSReset(ctx->eps));
  CHKERRQ(EPSSliceResetSR(eps));
  CHKERRQ(PetscFree(ctx->inertias));
  CHKERRQ(PetscFree(ctx->shifts));
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_KrylovSchur_Slice(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (!ctx->global) PetscFunctionReturn(0);
  /* Destroy auxiliary EPS */
  CHKERRQ(EPSReset_KrylovSchur_Slice(eps));
  CHKERRQ(EPSDestroy(&ctx->eps));
  if (ctx->npart>1) {
    CHKERRQ(PetscSubcommDestroy(&ctx->subc));
    if (ctx->commset) {
      CHKERRMPI(MPI_Comm_free(&ctx->commrank));
      ctx->commset = PETSC_FALSE;
    }
    CHKERRQ(ISDestroy(&ctx->isrow));
    CHKERRQ(ISDestroy(&ctx->iscol));
    CHKERRQ(MatDestroyMatrices(1,&ctx->submata));
    CHKERRQ(MatDestroyMatrices(1,&ctx->submatb));
  }
  CHKERRQ(PetscFree(ctx->subintervals));
  CHKERRQ(PetscFree(ctx->nconv_loc));
  PetscFunctionReturn(0);
}

/*
  EPSSliceAllocateSolution - Allocate memory storage for common variables such
  as eigenvalues and eigenvectors. The argument extra is used for methods
  that require a working basis slightly larger than ncv.
*/
static PetscErrorCode EPSSliceAllocateSolution(EPS eps,PetscInt extra)
{
  EPS_KRYLOVSCHUR    *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscReal          eta;
  PetscInt           k;
  PetscLogDouble     cnt;
  BVType             type;
  BVOrthogType       orthog_type;
  BVOrthogRefineType orthog_ref;
  BVOrthogBlockType  ob_type;
  Mat                matrix;
  Vec                t;
  EPS_SR             sr = ctx->sr;

  PetscFunctionBegin;
  /* allocate space for eigenvalues and friends */
  k = PetscMax(1,sr->numEigs);
  CHKERRQ(PetscFree4(sr->eigr,sr->eigi,sr->errest,sr->perm));
  CHKERRQ(PetscMalloc4(k,&sr->eigr,k,&sr->eigi,k,&sr->errest,k,&sr->perm));
  cnt = 2*k*sizeof(PetscScalar) + 2*k*sizeof(PetscReal) + k*sizeof(PetscInt);
  CHKERRQ(PetscLogObjectMemory((PetscObject)eps,cnt));

  /* allocate sr->V and transfer options from eps->V */
  CHKERRQ(BVDestroy(&sr->V));
  CHKERRQ(BVCreate(PetscObjectComm((PetscObject)eps),&sr->V));
  CHKERRQ(PetscLogObjectParent((PetscObject)eps,(PetscObject)sr->V));
  if (!eps->V) CHKERRQ(EPSGetBV(eps,&eps->V));
  if (!((PetscObject)(eps->V))->type_name) {
    CHKERRQ(BVSetType(sr->V,BVSVEC));
  } else {
    CHKERRQ(BVGetType(eps->V,&type));
    CHKERRQ(BVSetType(sr->V,type));
  }
  CHKERRQ(STMatCreateVecsEmpty(eps->st,&t,NULL));
  CHKERRQ(BVSetSizesFromVec(sr->V,t,k));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(EPS_SetInnerProduct(eps));
  CHKERRQ(BVGetMatrix(eps->V,&matrix,NULL));
  CHKERRQ(BVSetMatrix(sr->V,matrix,PETSC_FALSE));
  CHKERRQ(BVGetOrthogonalization(eps->V,&orthog_type,&orthog_ref,&eta,&ob_type));
  CHKERRQ(BVSetOrthogonalization(sr->V,orthog_type,orthog_ref,eta,ob_type));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSSliceGetEPS(EPS eps)
{
  EPS_KRYLOVSCHUR    *ctx=(EPS_KRYLOVSCHUR*)eps->data,*ctx_local;
  BV                 V;
  BVType             type;
  PetscReal          eta;
  BVOrthogType       orthog_type;
  BVOrthogRefineType orthog_ref;
  BVOrthogBlockType  ob_type;
  PetscInt           i;
  PetscReal          h,a,b;
  PetscRandom        rand;
  EPS_SR             sr=ctx->sr;

  PetscFunctionBegin;
  if (!ctx->eps) CHKERRQ(EPSKrylovSchurGetChildEPS(eps,&ctx->eps));

  /* Determine subintervals */
  if (ctx->npart==1) {
    a = eps->inta; b = eps->intb;
  } else {
    if (!ctx->subintset) { /* uniform distribution if no set by user */
      PetscCheck(sr->hasEnd,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Global interval must be bounded for splitting it in uniform subintervals");
      h = (eps->intb-eps->inta)/ctx->npart;
      a = eps->inta+ctx->subc->color*h;
      b = (ctx->subc->color==ctx->npart-1)?eps->intb:eps->inta+(ctx->subc->color+1)*h;
      CHKERRQ(PetscFree(ctx->subintervals));
      CHKERRQ(PetscMalloc1(ctx->npart+1,&ctx->subintervals));
      for (i=0;i<ctx->npart;i++) ctx->subintervals[i] = eps->inta+h*i;
      ctx->subintervals[ctx->npart] = eps->intb;
    } else {
      a = ctx->subintervals[ctx->subc->color];
      b = ctx->subintervals[ctx->subc->color+1];
    }
  }
  CHKERRQ(EPSSetInterval(ctx->eps,a,b));
  CHKERRQ(EPSSetConvergenceTest(ctx->eps,eps->conv));
  CHKERRQ(EPSSetDimensions(ctx->eps,ctx->nev,ctx->ncv,ctx->mpd));
  CHKERRQ(EPSKrylovSchurSetLocking(ctx->eps,ctx->lock));

  ctx_local = (EPS_KRYLOVSCHUR*)ctx->eps->data;
  ctx_local->detect = ctx->detect;

  /* transfer options from eps->V */
  CHKERRQ(EPSGetBV(ctx->eps,&V));
  CHKERRQ(BVGetRandomContext(V,&rand));  /* make sure the random context is available when duplicating */
  if (!eps->V) CHKERRQ(EPSGetBV(eps,&eps->V));
  if (!((PetscObject)(eps->V))->type_name) {
    CHKERRQ(BVSetType(V,BVSVEC));
  } else {
    CHKERRQ(BVGetType(eps->V,&type));
    CHKERRQ(BVSetType(V,type));
  }
  CHKERRQ(BVGetOrthogonalization(eps->V,&orthog_type,&orthog_ref,&eta,&ob_type));
  CHKERRQ(BVSetOrthogonalization(V,orthog_type,orthog_ref,eta,ob_type));

  ctx->eps->which = eps->which;
  ctx->eps->max_it = eps->max_it;
  ctx->eps->tol = eps->tol;
  ctx->eps->purify = eps->purify;
  if (eps->tol==PETSC_DEFAULT) eps->tol = SLEPC_DEFAULT_TOL;
  CHKERRQ(EPSSetProblemType(ctx->eps,eps->problem_type));
  CHKERRQ(EPSSetUp(ctx->eps));
  ctx->eps->nconv = 0;
  ctx->eps->its   = 0;
  for (i=0;i<ctx->eps->ncv;i++) {
    ctx->eps->eigr[i]   = 0.0;
    ctx->eps->eigi[i]   = 0.0;
    ctx->eps->errest[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSSliceGetInertia(EPS eps,PetscReal shift,PetscInt *inertia,PetscInt *zeros)
{
  KSP            ksp,kspr;
  PC             pc;
  Mat            F;
  PetscReal      nzshift=shift;
  PetscBool      flg;

  PetscFunctionBegin;
  if (shift >= PETSC_MAX_REAL) { /* Right-open interval */
    if (inertia) *inertia = eps->n;
  } else if (shift <= PETSC_MIN_REAL) {
    if (inertia) *inertia = 0;
    if (zeros) *zeros = 0;
  } else {
    /* If the shift is zero, perturb it to a very small positive value.
       The goal is that the nonzero pattern is the same in all cases and reuse
       the symbolic factorizations */
    nzshift = (shift==0.0)? 10.0/PETSC_MAX_REAL: shift;
    CHKERRQ(STSetShift(eps->st,nzshift));
    CHKERRQ(STGetKSP(eps->st,&ksp));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCREDUNDANT,&flg));
    if (flg) {
      CHKERRQ(PCRedundantGetKSP(pc,&kspr));
      CHKERRQ(KSPGetPC(kspr,&pc));
    }
    CHKERRQ(PCFactorGetMatrix(pc,&F));
    CHKERRQ(MatGetInertia(F,inertia,zeros,NULL));
  }
  if (inertia) CHKERRQ(PetscInfo(eps,"Computed inertia at shift %g: %" PetscInt_FMT "\n",(double)nzshift,*inertia));
  PetscFunctionReturn(0);
}

/*
   Dummy backtransform operation
 */
static PetscErrorCode EPSBackTransform_Skip(EPS eps)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetUp_KrylovSchur_Slice(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data,*ctx_glob;
  EPS_SR          sr,sr_loc,sr_glob;
  PetscInt        nEigs,dssz=1,i,zeros=0,off=0,method,hiteig=0;
  PetscMPIInt     nproc,rank=0,aux;
  PetscReal       r;
  MPI_Request     req;
  Mat             A,B=NULL;
  DSParallelType  ptype;
  MPI_Comm        child;

  PetscFunctionBegin;
  if (ctx->global) {
    EPSCheckHermitianDefiniteCondition(eps,PETSC_TRUE," with spectrum slicing");
    EPSCheckSinvertCayleyCondition(eps,PETSC_TRUE," with spectrum slicing");
    PetscCheck(eps->inta!=eps->intb,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues unless you provide a computational interval with EPSSetInterval()");
    PetscCheck(eps->intb<PETSC_MAX_REAL || eps->inta>PETSC_MIN_REAL,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"The defined computational interval should have at least one of their sides bounded");
    PetscCheck(eps->nds==0,PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Spectrum slicing not supported in combination with deflation space");
    EPSCheckUnsupportedCondition(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING,PETSC_TRUE," with spectrum slicing");
    EPSCheckIgnoredCondition(eps,EPS_FEATURE_BALANCE,PETSC_TRUE," with spectrum slicing");
    if (eps->tol==PETSC_DEFAULT) {
 #if defined(PETSC_USE_REAL_SINGLE)
      eps->tol = SLEPC_DEFAULT_TOL;
#else
      /* use tighter tolerance */
      eps->tol = SLEPC_DEFAULT_TOL*1e-2;
#endif
    }
    if (eps->max_it==PETSC_DEFAULT) eps->max_it = 100;
    if (ctx->nev==1) ctx->nev = PetscMin(40,eps->n);  /* nev not set, use default value */
    PetscCheck(eps->n<=10 || ctx->nev>=10,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"nev cannot be less than 10 in spectrum slicing runs");
  }
  eps->ops->backtransform = EPSBackTransform_Skip;

  /* create spectrum slicing context and initialize it */
  CHKERRQ(EPSSliceResetSR(eps));
  CHKERRQ(PetscNewLog(eps,&sr));
  ctx->sr = sr;
  sr->itsKs = 0;
  sr->nleap = 0;
  sr->nMAXCompl = eps->nev/4;
  sr->iterCompl = eps->max_it/4;
  sr->sPres = NULL;
  sr->nS = 0;

  if (ctx->npart==1 || ctx->global) {
    /* check presence of ends and finding direction */
    if ((eps->inta > PETSC_MIN_REAL && !(ctx->subintervals && ctx->subintervals[0]==ctx->subintervals[1])) || eps->intb >= PETSC_MAX_REAL) {
      sr->int0 = eps->inta;
      sr->int1 = eps->intb;
      sr->dir = 1;
      if (eps->intb >= PETSC_MAX_REAL) { /* Right-open interval */
        sr->hasEnd = PETSC_FALSE;
      } else sr->hasEnd = PETSC_TRUE;
    } else {
      sr->int0 = eps->intb;
      sr->int1 = eps->inta;
      sr->dir = -1;
      sr->hasEnd = PetscNot(eps->inta <= PETSC_MIN_REAL);
    }
  }
  if (ctx->global) {
    CHKERRQ(EPSSetDimensions_Default(eps,ctx->nev,&ctx->ncv,&ctx->mpd));
    /* create subintervals and initialize auxiliary eps for slicing runs */
    CHKERRQ(EPSKrylovSchurGetChildEPS(eps,&ctx->eps));
    /* prevent computation of factorization in global eps */
    CHKERRQ(STSetTransform(eps->st,PETSC_FALSE));
    CHKERRQ(EPSSliceGetEPS(eps));
    sr_loc = ((EPS_KRYLOVSCHUR*)ctx->eps->data)->sr;
    if (ctx->npart>1) {
      CHKERRQ(PetscSubcommGetChild(ctx->subc,&child));
      if ((sr->dir>0&&ctx->subc->color==0)||(sr->dir<0&&ctx->subc->color==ctx->npart-1)) sr->inertia0 = sr_loc->inertia0;
      CHKERRMPI(MPI_Comm_rank(child,&rank));
      if (!rank) {
        CHKERRMPI(MPI_Bcast(&sr->inertia0,1,MPIU_INT,(sr->dir>0)?0:ctx->npart-1,ctx->commrank));
      }
      CHKERRMPI(MPI_Bcast(&sr->inertia0,1,MPIU_INT,0,child));
      CHKERRQ(PetscFree(ctx->nconv_loc));
      CHKERRQ(PetscMalloc1(ctx->npart,&ctx->nconv_loc));
      CHKERRMPI(MPI_Comm_size(((PetscObject)eps)->comm,&nproc));
      if (sr->dir<0) off = 1;
      if (nproc%ctx->npart==0) { /* subcommunicators with the same size */
        CHKERRQ(PetscMPIIntCast(sr_loc->numEigs,&aux));
        CHKERRMPI(MPI_Allgather(&aux,1,MPI_INT,ctx->nconv_loc,1,MPI_INT,ctx->commrank));
        CHKERRMPI(MPI_Allgather(sr_loc->dir==sr->dir?&sr_loc->int0:&sr_loc->int1,1,MPIU_REAL,ctx->subintervals+off,1,MPIU_REAL,ctx->commrank));
      } else {
        CHKERRMPI(MPI_Comm_rank(child,&rank));
        if (!rank) {
          CHKERRQ(PetscMPIIntCast(sr_loc->numEigs,&aux));
          CHKERRMPI(MPI_Allgather(&aux,1,MPI_INT,ctx->nconv_loc,1,MPI_INT,ctx->commrank));
          CHKERRMPI(MPI_Allgather(sr_loc->dir==sr->dir?&sr_loc->int0:&sr_loc->int1,1,MPIU_REAL,ctx->subintervals+off,1,MPIU_REAL,ctx->commrank));
        }
        CHKERRQ(PetscMPIIntCast(ctx->npart,&aux));
        CHKERRMPI(MPI_Bcast(ctx->nconv_loc,aux,MPI_INT,0,child));
        CHKERRMPI(MPI_Bcast(ctx->subintervals+off,aux,MPIU_REAL,0,child));
      }
      nEigs = 0;
      for (i=0;i<ctx->npart;i++) nEigs += ctx->nconv_loc[i];
    } else {
      nEigs = sr_loc->numEigs;
      sr->inertia0 = sr_loc->inertia0;
      sr->dir = sr_loc->dir;
    }
    sr->inertia1 = sr->inertia0+sr->dir*nEigs;
    sr->numEigs = nEigs;
    eps->nev = nEigs;
    eps->ncv = nEigs;
    eps->mpd = nEigs;
  } else {
    ctx_glob = (EPS_KRYLOVSCHUR*)ctx->eps->data;
    sr_glob = ctx_glob->sr;
    if (ctx->npart>1) {
      sr->dir = sr_glob->dir;
      sr->int0 = (sr->dir==1)?eps->inta:eps->intb;
      sr->int1 = (sr->dir==1)?eps->intb:eps->inta;
      if ((sr->dir>0&&ctx->subc->color==ctx->npart-1)||(sr->dir<0&&ctx->subc->color==0)) sr->hasEnd = sr_glob->hasEnd;
      else sr->hasEnd = PETSC_TRUE;
    }
    /* sets first shift */
    CHKERRQ(STSetShift(eps->st,(sr->int0==0.0)?10.0/PETSC_MAX_REAL:sr->int0));
    CHKERRQ(STSetUp(eps->st));

    /* compute inertia0 */
    CHKERRQ(EPSSliceGetInertia(eps,sr->int0,&sr->inertia0,ctx->detect?&zeros:NULL));
    /* undocumented option to control what to do when an eigenvalue is found:
       - error out if it's the endpoint of the user-provided interval (or sub-interval)
       - if it's an endpoint computed internally:
          + if hiteig=0 error out
          + else if hiteig=1 the subgroup that hit the eigenvalue does nothing
          + otherwise the subgroup that hit the eigenvalue perturbs the shift and recomputes inertia
    */
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-eps_krylovschur_hiteigenvalue",&hiteig,NULL));
    if (zeros) { /* error in factorization */
      PetscCheck(sr->int0!=ctx->eps->inta && sr->int0!=ctx->eps->intb,((PetscObject)eps)->comm,PETSC_ERR_USER,"Found singular matrix for the transformed problem in the interval endpoint");
      PetscCheck(!ctx_glob->subintset || hiteig,((PetscObject)eps)->comm,PETSC_ERR_USER,"Found singular matrix for the transformed problem in an interval endpoint defined by user");
      if (hiteig==1) { /* idle subgroup */
        sr->inertia0 = -1;
      } else { /* perturb shift */
        sr->int0 *= (1.0+SLICE_PTOL);
        CHKERRQ(EPSSliceGetInertia(eps,sr->int0,&sr->inertia0,&zeros));
        PetscCheck(zeros==0,((PetscObject)eps)->comm,PETSC_ERR_CONV_FAILED,"Inertia computation fails in %g",(double)sr->int1);
      }
    }
    if (ctx->npart>1) {
      CHKERRQ(PetscSubcommGetChild(ctx->subc,&child));
      /* inertia1 is received from neighbour */
      CHKERRMPI(MPI_Comm_rank(child,&rank));
      if (!rank) {
        if (sr->inertia0!=-1 && ((sr->dir>0 && ctx->subc->color>0) || (sr->dir<0 && ctx->subc->color<ctx->npart-1))) { /* send inertia0 to neighbour0 */
          CHKERRMPI(MPI_Isend(&(sr->inertia0),1,MPIU_INT,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
          CHKERRMPI(MPI_Isend(&(sr->int0),1,MPIU_REAL,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
        }
        if ((sr->dir>0 && ctx->subc->color<ctx->npart-1)|| (sr->dir<0 && ctx->subc->color>0)) { /* receive inertia1 from neighbour1 */
          CHKERRMPI(MPI_Recv(&(sr->inertia1),1,MPIU_INT,ctx->subc->color+sr->dir,0,ctx->commrank,MPI_STATUS_IGNORE));
          CHKERRMPI(MPI_Recv(&(sr->int1),1,MPIU_REAL,ctx->subc->color+sr->dir,0,ctx->commrank,MPI_STATUS_IGNORE));
        }
        if (sr->inertia0==-1 && !(sr->dir>0 && ctx->subc->color==ctx->npart-1) && !(sr->dir<0 && ctx->subc->color==0)) {
          sr->inertia0 = sr->inertia1; sr->int0 = sr->int1;
          CHKERRMPI(MPI_Isend(&(sr->inertia0),1,MPIU_INT,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
          CHKERRMPI(MPI_Isend(&(sr->int0),1,MPIU_REAL,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
        }
      }
      if ((sr->dir>0 && ctx->subc->color<ctx->npart-1)||(sr->dir<0 && ctx->subc->color>0)) {
        CHKERRMPI(MPI_Bcast(&sr->inertia1,1,MPIU_INT,0,child));
        CHKERRMPI(MPI_Bcast(&sr->int1,1,MPIU_REAL,0,child));
      } else sr_glob->inertia1 = sr->inertia1;
    }

    /* last process in eps comm computes inertia1 */
    if (ctx->npart==1 || ((sr->dir>0 && ctx->subc->color==ctx->npart-1) || (sr->dir<0 && ctx->subc->color==0))) {
      CHKERRQ(EPSSliceGetInertia(eps,sr->int1,&sr->inertia1,ctx->detect?&zeros:NULL));
      PetscCheck(zeros==0,((PetscObject)eps)->comm,PETSC_ERR_USER,"Found singular matrix for the transformed problem in an interval endpoint defined by user");
      if (!rank && sr->inertia0==-1) {
        sr->inertia0 = sr->inertia1; sr->int0 = sr->int1;
        CHKERRMPI(MPI_Isend(&(sr->inertia0),1,MPIU_INT,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
        CHKERRMPI(MPI_Isend(&(sr->int0),1,MPIU_REAL,ctx->subc->color-sr->dir,0,ctx->commrank,&req));
      }
      if (sr->hasEnd) {
        sr->dir = -sr->dir; r = sr->int0; sr->int0 = sr->int1; sr->int1 = r;
        i = sr->inertia0; sr->inertia0 = sr->inertia1; sr->inertia1 = i;
      }
    }

    /* number of eigenvalues in interval */
    sr->numEigs = (sr->dir)*(sr->inertia1 - sr->inertia0);
    if (ctx->npart>1) {
      /* memory allocate for subinterval eigenpairs */
      CHKERRQ(EPSSliceAllocateSolution(eps,1));
    }
    dssz = eps->ncv+1;
    CHKERRQ(DSGetParallel(ctx->eps->ds,&ptype));
    CHKERRQ(DSSetParallel(eps->ds,ptype));
    CHKERRQ(DSGetMethod(ctx->eps->ds,&method));
    CHKERRQ(DSSetMethod(eps->ds,method));
  }
  CHKERRQ(DSSetType(eps->ds,DSHEP));
  CHKERRQ(DSSetCompact(eps->ds,PETSC_TRUE));
  CHKERRQ(DSAllocate(eps->ds,dssz));
  /* keep state of subcomm matrices to check that the user does not modify them */
  CHKERRQ(EPSGetOperators(eps,&A,&B));
  CHKERRQ(PetscObjectStateGet((PetscObject)A,&ctx->Astate));
  CHKERRQ(PetscObjectGetId((PetscObject)A,&ctx->Aid));
  if (B) {
    CHKERRQ(PetscObjectStateGet((PetscObject)B,&ctx->Bstate));
    CHKERRQ(PetscObjectGetId((PetscObject)B,&ctx->Bid));
  } else {
    ctx->Bstate=0;
    ctx->Bid=0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSSliceGatherEigenVectors(EPS eps)
{
  Vec             v,vg,v_loc;
  IS              is1,is2;
  VecScatter      vec_sc;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        nloc,m0,n0,i,si,idx,*idx1,*idx2,j;
  PetscScalar     *array;
  EPS_SR          sr_loc;
  BV              V_loc;

  PetscFunctionBegin;
  sr_loc = ((EPS_KRYLOVSCHUR*)ctx->eps->data)->sr;
  V_loc = sr_loc->V;

  /* Gather parallel eigenvectors */
  CHKERRQ(BVGetColumn(eps->V,0,&v));
  CHKERRQ(VecGetOwnershipRange(v,&n0,&m0));
  CHKERRQ(BVRestoreColumn(eps->V,0,&v));
  CHKERRQ(BVGetColumn(ctx->eps->V,0,&v));
  CHKERRQ(VecGetLocalSize(v,&nloc));
  CHKERRQ(BVRestoreColumn(ctx->eps->V,0,&v));
  CHKERRQ(PetscMalloc2(m0-n0,&idx1,m0-n0,&idx2));
  CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)eps),nloc,PETSC_DECIDE,&vg));
  idx = -1;
  for (si=0;si<ctx->npart;si++) {
    j = 0;
    for (i=n0;i<m0;i++) {
      idx1[j]   = i;
      idx2[j++] = i+eps->n*si;
    }
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)eps),(m0-n0),idx1,PETSC_COPY_VALUES,&is1));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)eps),(m0-n0),idx2,PETSC_COPY_VALUES,&is2));
    CHKERRQ(BVGetColumn(eps->V,0,&v));
    CHKERRQ(VecScatterCreate(v,is1,vg,is2,&vec_sc));
    CHKERRQ(BVRestoreColumn(eps->V,0,&v));
    CHKERRQ(ISDestroy(&is1));
    CHKERRQ(ISDestroy(&is2));
    for (i=0;i<ctx->nconv_loc[si];i++) {
      CHKERRQ(BVGetColumn(eps->V,++idx,&v));
      if (ctx->subc->color==si) {
        CHKERRQ(BVGetColumn(V_loc,i,&v_loc));
        CHKERRQ(VecGetArray(v_loc,&array));
        CHKERRQ(VecPlaceArray(vg,array));
      }
      CHKERRQ(VecScatterBegin(vec_sc,vg,v,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(vec_sc,vg,v,INSERT_VALUES,SCATTER_REVERSE));
      if (ctx->subc->color==si) {
        CHKERRQ(VecResetArray(vg));
        CHKERRQ(VecRestoreArray(v_loc,&array));
        CHKERRQ(BVRestoreColumn(V_loc,i,&v_loc));
      }
      CHKERRQ(BVRestoreColumn(eps->V,idx,&v));
    }
    CHKERRQ(VecScatterDestroy(&vec_sc));
  }
  CHKERRQ(PetscFree2(idx1,idx2));
  CHKERRQ(VecDestroy(&vg));
  PetscFunctionReturn(0);
}

/*
  EPSComputeVectors_Slice - Recover Eigenvectors from subcomunicators
 */
PetscErrorCode EPSComputeVectors_Slice(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  if (ctx->global && ctx->npart>1) {
    CHKERRQ(EPSComputeVectors(ctx->eps));
    CHKERRQ(EPSSliceGatherEigenVectors(eps));
  }
  PetscFunctionReturn(0);
}

#define SWAP(a,b,t) {t=a;a=b;b=t;}

static PetscErrorCode EPSSliceGetInertias(EPS eps,PetscInt *n,PetscReal **shifts,PetscInt **inertias)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i=0,j,tmpi;
  PetscReal       v,tmpr;
  EPS_shift       s;

  PetscFunctionBegin;
  PetscCheck(eps->state,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSetUp() first");
  PetscCheck(ctx->sr,PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONGSTATE,"Only available in interval computations, see EPSSetInterval()");
  if (!ctx->sr->s0) {  /* EPSSolve not called yet */
    *n = 2;
  } else {
    *n = 1;
    s = ctx->sr->s0;
    while (s) {
      (*n)++;
      s = s->neighb[1];
    }
  }
  CHKERRQ(PetscMalloc1(*n,shifts));
  CHKERRQ(PetscMalloc1(*n,inertias));
  if (!ctx->sr->s0) {  /* EPSSolve not called yet */
    (*shifts)[0]   = ctx->sr->int0;
    (*shifts)[1]   = ctx->sr->int1;
    (*inertias)[0] = ctx->sr->inertia0;
    (*inertias)[1] = ctx->sr->inertia1;
  } else {
    s = ctx->sr->s0;
    while (s) {
      (*shifts)[i]     = s->value;
      (*inertias)[i++] = s->inertia;
      s = s->neighb[1];
    }
    (*shifts)[i]   = ctx->sr->int1;
    (*inertias)[i] = ctx->sr->inertia1;
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

static PetscErrorCode EPSSliceGatherSolution(EPS eps)
{
  PetscMPIInt     rank,nproc,*disp,*ns_loc,aux;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,idx,j,*perm_loc,off=0,*inertias_loc,ns;
  PetscScalar     *eigr_loc;
  EPS_SR          sr_loc;
  PetscReal       *shifts_loc;
  MPI_Comm        child;

  PetscFunctionBegin;
  eps->nconv = 0;
  for (i=0;i<ctx->npart;i++) eps->nconv += ctx->nconv_loc[i];
  sr_loc = ((EPS_KRYLOVSCHUR*)ctx->eps->data)->sr;

  /* Gather the shifts used and the inertias computed */
  CHKERRQ(EPSSliceGetInertias(ctx->eps,&ns,&shifts_loc,&inertias_loc));
  if (ctx->sr->dir>0 && shifts_loc[ns-1]==sr_loc->int1 && ctx->subc->color<ctx->npart-1) ns--;
  if (ctx->sr->dir<0 && shifts_loc[ns-1]==sr_loc->int0 && ctx->subc->color>0) {
    ns--;
    for (i=0;i<ns;i++) {
      inertias_loc[i] = inertias_loc[i+1];
      shifts_loc[i] = shifts_loc[i+1];
    }
  }
  CHKERRQ(PetscMalloc1(ctx->npart,&ns_loc));
  CHKERRQ(PetscSubcommGetChild(ctx->subc,&child));
  CHKERRMPI(MPI_Comm_rank(child,&rank));
  CHKERRQ(PetscMPIIntCast(ns,&aux));
  if (!rank) CHKERRMPI(MPI_Allgather(&aux,1,MPI_INT,ns_loc,1,MPI_INT,ctx->commrank));
  CHKERRQ(PetscMPIIntCast(ctx->npart,&aux));
  CHKERRMPI(MPI_Bcast(ns_loc,aux,MPI_INT,0,child));
  ctx->nshifts = 0;
  for (i=0;i<ctx->npart;i++) ctx->nshifts += ns_loc[i];
  CHKERRQ(PetscFree(ctx->inertias));
  CHKERRQ(PetscFree(ctx->shifts));
  CHKERRQ(PetscMalloc1(ctx->nshifts,&ctx->inertias));
  CHKERRQ(PetscMalloc1(ctx->nshifts,&ctx->shifts));

  /* Gather eigenvalues (same ranks have fully set of eigenvalues)*/
  eigr_loc = sr_loc->eigr;
  perm_loc = sr_loc->perm;
  CHKERRMPI(MPI_Comm_size(((PetscObject)eps)->comm,&nproc));
  CHKERRQ(PetscMalloc1(ctx->npart,&disp));
  disp[0] = 0;
  for (i=1;i<ctx->npart;i++) disp[i] = disp[i-1]+ctx->nconv_loc[i-1];
  if (nproc%ctx->npart==0) { /* subcommunicators with the same size */
    CHKERRQ(PetscMPIIntCast(sr_loc->numEigs,&aux));
    CHKERRMPI(MPI_Allgatherv(eigr_loc,aux,MPIU_SCALAR,eps->eigr,ctx->nconv_loc,disp,MPIU_SCALAR,ctx->commrank)); /* eigenvalues */
    CHKERRMPI(MPI_Allgatherv(perm_loc,aux,MPIU_INT,eps->perm,ctx->nconv_loc,disp,MPIU_INT,ctx->commrank)); /* perm */
    for (i=1;i<ctx->npart;i++) disp[i] = disp[i-1]+ns_loc[i-1];
    CHKERRQ(PetscMPIIntCast(ns,&aux));
    CHKERRMPI(MPI_Allgatherv(shifts_loc,aux,MPIU_REAL,ctx->shifts,ns_loc,disp,MPIU_REAL,ctx->commrank)); /* shifts */
    CHKERRMPI(MPI_Allgatherv(inertias_loc,aux,MPIU_INT,ctx->inertias,ns_loc,disp,MPIU_INT,ctx->commrank)); /* inertias */
    CHKERRMPI(MPIU_Allreduce(&sr_loc->itsKs,&eps->its,1,MPIU_INT,MPI_SUM,ctx->commrank));
  } else { /* subcommunicators with different size */
    if (!rank) {
      CHKERRQ(PetscMPIIntCast(sr_loc->numEigs,&aux));
      CHKERRMPI(MPI_Allgatherv(eigr_loc,aux,MPIU_SCALAR,eps->eigr,ctx->nconv_loc,disp,MPIU_SCALAR,ctx->commrank)); /* eigenvalues */
      CHKERRMPI(MPI_Allgatherv(perm_loc,aux,MPIU_INT,eps->perm,ctx->nconv_loc,disp,MPIU_INT,ctx->commrank)); /* perm */
      for (i=1;i<ctx->npart;i++) disp[i] = disp[i-1]+ns_loc[i-1];
      CHKERRQ(PetscMPIIntCast(ns,&aux));
      CHKERRMPI(MPI_Allgatherv(shifts_loc,aux,MPIU_REAL,ctx->shifts,ns_loc,disp,MPIU_REAL,ctx->commrank)); /* shifts */
      CHKERRMPI(MPI_Allgatherv(inertias_loc,aux,MPIU_INT,ctx->inertias,ns_loc,disp,MPIU_INT,ctx->commrank)); /* inertias */
      CHKERRMPI(MPIU_Allreduce(&sr_loc->itsKs,&eps->its,1,MPIU_INT,MPI_SUM,ctx->commrank));
    }
    CHKERRQ(PetscMPIIntCast(eps->nconv,&aux));
    CHKERRMPI(MPI_Bcast(eps->eigr,aux,MPIU_SCALAR,0,child));
    CHKERRMPI(MPI_Bcast(eps->perm,aux,MPIU_INT,0,child));
    CHKERRMPI(MPI_Bcast(ctx->shifts,ctx->nshifts,MPIU_REAL,0,child));
    CHKERRQ(PetscMPIIntCast(ctx->nshifts,&aux));
    CHKERRMPI(MPI_Bcast(ctx->inertias,aux,MPIU_INT,0,child));
    CHKERRMPI(MPI_Bcast(&eps->its,1,MPIU_INT,0,child));
  }
  /* Update global array eps->perm */
  idx = ctx->nconv_loc[0];
  for (i=1;i<ctx->npart;i++) {
    off += ctx->nconv_loc[i-1];
    for (j=0;j<ctx->nconv_loc[i];j++) eps->perm[idx++] += off;
  }

  /* Gather parallel eigenvectors */
  CHKERRQ(PetscFree(ns_loc));
  CHKERRQ(PetscFree(disp));
  CHKERRQ(PetscFree(shifts_loc));
  CHKERRQ(PetscFree(inertias_loc));
  PetscFunctionReturn(0);
}

/*
   Fills the fields of a shift structure
*/
static PetscErrorCode EPSCreateShift(EPS eps,PetscReal val,EPS_shift neighb0,EPS_shift neighb1)
{
  EPS_shift       s,*pending2;
  PetscInt        i;
  EPS_SR          sr;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  if ((neighb0 && val==neighb0->value) || (neighb1 && val==neighb1->value)) {
    sr->nPend++;
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscNewLog(eps,&s));
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
    CHKERRQ(PetscLogObjectMemory((PetscObject)eps,sr->maxPend*sizeof(EPS_shift*)));
    for (i=0;i<sr->nPend;i++) pending2[i] = sr->pending[i];
    CHKERRQ(PetscFree(sr->pending));
    sr->pending = pending2;
  }
  sr->pending[sr->nPend++]=s;
  PetscFunctionReturn(0);
}

/* Prepare for Rational Krylov update */
static PetscErrorCode EPSPrepareRational(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        dir,i,k,ld,nv;
  PetscScalar     *A;
  EPS_SR          sr = ctx->sr;
  Vec             v;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  dir = (sr->sPres->neighb[0] == sr->sPrev)?1:-1;
  dir*=sr->dir;
  k = 0;
  for (i=0;i<sr->nS;i++) {
    if (dir*PetscRealPart(sr->S[i])>0.0) {
      sr->S[k] = sr->S[i];
      sr->S[sr->nS+k] = sr->S[sr->nS+i];
      CHKERRQ(BVGetColumn(sr->Vnext,k,&v));
      CHKERRQ(BVCopyVec(eps->V,eps->nconv+i,v));
      CHKERRQ(BVRestoreColumn(sr->Vnext,k,&v));
      k++;
      if (k>=sr->nS/2)break;
    }
  }
  /* Copy to DS */
  CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&A));
  CHKERRQ(PetscArrayzero(A,ld*ld));
  for (i=0;i<k;i++) {
    A[i*(1+ld)] = sr->S[i];
    A[k+i*ld] = sr->S[sr->nS+i];
  }
  sr->nS = k;
  CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&A));
  CHKERRQ(DSGetDimensions(eps->ds,&nv,NULL,NULL,NULL));
  CHKERRQ(DSSetDimensions(eps->ds,nv,0,k));
  /* Append u to V */
  CHKERRQ(BVGetColumn(sr->Vnext,sr->nS,&v));
  CHKERRQ(BVCopyVec(eps->V,sr->nv,v));
  CHKERRQ(BVRestoreColumn(sr->Vnext,sr->nS,&v));
  PetscFunctionReturn(0);
}

/* Provides next shift to be computed */
static PetscErrorCode EPSExtractShift(EPS eps)
{
  PetscInt        iner,zeros=0;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR          sr;
  PetscReal       newShift,diam,ptol;
  EPS_shift       sPres;

  PetscFunctionBegin;
  sr = ctx->sr;
  if (sr->nPend > 0) {
    if (sr->sPres==sr->pending[sr->nPend-1]) {
      eps->reason = EPS_CONVERGED_ITERATING;
      eps->its = 0;
      sr->nPend--;
      sr->sPres->rep = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
    sr->sPrev = sr->sPres;
    sr->sPres = sr->pending[--sr->nPend];
    sPres = sr->sPres;
    CHKERRQ(EPSSliceGetInertia(eps,sPres->value,&iner,ctx->detect?&zeros:NULL));
    if (zeros) {
      diam = PetscMin(PetscAbsReal(sPres->neighb[0]->value-sPres->value),PetscAbsReal(sPres->neighb[1]->value-sPres->value));
      ptol = PetscMin(SLICE_PTOL,diam/2);
      newShift = sPres->value*(1.0+ptol);
      if (sr->dir*(sPres->neighb[0] && newShift-sPres->neighb[0]->value) < 0) newShift = (sPres->value+sPres->neighb[0]->value)/2;
      else if (sPres->neighb[1] && sr->dir*(sPres->neighb[1]->value-newShift) < 0) newShift = (sPres->value+sPres->neighb[1]->value)/2;
      CHKERRQ(EPSSliceGetInertia(eps,newShift,&iner,&zeros));
      PetscCheck(zeros==0,((PetscObject)eps)->comm,PETSC_ERR_CONV_FAILED,"Inertia computation fails in %g",(double)newShift);
      sPres->value = newShift;
    }
    sr->sPres->inertia = iner;
    eps->target = sr->sPres->value;
    eps->reason = EPS_CONVERGED_ITERATING;
    eps->its = 0;
  } else sr->sPres = NULL;
  PetscFunctionReturn(0);
}

/*
   Symmetric KrylovSchur adapted to spectrum slicing:
   Allows searching an specific amount of eigenvalues in the subintervals left and right.
   Returns whether the search has succeeded
*/
static PetscErrorCode EPSKrylovSchur_Slice(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,k,l,ld,nv,*iwork,j,count0,count1,iterCompl=0,n0,n1;
  Mat             U,Op;
  PetscScalar     *Q,*A;
  PetscReal       *a,*b,beta,lambda;
  EPS_shift       sPres;
  PetscBool       breakdown,complIterating,sch0,sch1;
  EPS_SR          sr = ctx->sr;

  PetscFunctionBegin;
  /* Spectrum slicing data */
  sPres = sr->sPres;
  complIterating =PETSC_FALSE;
  sch1 = sch0 = PETSC_TRUE;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(PetscMalloc1(2*ld,&iwork));
  count0=0;count1=0; /* Found on both sides */
  if (!sPres->rep && sr->nS > 0 && (sPres->neighb[0] == sr->sPrev || sPres->neighb[1] == sr->sPrev)) {
    /* Rational Krylov */
    CHKERRQ(DSTranslateRKS(eps->ds,sr->sPrev->value-sPres->value));
    CHKERRQ(DSGetDimensions(eps->ds,NULL,NULL,&l,NULL));
    CHKERRQ(DSSetDimensions(eps->ds,l+1,0,0));
    CHKERRQ(BVSetActiveColumns(eps->V,0,l+1));
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
    CHKERRQ(BVMultInPlace(eps->V,U,0,l+1));
    CHKERRQ(MatDestroy(&U));
  } else {
    /* Get the starting Lanczos vector */
    CHKERRQ(EPSGetStartVector(eps,0,NULL));
    l = 0;
  }
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++; sr->itsKs++;
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
    b = a + ld;
    CHKERRQ(STGetOperator(eps->st,&Op));
    CHKERRQ(BVMatLanczos(eps->V,Op,a,b,eps->nconv+l,&nv,&breakdown));
    CHKERRQ(STRestoreOperator(eps->st,&Op));
    sr->nv = nv;
    beta = b[nv-1];
    CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
    CHKERRQ(DSSetDimensions(eps->ds,nv,eps->nconv,eps->nconv+l));
    if (l==0) {
      CHKERRQ(DSSetState(eps->ds,DS_STATE_INTERMEDIATE));
    } else {
      CHKERRQ(DSSetState(eps->ds,DS_STATE_RAW));
    }
    CHKERRQ(BVSetActiveColumns(eps->V,eps->nconv,nv));

    /* Solve projected problem and compute residual norm estimates */
    if (eps->its == 1 && l > 0) {/* After rational update */
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_A,&A));
      CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
      b = a + ld;
      k = eps->nconv+l;
      A[k*ld+k-1] = A[(k-1)*ld+k];
      A[k*ld+k] = a[k];
      for (j=k+1; j< nv; j++) {
        A[j*ld+j] = a[j];
        A[j*ld+j-1] = b[j-1] ;
        A[(j-1)*ld+j] = b[j-1];
      }
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_A,&A));
      CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
      CHKERRQ(DSSolve(eps->ds,eps->eigr,NULL));
      CHKERRQ(DSSort(eps->ds,eps->eigr,NULL,NULL,NULL,NULL));
      CHKERRQ(DSSetCompact(eps->ds,PETSC_TRUE));
    } else { /* Restart */
      CHKERRQ(DSSolve(eps->ds,eps->eigr,NULL));
      CHKERRQ(DSSort(eps->ds,eps->eigr,NULL,NULL,NULL,NULL));
    }
    CHKERRQ(DSSynchronize(eps->ds,eps->eigr,NULL));

    /* Residual */
    CHKERRQ(EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv-eps->nconv,beta,0.0,1.0,&k));
    /* Checking values obtained for completing */
    for (i=0;i<k;i++) {
      sr->back[i]=eps->eigr[i];
    }
    CHKERRQ(STBackTransform(eps->st,k,sr->back,eps->eigi));
    count0=count1=0;
    for (i=0;i<k;i++) {
      lambda = PetscRealPart(sr->back[i]);
      if (((sr->dir)*(sPres->value - lambda) > 0) && ((sr->dir)*(lambda - sPres->ext[0]) > 0)) count0++;
      if (((sr->dir)*(lambda - sPres->value) > 0) && ((sr->dir)*(sPres->ext[1] - lambda) > 0)) count1++;
    }
    if (k>eps->nev && eps->ncv-k<5) eps->reason = EPS_CONVERGED_TOL;
    else {
      /* Checks completion */
      if ((!sch0||count0 >= sPres->nsch[0]) && (!sch1 ||count1 >= sPres->nsch[1])) {
        eps->reason = EPS_CONVERGED_TOL;
      } else {
        if (!complIterating && eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
        if (complIterating) {
          if (--iterCompl <= 0) eps->reason = EPS_DIVERGED_ITS;
        } else if (k >= eps->nev) {
          n0 = sPres->nsch[0]-count0;
          n1 = sPres->nsch[1]-count1;
          if (sr->iterCompl>0 && ((n0>0 && n0<= sr->nMAXCompl)||(n1>0&&n1<=sr->nMAXCompl))) {
            /* Iterating for completion*/
            complIterating = PETSC_TRUE;
            if (n0 >sr->nMAXCompl)sch0 = PETSC_FALSE;
            if (n1 >sr->nMAXCompl)sch1 = PETSC_FALSE;
            iterCompl = sr->iterCompl;
          } else eps->reason = EPS_CONVERGED_TOL;
        }
      }
    }
    /* Update l */
    if (eps->reason == EPS_CONVERGED_ITERATING) l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
    else l = nv-k;
    if (breakdown) l=0;
    if (!ctx->lock && l>0 && eps->reason == EPS_CONVERGED_ITERATING) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        CHKERRQ(PetscInfo(eps,"Breakdown in Krylov-Schur method (it=%" PetscInt_FMT " norm=%g)\n",eps->its,(double)beta));
        CHKERRQ(EPSGetStartVector(eps,k,&breakdown));
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          CHKERRQ(PetscInfo(eps,"Unable to generate more start vectors\n"));
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&a));
        CHKERRQ(DSGetArray(eps->ds,DS_MAT_Q,&Q));
        b = a + ld;
        for (i=k;i<k+l;i++) {
          a[i] = PetscRealPart(eps->eigr[i]);
          b[i] = PetscRealPart(Q[nv-1+i*ld]*beta);
        }
        CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&a));
        CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_Q,&Q));
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    CHKERRQ(DSGetMat(eps->ds,DS_MAT_Q,&U));
    CHKERRQ(BVMultInPlace(eps->V,U,eps->nconv,k+l));
    CHKERRQ(MatDestroy(&U));

    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      CHKERRQ(BVCopyColumn(eps->V,nv,k+l));
    }
    eps->nconv = k;
    if (eps->reason != EPS_CONVERGED_ITERATING) {
      /* Store approximated values for next shift */
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_Q,&Q));
      sr->nS = l;
      for (i=0;i<l;i++) {
        sr->S[i] = eps->eigr[i+k];/* Diagonal elements */
        sr->S[i+l] = Q[nv-1+(i+k)*ld]*beta; /* Out of diagonal elements */
      }
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_Q,&Q));
    }
  }
  /* Check for completion */
  for (i=0;i< eps->nconv; i++) {
    if ((sr->dir)*PetscRealPart(eps->eigr[i])>0) sPres->nconv[1]++;
    else sPres->nconv[0]++;
  }
  sPres->comp[0] = PetscNot(count0 < sPres->nsch[0]);
  sPres->comp[1] = PetscNot(count1 < sPres->nsch[1]);
  CHKERRQ(PetscInfo(eps,"Lanczos: %" PetscInt_FMT " evals in [%g,%g]%s and %" PetscInt_FMT " evals in [%g,%g]%s\n",count0,(double)(sr->dir==1?sPres->ext[0]:sPres->value),(double)(sr->dir==1?sPres->value:sPres->ext[0]),sPres->comp[0]?"*":"",count1,(double)(sr->dir==1?sPres->value:sPres->ext[1]),(double)(sr->dir==1?sPres->ext[1]:sPres->value),sPres->comp[1]?"*":""));
  PetscCheck(count0<=sPres->nsch[0] && count1<=sPres->nsch[1],PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia%s",ctx->detect?"":", consider using EPSKrylovSchurSetDetectZeros()");
  CHKERRQ(PetscFree(iwork));
  PetscFunctionReturn(0);
}

/*
  Obtains value of subsequent shift
*/
static PetscErrorCode EPSGetNewShiftValue(EPS eps,PetscInt side,PetscReal *newS)
{
  PetscReal       lambda,d_prev;
  PetscInt        i,idxP;
  EPS_SR          sr;
  EPS_shift       sPres,s;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

  PetscFunctionBegin;
  sr = ctx->sr;
  sPres = sr->sPres;
  if (sPres->neighb[side]) {
    /* Completing a previous interval */
    *newS = (sPres->value + sPres->neighb[side]->value)/2;
    if (PetscAbsReal(sPres->value - *newS)/PetscAbsReal(sPres->value)<=100*PETSC_SQRT_MACHINE_EPSILON) *newS = sPres->value;
  } else { /* (Only for side=1). Creating a new interval. */
    if (sPres->neigs==0) {/* No value has been accepted*/
      if (sPres->neighb[0]) {
        /* Multiplying by 10 the previous distance */
        *newS = sPres->value + 10*(sr->dir)*PetscAbsReal(sPres->value - sPres->neighb[0]->value);
        sr->nleap++;
        /* Stops when the interval is open and no values are found in the last 5 shifts (there might be infinite eigenvalues) */
        PetscCheck(sr->hasEnd || sr->nleap<=5,PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Unable to compute the wanted eigenvalues with open interval");
      } else { /* First shift */
        PetscCheck(eps->nconv!=0,PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"First shift renders no information");
        /* Unaccepted values give information for next shift */
        idxP=0;/* Number of values left from shift */
        for (i=0;i<eps->nconv;i++) {
          lambda = PetscRealPart(eps->eigr[i]);
          if ((sr->dir)*(lambda - sPres->value) <0) idxP++;
          else break;
        }
        /* Avoiding subtraction of eigenvalues (might be the same).*/
        if (idxP>0) {
          d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[0]))/(idxP+0.3);
        } else {
          d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[eps->nconv-1]))/(eps->nconv+0.3);
        }
        *newS = sPres->value + ((sr->dir)*d_prev*eps->nev)/2;
      }
    } else { /* Accepted values found */
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
        if ((sr->dir)*(PetscRealPart(sr->eigr[0])-sPres->value)>0 && PetscAbsReal((PetscRealPart(sr->eigr[sr->indexEig-1]) - PetscRealPart(sr->eigr[0]))/PetscRealPart(sr->eigr[0])) > PetscSqrtReal(eps->tol)) {
          d_prev =  PetscAbsReal((PetscRealPart(sr->eigr[sr->indexEig-1]) - PetscRealPart(sr->eigr[0])))/(sPres->neigs+0.3);
        } else {
          d_prev = PetscAbsReal(PetscRealPart(sr->eigr[sr->indexEig-1]) - sPres->value)/(sPres->neigs+0.3);
        }
      }
      /* Average distance is used for next shift by adding it to value on the right or to shift */
      if ((sr->dir)*(PetscRealPart(sr->eigr[sPres->index + sPres->neigs -1]) - sPres->value)>0) {
        *newS = PetscRealPart(sr->eigr[sPres->index + sPres->neigs -1])+ ((sr->dir)*d_prev*(eps->nev))/2;
      } else { /* Last accepted value is on the left of shift. Adding to shift */
        *newS = sPres->value + ((sr->dir)*d_prev*(eps->nev))/2;
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
static PetscErrorCode EPSStoreEigenpairs(EPS eps)
{
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscReal       lambda,err,norm;
  PetscInt        i,count;
  PetscBool       iscayley;
  EPS_SR          sr = ctx->sr;
  EPS_shift       sPres;
  Vec             v,w;

  PetscFunctionBegin;
  sPres = sr->sPres;
  sPres->index = sr->indexEig;
  count = sr->indexEig;
  /* Back-transform */
  CHKERRQ(STBackTransform(eps->st,eps->nconv,eps->eigr,eps->eigi));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STCAYLEY,&iscayley));
  /* Sort eigenvalues */
  CHKERRQ(sortRealEigenvalues(eps->eigr,eps->perm,eps->nconv,PETSC_FALSE,sr->dir));
  /* Values stored in global array */
  for (i=0;i<eps->nconv;i++) {
    lambda = PetscRealPart(eps->eigr[eps->perm[i]]);
    err = eps->errest[eps->perm[i]];

    if ((sr->dir)*(lambda - sPres->ext[0]) > 0 && (sr->dir)*(sPres->ext[1] - lambda) > 0) {/* Valid value */
      PetscCheck(count<sr->numEigs,PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Unexpected error in Spectrum Slicing");
      sr->eigr[count] = lambda;
      sr->errest[count] = err;
      /* Explicit purification */
      CHKERRQ(BVGetColumn(eps->V,eps->perm[i],&w));
      if (eps->purify) {
        CHKERRQ(BVGetColumn(sr->V,count,&v));
        CHKERRQ(STApply(eps->st,w,v));
        CHKERRQ(BVRestoreColumn(sr->V,count,&v));
      } else {
        CHKERRQ(BVInsertVec(sr->V,count,w));
      }
      CHKERRQ(BVRestoreColumn(eps->V,eps->perm[i],&w));
      CHKERRQ(BVNormColumn(sr->V,count,NORM_2,&norm));
      CHKERRQ(BVScaleColumn(sr->V,count,1.0/norm));
      count++;
    }
  }
  sPres->neigs = count - sr->indexEig;
  sr->indexEig = count;
  /* Global ordering array updating */
  CHKERRQ(sortRealEigenvalues(sr->eigr,sr->perm,count,PETSC_TRUE,sr->dir));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSLookForDeflation(EPS eps)
{
  PetscReal       val;
  PetscInt        i,count0=0,count1=0;
  EPS_shift       sPres;
  PetscInt        ini,fin,k,idx0,idx1;
  EPS_SR          sr;
  Vec             v;
  EPS_KRYLOVSCHUR *ctx=(EPS_KRYLOVSCHUR*)eps->data;

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
    PetscCheck(sPres->nsch[0]>=0,PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia%s",ctx->detect?"":", consider using EPSKrylovSchurSetDetectZeros()");
  } else sPres->nsch[0] = 0;

  if (sPres->neighb[1]) {
    sPres->nsch[1] = (sr->dir)*(sPres->neighb[1]->inertia - sPres->inertia) - count1;
    PetscCheck(sPres->nsch[1]>=0,PetscObjectComm((PetscObject)eps),PETSC_ERR_PLIB,"Mismatch between number of values found and information from inertia%s",ctx->detect?"":", consider using EPSKrylovSchurSetDetectZeros()");
  } else sPres->nsch[1] = (sr->dir)*(sr->inertia1 - sPres->inertia);

  /* Completing vector of indexes for deflation */
  idx0 = ini;
  idx1 = ini+count0+count1;
  k=0;
  for (i=idx0;i<idx1;i++) sr->idxDef[k++]=sr->perm[i];
  CHKERRQ(BVDuplicateResize(eps->V,k+eps->ncv+1,&sr->Vnext));
  CHKERRQ(BVSetNumConstraints(sr->Vnext,k));
  for (i=0;i<k;i++) {
    CHKERRQ(BVGetColumn(sr->Vnext,-i-1,&v));
    CHKERRQ(BVCopyVec(sr->V,sr->idxDef[i],v));
    CHKERRQ(BVRestoreColumn(sr->Vnext,-i-1,&v));
  }

  /* For rational Krylov */
  if (sr->nS>0 && (sr->sPrev == sr->sPres->neighb[0] || sr->sPrev == sr->sPres->neighb[1])) {
    CHKERRQ(EPSPrepareRational(eps));
  }
  eps->nconv = 0;
  /* Get rid of temporary Vnext */
  CHKERRQ(BVDestroy(&eps->V));
  eps->V = sr->Vnext;
  sr->Vnext = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS eps)
{
  PetscInt         i,lds,ti;
  PetscReal        newS;
  EPS_KRYLOVSCHUR  *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  EPS_SR           sr=ctx->sr;
  Mat              A,B=NULL;
  PetscObjectState Astate,Bstate=0;
  PetscObjectId    Aid,Bid=0;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));
  if (ctx->global) {
    CHKERRQ(EPSSolve_KrylovSchur_Slice(ctx->eps));
    ctx->eps->state = EPS_STATE_SOLVED;
    eps->reason = EPS_CONVERGED_TOL;
    if (ctx->npart>1) {
      /* Gather solution from subsolvers */
      CHKERRQ(EPSSliceGatherSolution(eps));
    } else {
      eps->nconv = sr->numEigs;
      eps->its   = ctx->eps->its;
      CHKERRQ(PetscFree(ctx->inertias));
      CHKERRQ(PetscFree(ctx->shifts));
      CHKERRQ(EPSSliceGetInertias(ctx->eps,&ctx->nshifts,&ctx->shifts,&ctx->inertias));
    }
  } else {
    if (ctx->npart==1) {
      sr->eigr   = ctx->eps->eigr;
      sr->eigi   = ctx->eps->eigi;
      sr->perm   = ctx->eps->perm;
      sr->errest = ctx->eps->errest;
      sr->V      = ctx->eps->V;
    }
    /* Check that the user did not modify subcomm matrices */
    CHKERRQ(EPSGetOperators(eps,&A,&B));
    CHKERRQ(PetscObjectStateGet((PetscObject)A,&Astate));
    CHKERRQ(PetscObjectGetId((PetscObject)A,&Aid));
    if (B) {
      CHKERRQ(PetscObjectStateGet((PetscObject)B,&Bstate));
      CHKERRQ(PetscObjectGetId((PetscObject)B,&Bid));
    }
    PetscCheck(Astate==ctx->Astate && (!B || Bstate==ctx->Bstate) && Aid==ctx->Aid && (!B || Bid==ctx->Bid),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subcomm matrices have been modified by user");
    /* Only with eigenvalues present in the interval ...*/
    if (sr->numEigs==0) {
      eps->reason = EPS_CONVERGED_TOL;
      PetscFunctionReturn(0);
    }
    /* Array of pending shifts */
    sr->maxPend = 100; /* Initial size */
    sr->nPend = 0;
    CHKERRQ(PetscMalloc1(sr->maxPend,&sr->pending));
    CHKERRQ(PetscLogObjectMemory((PetscObject)eps,sr->maxPend*sizeof(EPS_shift*)));
    CHKERRQ(EPSCreateShift(eps,sr->int0,NULL,NULL));
    /* extract first shift */
    sr->sPrev = NULL;
    sr->sPres = sr->pending[--sr->nPend];
    sr->sPres->inertia = sr->inertia0;
    eps->target = sr->sPres->value;
    sr->s0 = sr->sPres;
    sr->indexEig = 0;
    /* Memory reservation for auxiliary variables */
    lds = PetscMin(eps->mpd,eps->ncv);
    CHKERRQ(PetscCalloc1(lds*lds,&sr->S));
    CHKERRQ(PetscMalloc1(eps->ncv,&sr->back));
    CHKERRQ(PetscLogObjectMemory((PetscObject)eps,(lds*lds+eps->ncv)*sizeof(PetscScalar)));
    for (i=0;i<sr->numEigs;i++) {
      sr->eigr[i]   = 0.0;
      sr->eigi[i]   = 0.0;
      sr->errest[i] = 0.0;
      sr->perm[i]   = i;
    }
    /* Vectors for deflation */
    CHKERRQ(PetscMalloc1(sr->numEigs,&sr->idxDef));
    CHKERRQ(PetscLogObjectMemory((PetscObject)eps,sr->numEigs*sizeof(PetscInt)));
    sr->indexEig = 0;
    /* Main loop */
    while (sr->sPres) {
      /* Search for deflation */
      CHKERRQ(EPSLookForDeflation(eps));
      /* KrylovSchur */
      CHKERRQ(EPSKrylovSchur_Slice(eps));

      CHKERRQ(EPSStoreEigenpairs(eps));
      /* Select new shift */
      if (!sr->sPres->comp[1]) {
        CHKERRQ(EPSGetNewShiftValue(eps,1,&newS));
        CHKERRQ(EPSCreateShift(eps,newS,sr->sPres,sr->sPres->neighb[1]));
      }
      if (!sr->sPres->comp[0]) {
        /* Completing earlier interval */
        CHKERRQ(EPSGetNewShiftValue(eps,0,&newS));
        CHKERRQ(EPSCreateShift(eps,newS,sr->sPres->neighb[0],sr->sPres));
      }
      /* Preparing for a new search of values */
      CHKERRQ(EPSExtractShift(eps));
    }

    /* Updating eps values prior to exit */
    CHKERRQ(PetscFree(sr->S));
    CHKERRQ(PetscFree(sr->idxDef));
    CHKERRQ(PetscFree(sr->pending));
    CHKERRQ(PetscFree(sr->back));
    CHKERRQ(BVDuplicateResize(eps->V,eps->ncv+1,&sr->Vnext));
    CHKERRQ(BVSetNumConstraints(sr->Vnext,0));
    CHKERRQ(BVDestroy(&eps->V));
    eps->V      = sr->Vnext;
    eps->nconv  = sr->indexEig;
    eps->reason = EPS_CONVERGED_TOL;
    eps->its    = sr->itsKs;
    eps->nds    = 0;
    if (sr->dir<0) {
      for (i=0;i<eps->nconv/2;i++) {
        ti = sr->perm[i]; sr->perm[i] = sr->perm[eps->nconv-1-i]; sr->perm[eps->nconv-1-i] = ti;
      }
    }
  }
  PetscFunctionReturn(0);
}
