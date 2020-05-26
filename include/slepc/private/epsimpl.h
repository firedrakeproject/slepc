/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(SLEPCEPSIMPL_H)
#define SLEPCEPSIMPL_H

#include <slepceps.h>
#include <slepc/private/slepcimpl.h>

SLEPC_EXTERN PetscBool EPSRegisterAllCalled;
SLEPC_EXTERN PetscErrorCode EPSRegisterAll(void);
SLEPC_EXTERN PetscLogEvent EPS_SetUp,EPS_Solve;

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  PetscErrorCode (*solve)(EPS);
  PetscErrorCode (*setup)(EPS);
  PetscErrorCode (*setupsort)(EPS);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,EPS);
  PetscErrorCode (*publishoptions)(EPS);
  PetscErrorCode (*destroy)(EPS);
  PetscErrorCode (*reset)(EPS);
  PetscErrorCode (*view)(EPS,PetscViewer);
  PetscErrorCode (*backtransform)(EPS);
  PetscErrorCode (*computevectors)(EPS);
  PetscErrorCode (*setdefaultst)(EPS);
};

/*
   Maximum number of monitors you can run with a single EPS
*/
#define MAXEPSMONITORS 5

/*
   The solution process goes through several states
*/
typedef enum { EPS_STATE_INITIAL,
               EPS_STATE_SETUP,
               EPS_STATE_SOLVED,
               EPS_STATE_EIGENVECTORS } EPSStateType;

/*
   To classify the different solvers into categories
*/
typedef enum { EPS_CATEGORY_KRYLOV,      /* Krylov solver: relies on STApply and STBackTransform (same as OTHER) */
               EPS_CATEGORY_PRECOND,     /* Preconditioned solver: uses ST only to manage preconditioner */
               EPS_CATEGORY_CONTOUR,     /* Contour integral: ST used to solve linear systems at integration points */
               EPS_CATEGORY_OTHER } EPSSolverType;

/*
   To check for unsupported features at EPSSetUp_XXX()
*/
typedef enum { EPS_FEATURE_BALANCE=1,       /* balancing */
               EPS_FEATURE_ARBITRARY=2,     /* arbitrary selection of eigepairs */
               EPS_FEATURE_REGION=4,        /* nontrivial region for filtering */
               EPS_FEATURE_EXTRACTION=8,    /* extraction technique different from Ritz */
               EPS_FEATURE_CONVERGENCE=16,  /* convergence test selected by user */
               EPS_FEATURE_STOPPING=32,     /* stopping test */
               EPS_FEATURE_TWOSIDED=64      /* two-sided variant */
             } EPSFeatureType;

/*
   Defines the EPS data structure
*/
struct _p_EPS {
  PETSCHEADER(struct _EPSOps);
  /*------------------------- User parameters ---------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini,ninil;       /* number of initial vectors (negative means not copied yet) */
  PetscInt       nds;              /* number of basis vectors of deflation space */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  EPSConv        conv;             /* convergence test */
  EPSStop        stop;             /* stopping test */
  EPSWhich       which;            /* which part of the spectrum to be sought */
  PetscReal      inta,intb;        /* interval [a,b] for spectrum slicing */
  EPSProblemType problem_type;     /* which kind of problem to be solved */
  EPSExtraction  extraction;       /* which kind of extraction to be applied */
  EPSBalance     balance;          /* the balancing method */
  PetscInt       balance_its;      /* number of iterations of the balancing method */
  PetscReal      balance_cutoff;   /* cutoff value for balancing */
  PetscBool      trueres;          /* whether the true residual norm must be computed */
  PetscBool      trackall;         /* whether all the residuals must be computed */
  PetscBool      purify;           /* whether eigenvectors need to be purified */
  PetscBool      twosided;         /* whether to compute left eigenvectors (two-sided solver) */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*converged)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  PetscErrorCode (*convergeduser)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  PetscErrorCode (*stopping)(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);
  PetscErrorCode (*stoppinguser)(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);
  PetscErrorCode (*stoppingdestroy)(void*);
  PetscErrorCode (*arbitrary)(PetscScalar,PetscScalar,Vec,Vec,PetscScalar*,PetscScalar*,void*);
  void           *convergedctx;
  void           *stoppingctx;
  void           *arbitraryctx;
  PetscErrorCode (*monitor[MAXEPSMONITORS])(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXEPSMONITORS])(void**);
  void           *monitorcontext[MAXEPSMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  ST             st;               /* spectral transformation object */
  DS             ds;               /* direct solver object */
  DS             dsts;             /* auxiliary direct solver object used in two-sided case */
  BV             V;                /* set of basis vectors and computed eigenvectors */
  BV             W;                /* left basis vectors (if left eigenvectors requested) */
  RG             rg;               /* optional region for filtering */
  SlepcSC        sc;               /* sorting criterion data */
  Vec            D;                /* diagonal matrix for balancing */
  Vec            *IS,*ISL;         /* references to user-provided initial spaces */
  Vec            *defl;            /* references to user-provided deflation space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  PetscScalar    *rr,*ri;          /* values computed by user's arbitrary selection function */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       nwork;            /* number of work vectors */
  Vec            *work;            /* work vectors */
  void           *data;            /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables --------------------------*/
  EPSStateType   state;            /* initial -> setup -> solved -> eigenvectors */
  EPSSolverType  categ;            /* solver category */
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscReal      nrma,nrmb;        /* computed matrix norms */
  PetscBool      useds;            /* whether the solver uses the DS object or not */
  PetscBool      isgeneralized;
  PetscBool      ispositive;
  PetscBool      ishermitian;
  EPSConvergedReason reason;
};

/*
    Macros to test valid EPS arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define EPSCheckSolved(h,arg) do {} while (0)

#else

#define EPSCheckSolved(h,arg) \
  do { \
    if ((h)->state<EPS_STATE_SOLVED) SETERRQ1(PetscObjectComm((PetscObject)(h)),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSolve() first: Parameter #%d",arg); \
  } while (0)

#endif

/*
    Macros to check settings at EPSSetUp()
*/

/* EPSCheckHermitianDefinite: the problem is HEP or GHEP */
#define EPSCheckHermitianDefiniteCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      if (!(eps)->ishermitian) SETERRQ3(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s cannot be used for non-%s problems",((PetscObject)(eps))->type_name,(msg),SLEPC_STRING_HERMITIAN); \
      else if ((eps)->isgeneralized && !(eps)->ispositive) SETERRQ3(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s requires that the problem is %s-definite",((PetscObject)(eps))->type_name,(msg),SLEPC_STRING_HERMITIAN); \
    } \
  } while (0)
#define EPSCheckHermitianDefinite(eps) EPSCheckHermitianDefiniteCondition(eps,PETSC_TRUE,"")

/* EPSCheckHermitian: the problem is HEP, GHEP, or GHIEP */
#define EPSCheckHermitianCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      if (!(eps)->ishermitian) SETERRQ3(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s cannot be used for non-%s problems",((PetscObject)(eps))->type_name,(msg),SLEPC_STRING_HERMITIAN); \
    } \
  } while (0)
#define EPSCheckHermitian(eps) EPSCheckHermitianCondition(eps,PETSC_TRUE,"")

/* EPSCheckDefinite: the problem is not GHIEP */
#define EPSCheckDefiniteCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      if ((eps)->isgeneralized && (eps)->ishermitian && !(eps)->ispositive) SETERRQ3(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s cannot be used for %s-indefinite problems",((PetscObject)(eps))->type_name,(msg),SLEPC_STRING_HERMITIAN); \
    } \
  } while (0)
#define EPSCheckDefinite(eps) EPSCheckDefiniteCondition(eps,PETSC_TRUE,"")

/* EPSCheckStandard: the problem is HEP or NHEP */
#define EPSCheckStandardCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      if ((eps)->isgeneralized) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s cannot be used for generalized problems",((PetscObject)(eps))->type_name,(msg)); \
    } \
  } while (0)
#define EPSCheckStandard(eps) EPSCheckStandardCondition(eps,PETSC_TRUE,"")

/* EPSCheckSinvert: shift-and-invert ST */
#define EPSCheckSinvertCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      PetscBool __flg; \
      ierr = PetscObjectTypeCompare((PetscObject)(eps)->st,STSINVERT,&__flg);CHKERRQ(ierr); \
      if (!__flg) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s requires a shift-and-invert spectral transform",((PetscObject)(eps))->type_name,(msg)); \
    } \
  } while (0)
#define EPSCheckSinvert(eps) EPSCheckSinvertCondition(eps,PETSC_TRUE,"")

/* EPSCheckSinvertCayley: shift-and-invert or Cayley ST */
#define EPSCheckSinvertCayleyCondition(eps,condition,msg) \
  do { \
    if (condition) { \
      PetscBool __flg; \
      ierr = PetscObjectTypeCompareAny((PetscObject)(eps)->st,&__flg,STSINVERT,STCAYLEY,"");CHKERRQ(ierr); \
      if (!__flg) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s requires shift-and-invert or Cayley transform",((PetscObject)(eps))->type_name,(msg)); \
    } \
  } while (0)
#define EPSCheckSinvertCayley(eps) EPSCheckSinvertCayleyCondition(eps,PETSC_TRUE,"")

/* Check for unsupported features */
#define EPSCheckUnsupportedCondition(eps,mask,condition,msg) \
  do { \
    if (condition) { \
      if (((mask) & EPS_FEATURE_BALANCE) && (eps)->balance!=EPS_BALANCE_NONE) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s does not support balancing",((PetscObject)(eps))->type_name,(msg)); \
      if (((mask) & EPS_FEATURE_ARBITRARY) && (eps)->arbitrary) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s does not support arbitrary selection of eigenpairs",((PetscObject)(eps))->type_name,(msg)); \
      if ((mask) & EPS_FEATURE_REGION) { \
        PetscBool      __istrivial; \
        PetscErrorCode __ierr = RGIsTrivial((eps)->rg,&__istrivial);CHKERRQ(__ierr); \
        if (!__istrivial) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s does not support region filtering",((PetscObject)(eps))->type_name,(msg)); \
      } \
      if (((mask) & EPS_FEATURE_EXTRACTION) && (eps)->extraction!=EPS_RITZ) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s only supports Ritz extraction",((PetscObject)(eps))->type_name,(msg)); \
      if (((mask) & EPS_FEATURE_CONVERGENCE) && (eps)->converged!=EPSConvergedRelative) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s only supports the default convergence test",((PetscObject)(eps))->type_name,(msg)); \
      if (((mask) & EPS_FEATURE_STOPPING) && (eps)->stopping!=EPSStoppingBasic) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s only supports the default stopping test",((PetscObject)(eps))->type_name,(msg)); \
      if (((mask) & EPS_FEATURE_TWOSIDED) && (eps)->twosided) SETERRQ2(PetscObjectComm((PetscObject)(eps)),PETSC_ERR_SUP,"The solver '%s'%s cannot compute left eigenvectors (no two-sided variant)",((PetscObject)(eps))->type_name,(msg)); \
    } \
  } while (0)
#define EPSCheckUnsupported(eps,mask) EPSCheckUnsupportedCondition(eps,mask,PETSC_TRUE,"")

/* Check for ignored features */
#define EPSCheckIgnoredCondition(eps,mask,condition,msg) \
  do { \
    PetscErrorCode __ierr; \
    if (condition) { \
      if (((mask) & EPS_FEATURE_BALANCE) && (eps)->balance!=EPS_BALANCE_NONE) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the balancing settings\n",((PetscObject)(eps))->type_name,(msg)); } \
      if (((mask) & EPS_FEATURE_ARBITRARY) && (eps)->arbitrary) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the settings for arbitrary selection of eigenpairs\n",((PetscObject)(eps))->type_name,(msg)); } \
      if ((mask) & EPS_FEATURE_REGION) { \
        PetscBool __istrivial; \
        __ierr = RGIsTrivial((eps)->rg,&__istrivial);CHKERRQ(__ierr); \
        if (!__istrivial) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the specified region\n",((PetscObject)(eps))->type_name,(msg)); } \
      } \
      if (((mask) & EPS_FEATURE_EXTRACTION) && (eps)->extraction!=EPS_RITZ) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the extraction settings\n",((PetscObject)(eps))->type_name,(msg)); } \
      if (((mask) & EPS_FEATURE_CONVERGENCE) && (eps)->converged!=EPSConvergedRelative) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the convergence test settings\n",((PetscObject)(eps))->type_name,(msg)); } \
      if (((mask) & EPS_FEATURE_STOPPING) && (eps)->stopping!=EPSStoppingBasic) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the stopping test settings\n",((PetscObject)(eps))->type_name,(msg)); } \
      if (((mask) & EPS_FEATURE_TWOSIDED) && (eps)->twosided) { __ierr = PetscInfo2((eps),"The solver '%s'%s ignores the two-sided flag\n",((PetscObject)(eps))->type_name,(msg)); } \
    } \
  } while (0)
#define EPSCheckIgnored(eps,mask) EPSCheckIgnoredCondition(eps,mask,PETSC_TRUE,"")

/*
  EPS_SetInnerProduct - set B matrix for inner product if appropriate.
*/
PETSC_STATIC_INLINE PetscErrorCode EPS_SetInnerProduct(EPS eps)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
  if (eps->ispositive || (eps->isgeneralized && eps->ishermitian)) {
    ierr = STGetBilinearForm(eps->st,&B);CHKERRQ(ierr);
    ierr = BVSetMatrix(eps->V,B,PetscNot(eps->ispositive));CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = BVSetMatrix(eps->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  EPS_Purify - purify the first k vectors in the V basis
*/
PETSC_STATIC_INLINE PetscErrorCode EPS_Purify(EPS eps,PetscInt k)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            v,z;

  PetscFunctionBegin;
  ierr = BVCreateVec(eps->V,&v);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = BVCopyVec(eps->V,i,v);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,i,&z);CHKERRQ(ierr);
    ierr = STApply(eps->st,v,z);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,i,&z);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_INTERN PetscErrorCode EPSSetWhichEigenpairs_Default(EPS);
SLEPC_INTERN PetscErrorCode EPSSetDimensions_Default(EPS,PetscInt,PetscInt*,PetscInt*);
SLEPC_INTERN PetscErrorCode EPSBackTransform_Default(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors_Hermitian(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors_Schur(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors_Indefinite(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors_Twosided(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeVectors_Slice(EPS);
SLEPC_INTERN PetscErrorCode EPSComputeResidualNorm_Private(EPS,PetscBool,PetscScalar,PetscScalar,Vec,Vec,Vec*,PetscReal*);
SLEPC_INTERN PetscErrorCode EPSComputeRitzVector(EPS,PetscScalar*,PetscScalar*,BV,Vec,Vec);
SLEPC_INTERN PetscErrorCode EPSGetStartVector(EPS,PetscInt,PetscBool*);
SLEPC_INTERN PetscErrorCode EPSGetLeftStartVector(EPS,PetscInt,PetscBool*);

/* Private functions of the solver implementations */

SLEPC_INTERN PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);
SLEPC_INTERN PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);
SLEPC_INTERN PetscErrorCode EPSKrylovConvergence(EPS,PetscBool,PetscInt,PetscInt,PetscReal,PetscReal,PetscReal,PetscInt*);
SLEPC_INTERN PetscErrorCode EPSPseudoLanczos(EPS,PetscReal*,PetscReal*,PetscReal*,PetscInt,PetscInt*,PetscBool*,PetscBool*,PetscReal*,Vec);
SLEPC_INTERN PetscErrorCode EPSBuildBalance_Krylov(EPS);
SLEPC_INTERN PetscErrorCode EPSSetDefaultST(EPS);
SLEPC_INTERN PetscErrorCode EPSSetDefaultST_Precond(EPS);
SLEPC_INTERN PetscErrorCode EPSSetDefaultST_GMRES(EPS);
SLEPC_INTERN PetscErrorCode EPSSetDefaultST_NoFactor(EPS);
SLEPC_INTERN PetscErrorCode EPSSetUpSort_Basic(EPS);
SLEPC_INTERN PetscErrorCode EPSSetUpSort_Default(EPS);

#endif
