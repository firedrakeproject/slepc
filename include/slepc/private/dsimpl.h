/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(_DSIMPL)
#define _DSIMPL

#include <slepcds.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool DSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode DSRegisterAll(void);
PETSC_EXTERN PetscLogEvent DS_Solve,DS_Vectors,DS_Synchronize,DS_Other;
PETSC_INTERN const char *DSMatName[];

typedef struct _DSOps *DSOps;

struct _DSOps {
  PetscErrorCode (*allocate)(DS,PetscInt);
  PetscErrorCode (*view)(DS,PetscViewer);
  PetscErrorCode (*vectors)(DS,DSMatType,PetscInt*,PetscReal*);
  PetscErrorCode (*solve[DS_MAX_SOLVE])(DS,PetscScalar*,PetscScalar*);
  PetscErrorCode (*sort)(DS,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*);
  PetscErrorCode (*truncate)(DS,PetscInt);
  PetscErrorCode (*update)(DS);
  PetscErrorCode (*cond)(DS,PetscReal*);
  PetscErrorCode (*transharm)(DS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
  PetscErrorCode (*transrks)(DS,PetscScalar);
  PetscErrorCode (*destroy)(DS);
  PetscErrorCode (*matgetsize)(DS,DSMatType,PetscInt*,PetscInt*);
  PetscErrorCode (*hermitian)(DS,DSMatType,PetscBool*);
  PetscErrorCode (*synchronize)(DS,PetscScalar*,PetscScalar*);
};

struct _p_DS {
  PETSCHEADER(struct _DSOps);
  /*------------------------- User parameters --------------------------*/
  DSStateType    state;              /* the current state */
  PetscInt       method;             /* identifies the variant to be used */
  PetscBool      compact;            /* whether the matrices are stored in compact form */
  PetscBool      refined;            /* get refined vectors instead of regular vectors */
  PetscBool      extrarow;           /* assume the matrix dimension is (n+1) x n */
  PetscInt       ld;                 /* leading dimension */
  PetscInt       l;                  /* number of locked (inactive) leading columns */
  PetscInt       n;                  /* current dimension */
  PetscInt       m;                  /* current column dimension (for SVD only) */
  PetscInt       k;                  /* intermediate dimension (e.g. position of arrow) */
  PetscInt       t;                  /* length of decomposition when it was truncated */
  PetscInt       bs;                 /* block size */
  SlepcSC        sc;                 /* sorting criterion */
  DSParallelType pmode;              /* parallel mode (redundant or synchronized) */

  /*----------------- Status variables and working data ----------------*/
  PetscScalar    *mat[DS_NUM_MAT];   /* the matrices */
  PetscReal      *rmat[DS_NUM_MAT];  /* the matrices (real) */
  Mat            omat[DS_NUM_MAT];   /* the matrices (PETSc object) */
  PetscInt       *perm;              /* permutation */
  void           *data;              /* placeholder for solver-specific stuff */
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *iwork;
  PetscInt       lwork,lrwork,liwork;
};

/*
    Macros to test valid DS arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define DSCheckAlloc(h,arg) do {} while (0)
#define DSCheckSolved(h,arg) do {} while (0)
#define DSCheckValidMat(ds,m,arg) do {} while (0)
#define DSCheckValidMatReal(ds,m,arg) do {} while (0)

#else

#define DSCheckAlloc(h,arg) \
  do { \
    if (!h->ld) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"Must call DSAllocate() first: Parameter #%d",arg); \
  } while (0)

#define DSCheckSolved(h,arg) \
  do { \
    if (h->state<DS_STATE_CONDENSED) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"Must call DSSolve() first: Parameter #%d",arg); \
  } while (0)

#define DSCheckValidMat(ds,m,arg) \
  do { \
    if (m>=DS_NUM_MAT) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Invalid matrix: Parameter #%d",arg); \
    if (!ds->mat[m]) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this DS: Parameter #%d",arg); \
  } while (0)

#define DSCheckValidMatReal(ds,m,arg) \
  do { \
    if (m>=DS_NUM_MAT) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONG,"Invalid matrix: Parameter #%d",arg); \
    if (!ds->rmat[m]) SETERRQ1(PetscObjectComm((PetscObject)ds),PETSC_ERR_ARG_WRONGSTATE,"Requested matrix was not created in this DS: Parameter #%d",arg); \
  } while (0)

#endif

PETSC_INTERN PetscErrorCode DSAllocateMatrix_Private(DS,DSMatType,PetscBool);
PETSC_STATIC_INLINE PetscErrorCode DSAllocateMat_Private(DS ds,DSMatType m) {return DSAllocateMatrix_Private(ds,m,PETSC_FALSE);}
PETSC_STATIC_INLINE PetscErrorCode DSAllocateMatReal_Private(DS ds,DSMatType m) {return DSAllocateMatrix_Private(ds,m,PETSC_TRUE);}
PETSC_INTERN PetscErrorCode DSAllocateWork_Private(DS,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode DSSortEigenvalues_Private(DS,PetscScalar*,PetscScalar*,PetscInt*,PetscBool);
PETSC_INTERN PetscErrorCode DSSortEigenvaluesReal_Private(DS,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode DSPermuteColumns_Private(DS,PetscInt,PetscInt,DSMatType,PetscInt*);
PETSC_INTERN PetscErrorCode DSPermuteRows_Private(DS,PetscInt,PetscInt,DSMatType,PetscInt*);
PETSC_INTERN PetscErrorCode DSPermuteBoth_Private(DS,PetscInt,PetscInt,DSMatType,DSMatType,PetscInt*);
PETSC_INTERN PetscErrorCode DSCopyMatrix_Private(DS,DSMatType,DSMatType);

PETSC_INTERN PetscErrorCode DSGHIEPOrthogEigenv(DS,DSMatType,PetscScalar*,PetscScalar*,PetscBool);
PETSC_INTERN PetscErrorCode DSGHIEPComplexEigs(DS,PetscInt,PetscInt,PetscScalar*,PetscScalar*);
PETSC_INTERN PetscErrorCode DSGHIEPInverseIteration(DS,PetscScalar*,PetscScalar*);
PETSC_INTERN PetscErrorCode DSIntermediate_GHIEP(DS);
PETSC_INTERN PetscErrorCode DSSwitchFormat_GHIEP(DS,PetscBool);
PETSC_INTERN PetscErrorCode DSGHIEPRealBlocks(DS);
PETSC_INTERN PetscErrorCode DSSolve_GHIEP_HZ(DS,PetscScalar*,PetscScalar*);

PETSC_INTERN PetscErrorCode BDC_dibtdc_(const char*,PetscBLASInt,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscBLASInt,PetscBLASInt,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscReal,PetscReal*,PetscReal*,PetscBLASInt,PetscReal*,PetscBLASInt,PetscBLASInt*,PetscBLASInt,PetscBLASInt*,PetscBLASInt);
PETSC_INTERN PetscErrorCode BDC_dlaed3m_(const char*,const char*,PetscBLASInt,PetscBLASInt,PetscBLASInt,PetscReal*,PetscReal*,PetscBLASInt,PetscReal,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_INTERN PetscErrorCode BDC_dmerg2_(const char*,PetscBLASInt,PetscBLASInt,PetscReal*,PetscReal*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt,PetscReal*,PetscBLASInt,PetscBLASInt,PetscReal*,PetscBLASInt,PetscBLASInt*,PetscReal,PetscBLASInt*,PetscBLASInt);
PETSC_INTERN PetscErrorCode BDC_dsbtdc_(const char*,const char*,PetscBLASInt,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscBLASInt,PetscBLASInt,PetscReal*,PetscBLASInt,PetscBLASInt,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscBLASInt,PetscReal*,PetscBLASInt,PetscBLASInt*,PetscBLASInt,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt,PetscBLASInt);
PETSC_INTERN PetscErrorCode BDC_dsrtdf_(PetscBLASInt*,PetscBLASInt,PetscBLASInt,PetscReal*,PetscReal*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#endif
