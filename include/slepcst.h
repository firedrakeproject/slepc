
/*
      Spectral transformation module for eigenvalue problems.  
*/
#if !defined(__SLEPCST_H)
#define __SLEPCST_H
#include "petscksp.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie ST_COOKIE;

typedef struct _p_ST* ST;

#define STSHELL     "shell"
#define STSHIFT     "shift"
#define STSINV      "sinvert"
#define STCAYLEY    "cayley"
#define STFOLD      "fold"
typedef char *STType;

EXTERN PetscErrorCode STCreate(MPI_Comm,ST*);
EXTERN PetscErrorCode STDestroy(ST);
EXTERN PetscErrorCode STSetType(ST,STType);
EXTERN PetscErrorCode STGetType(ST,STType*);
EXTERN PetscErrorCode STSetOperators(ST,Mat,Mat);
EXTERN PetscErrorCode STGetOperators(ST,Mat*,Mat*);
EXTERN PetscErrorCode STSetUp(ST);
EXTERN PetscErrorCode STSetFromOptions(ST);
EXTERN PetscErrorCode STView(ST,PetscViewer);

EXTERN PetscErrorCode STApply(ST,Vec,Vec);
EXTERN PetscErrorCode STApplyB(ST,Vec,Vec);
EXTERN PetscErrorCode STApplyNoB(ST,Vec,Vec);
EXTERN PetscErrorCode STApplyTranspose(ST,Vec,Vec);
EXTERN PetscErrorCode STComputeExplicitOperator(ST,Mat*);

extern PetscFList STList;
EXTERN PetscErrorCode STRegisterAll(char*);
EXTERN PetscErrorCode STRegisterDestroy(void);
EXTERN PetscErrorCode STRegister(char*,char*,char*,int(*)(ST));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,0)
#else
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode STSetKSP(ST,KSP);
EXTERN PetscErrorCode STGetKSP(ST,KSP*);
EXTERN PetscErrorCode STSetShift(ST,PetscScalar);
EXTERN PetscErrorCode STGetShift(ST,PetscScalar*);

EXTERN PetscErrorCode STSetOptionsPrefix(ST,char*);
EXTERN PetscErrorCode STAppendOptionsPrefix(ST,char*);
EXTERN PetscErrorCode STGetOptionsPrefix(ST,const char*[]);

EXTERN PetscErrorCode STBackTransform(ST,PetscScalar*,PetscScalar*);

EXTERN PetscErrorCode STCheckNullSpace(ST,int,Vec*);

EXTERN PetscErrorCode STGetNumberLinearIterations(ST,int*);
EXTERN PetscErrorCode STResetNumberLinearIterations(ST);

typedef enum { STMATMODE_COPY, STMATMODE_INPLACE, 
               STMATMODE_SHELL } STMatMode;
EXTERN PetscErrorCode STSetMatMode(ST,STMatMode);
EXTERN PetscErrorCode STGetMatMode(ST,STMatMode*);
EXTERN PetscErrorCode STSetMatStructure(ST,MatStructure);

typedef enum { STINNER_HERMITIAN, STINNER_SYMMETRIC,
               STINNER_B_HERMITIAN, STINNER_B_SYMMETRIC } STBilinearForm;
EXTERN PetscErrorCode STSetBilinearForm(ST,STBilinearForm);

EXTERN PetscErrorCode STInnerProduct(ST st,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode STInnerProductBegin(ST st,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode STInnerProductEnd(ST st,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode STMInnerProduct(ST st,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode STMInnerProductBegin(ST st,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode STMInnerProductEnd(ST st,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode STNorm(ST st,Vec,PetscReal*);
EXTERN PetscErrorCode STNormBegin(ST st,Vec,PetscReal*);
EXTERN PetscErrorCode STNormEnd(ST st,Vec,PetscReal*);

/* --------- options specific to particular spectral transformations-------- */

EXTERN PetscErrorCode STShellSetApply(ST, int (*)(void*,Vec,Vec), void*);
EXTERN PetscErrorCode STShellSetApplyTranspose(ST, int (*)(void*,Vec,Vec), void*);
EXTERN PetscErrorCode STShellSetBackTransform(ST, int (*)(void*,PetscScalar*,PetscScalar*));
EXTERN PetscErrorCode STShellSetName(ST,char*);
EXTERN PetscErrorCode STShellGetName(ST,char**);

EXTERN PetscErrorCode STCayleySetAntishift(ST,PetscScalar);

EXTERN PetscErrorCode STFoldSetLeftSide(ST st,PetscTruth left);

PETSC_EXTERN_CXX_END
#endif

