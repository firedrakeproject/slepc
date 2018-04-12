/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   User interface for the SLEPc object for solving linear matrix equations
*/

#if !defined(__SLEPCLME_H)
#define __SLEPCLME_H
#include <slepcbv.h>

PETSC_EXTERN PetscErrorCode LMEInitializePackage(void);

/*S
    LME - SLEPc object that encapsulates functionality for linear matrix equations

    Level: beginner

.seealso:  LMECreate()
S*/
typedef struct _p_LME* LME;

/*J
    LMEType - String with the name of a method for solving linear matrix equations

    Level: beginner

.seealso: LMESetType(), LME
J*/
typedef const char* LMEType;
#define LMEKRYLOV   "krylov"

/* Logging support */
PETSC_EXTERN PetscClassId LME_CLASSID;

/*E
    LMEProblemType - Determines the type of linear matrix equation

    Level: beginner

.seealso: LMESetProblemType(), LMEGetProblemType()
E*/
typedef enum { LME_LYAPUNOV,
               LME_SYLVESTER,
               LME_GEN_LYAPUNOV,
               LME_GEN_SYLVESTER,
               LME_DT_LYAPUNOV ,
               LME_STEIN} LMEProblemType;
PETSC_EXTERN const char *LMEProblemTypes[];

PETSC_EXTERN PetscErrorCode LMECreate(MPI_Comm,LME *);
PETSC_EXTERN PetscErrorCode LMEDestroy(LME*);
PETSC_EXTERN PetscErrorCode LMEReset(LME);
PETSC_EXTERN PetscErrorCode LMESetType(LME,LMEType);
PETSC_EXTERN PetscErrorCode LMEGetType(LME,LMEType*);
PETSC_EXTERN PetscErrorCode LMESetProblemType(LME,LMEProblemType);
PETSC_EXTERN PetscErrorCode LMEGetProblemType(LME,LMEProblemType*);
PETSC_EXTERN PetscErrorCode LMESetCoefficients(LME,Mat,Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode LMEGetCoefficients(LME,Mat*,Mat*,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode LMESetRHS(LME,Mat);
PETSC_EXTERN PetscErrorCode LMEGetRHS(LME,Mat*);
PETSC_EXTERN PetscErrorCode LMESetSolution(LME,Mat);
PETSC_EXTERN PetscErrorCode LMEGetSolution(LME,Mat*);
PETSC_EXTERN PetscErrorCode LMESetFromOptions(LME);
PETSC_EXTERN PetscErrorCode LMESetUp(LME);
PETSC_EXTERN PetscErrorCode LMESolve(LME);
PETSC_EXTERN PetscErrorCode LMEView(LME,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode LMEViewFromOptions(LME lme,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)lme,obj,name);}
PETSC_EXTERN PetscErrorCode LMEReasonView(LME,PetscViewer);
PETSC_EXTERN PetscErrorCode LMEReasonViewFromOptions(LME);

PETSC_EXTERN PetscErrorCode LMESetBV(LME,BV);
PETSC_EXTERN PetscErrorCode LMEGetBV(LME,BV*);
PETSC_EXTERN PetscErrorCode LMESetTolerances(LME,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode LMEGetTolerances(LME,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode LMESetDimensions(LME,PetscInt);
PETSC_EXTERN PetscErrorCode LMEGetDimensions(LME,PetscInt*);

PETSC_EXTERN PetscErrorCode LMEMonitor(LME,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode LMEMonitorSet(LME,PetscErrorCode (*)(LME,PetscInt,PetscReal,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode LMEMonitorSetFromOptions(LME,const char*,const char*,const char*,PetscErrorCode (*)(LME,PetscInt,PetscReal,PetscViewerAndFormat*));
PETSC_EXTERN PetscErrorCode LMEMonitorCancel(LME);
PETSC_EXTERN PetscErrorCode LMEGetMonitorContext(LME,void **);
PETSC_EXTERN PetscErrorCode LMEGetIterationNumber(LME,PetscInt*);

PETSC_EXTERN PetscErrorCode LMEGetErrorEstimate(LME,PetscReal*);
PETSC_EXTERN PetscErrorCode LMEComputeError(LME,PetscReal*);
PETSC_EXTERN PetscErrorCode LMESetErrorIfNotConverged(LME,PetscBool);
PETSC_EXTERN PetscErrorCode LMEGetErrorIfNotConverged(LME,PetscBool*);

PETSC_EXTERN PetscErrorCode LMEDenseLyapunovChol(LME,PetscScalar*,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode LMEMonitorDefault(LME,PetscInt,PetscReal,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode LMEMonitorLGCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode LMEMonitorLG(LME,PetscInt,PetscReal,void*);

PETSC_EXTERN PetscErrorCode LMESetOptionsPrefix(LME,const char*);
PETSC_EXTERN PetscErrorCode LMEAppendOptionsPrefix(LME,const char*);
PETSC_EXTERN PetscErrorCode LMEGetOptionsPrefix(LME,const char*[]);

/*E
    LMEConvergedReason - reason a matrix function iteration was said to
         have converged or diverged

    Level: intermediate

.seealso: LMESolve(), LMEGetConvergedReason(), LMESetTolerances()
E*/
typedef enum {/* converged */
              LME_CONVERGED_TOL                =  1,
              /* diverged */
              LME_DIVERGED_ITS                 = -1,
              LME_DIVERGED_BREAKDOWN           = -2,
              LME_CONVERGED_ITERATING          =  0} LMEConvergedReason;
PETSC_EXTERN const char *const*LMEConvergedReasons;

PETSC_EXTERN PetscErrorCode LMEGetConvergedReason(LME,LMEConvergedReason *);

PETSC_EXTERN PetscFunctionList LMEList;
PETSC_EXTERN PetscErrorCode LMERegister(const char[],PetscErrorCode(*)(LME));

PETSC_EXTERN PetscErrorCode LMEAllocateSolution(LME,PetscInt);

#endif

