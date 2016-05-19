/*
   User interface for SLEPc's polynomial eigenvalue solvers.

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

#if !defined(__SLEPCPEP_H)
#define __SLEPCPEP_H
#include <slepceps.h>

PETSC_EXTERN PetscErrorCode PEPInitializePackage(void);

/*S
     PEP - Abstract SLEPc object that manages all the polynomial eigenvalue
     problem solvers.

   Level: beginner

.seealso:  PEPCreate()
S*/
typedef struct _p_PEP* PEP;

/*J
    PEPType - String with the name of a polynomial eigensolver

   Level: beginner

.seealso: PEPSetType(), PEP
J*/
typedef const char* PEPType;
#define PEPLINEAR    "linear"
#define PEPQARNOLDI  "qarnoldi"
#define PEPTOAR      "toar"
#define PEPSTOAR     "stoar"
#define PEPJD        "jd"

/* Logging support */
PETSC_EXTERN PetscClassId PEP_CLASSID;

/*E
    PEPProblemType - Determines the type of the polynomial eigenproblem

    Level: intermediate

.seealso: PEPSetProblemType(), PEPGetProblemType()
E*/
typedef enum { PEP_GENERAL=1,
               PEP_HERMITIAN,   /* All A_i  Hermitian */
               PEP_GYROSCOPIC   /* QEP with M, K  Hermitian, M>0, C skew-Hermitian */
             } PEPProblemType;

/*E
    PEPWhich - Determines which part of the spectrum is requested

    Level: intermediate

.seealso: PEPSetWhichEigenpairs(), PEPGetWhichEigenpairs()
E*/
typedef enum { PEP_LARGEST_MAGNITUDE=1,
               PEP_SMALLEST_MAGNITUDE,
               PEP_LARGEST_REAL,
               PEP_SMALLEST_REAL,
               PEP_LARGEST_IMAGINARY,
               PEP_SMALLEST_IMAGINARY,
               PEP_TARGET_MAGNITUDE,
               PEP_TARGET_REAL,
               PEP_TARGET_IMAGINARY,
               PEP_WHICH_USER } PEPWhich;

/*E
    PEPBasis - The type of polynomial basis used to represent the polynomial
    eigenproblem

    Level: intermediate

.seealso: PEPSetBasis()
E*/
typedef enum { PEP_BASIS_MONOMIAL,
               PEP_BASIS_CHEBYSHEV1,
               PEP_BASIS_CHEBYSHEV2,
               PEP_BASIS_LEGENDRE,
               PEP_BASIS_LAGUERRE,
               PEP_BASIS_HERMITE } PEPBasis;
PETSC_EXTERN const char *PEPBasisTypes[];

/*E
    PEPScale - The scaling strategy

    Level: intermediate

.seealso: PEPSetScale()
E*/
typedef enum { PEP_SCALE_NONE,
               PEP_SCALE_SCALAR,
               PEP_SCALE_DIAGONAL,
               PEP_SCALE_BOTH } PEPScale;
PETSC_EXTERN const char *PEPScaleTypes[];

/*E
    PEPRefine - The refinement type

    Level: intermediate

.seealso: PEPSetRefine()
E*/
typedef enum { PEP_REFINE_NONE,
               PEP_REFINE_SIMPLE,
               PEP_REFINE_MULTIPLE } PEPRefine;
PETSC_EXTERN const char *PEPRefineTypes[];

/*E
    PEPRefineScheme - The scheme used for solving linear systems during iterative refinement

    Level: intermediate

.seealso: PEPSetRefine()
E*/
typedef enum { PEP_REFINE_SCHEME_SCHUR=1,
               PEP_REFINE_SCHEME_MBE,
               PEP_REFINE_SCHEME_EXPLICIT } PEPRefineScheme;
PETSC_EXTERN const char *PEPRefineSchemes[];

/*E
    PEPExtract - The extraction type

    Level: intermediate

.seealso: PEPSetExtract()
E*/
typedef enum { PEP_EXTRACT_NONE=1,
               PEP_EXTRACT_NORM,
               PEP_EXTRACT_RESIDUAL,
               PEP_EXTRACT_STRUCTURED } PEPExtract;
PETSC_EXTERN const char *PEPExtractTypes[];

/*E
    PEPErrorType - The error type used to assess accuracy of computed solutions

    Level: intermediate

.seealso: PEPComputeError()
E*/
typedef enum { PEP_ERROR_ABSOLUTE,
               PEP_ERROR_RELATIVE,
               PEP_ERROR_BACKWARD } PEPErrorType;
PETSC_EXTERN const char *PEPErrorTypes[];

/*E
    PEPConv - Determines the convergence test

    Level: intermediate

.seealso: PEPSetConvergenceTest(), PEPSetConvergenceTestFunction()
E*/
typedef enum { PEP_CONV_ABS,
               PEP_CONV_REL,
               PEP_CONV_NORM,
               PEP_CONV_USER } PEPConv;

/*E
    PEPStop - Determines the stopping test

    Level: advanced

.seealso: PEPSetStoppingTest(), PEPSetStoppingTestFunction()
E*/
typedef enum { PEP_STOP_BASIC,
               PEP_STOP_USER } PEPStop;

/*E
    PEPConvergedReason - Reason an eigensolver was said to
         have converged or diverged

    Level: intermediate

.seealso: PEPSolve(), PEPGetConvergedReason(), PEPSetTolerances()
E*/
typedef enum {/* converged */
              PEP_CONVERGED_TOL                =  1,
              PEP_CONVERGED_USER               =  2,
              /* diverged */
              PEP_DIVERGED_ITS                 = -1,
              PEP_DIVERGED_BREAKDOWN           = -2,
              PEP_DIVERGED_SYMMETRY_LOST       = -3,
              PEP_CONVERGED_ITERATING          =  0} PEPConvergedReason;
PETSC_EXTERN const char *const*PEPConvergedReasons;

PETSC_EXTERN PetscErrorCode PEPCreate(MPI_Comm,PEP*);
PETSC_EXTERN PetscErrorCode PEPDestroy(PEP*);
PETSC_EXTERN PetscErrorCode PEPReset(PEP);
PETSC_EXTERN PetscErrorCode PEPSetType(PEP,PEPType);
PETSC_EXTERN PetscErrorCode PEPGetType(PEP,PEPType*);
PETSC_EXTERN PetscErrorCode PEPSetProblemType(PEP,PEPProblemType);
PETSC_EXTERN PetscErrorCode PEPGetProblemType(PEP,PEPProblemType*);
PETSC_EXTERN PetscErrorCode PEPSetOperators(PEP,PetscInt,Mat[]);
PETSC_EXTERN PetscErrorCode PEPGetOperators(PEP,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode PEPGetNumMatrices(PEP,PetscInt*);
PETSC_EXTERN PetscErrorCode PEPSetTarget(PEP,PetscScalar);
PETSC_EXTERN PetscErrorCode PEPGetTarget(PEP,PetscScalar*);
PETSC_EXTERN PetscErrorCode PEPSetFromOptions(PEP);
PETSC_EXTERN PetscErrorCode PEPSetUp(PEP);
PETSC_EXTERN PetscErrorCode PEPSolve(PEP);
PETSC_EXTERN PetscErrorCode PEPView(PEP,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode PEPViewFromOptions(PEP pep,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)pep,obj,name);}
PETSC_EXTERN PetscErrorCode PEPErrorView(PEP,PEPErrorType,PetscViewer);
PETSC_DEPRECATED("Use PEPErrorView()") PETSC_STATIC_INLINE PetscErrorCode PEPPrintSolution(PEP pep,PetscViewer v) {return PEPErrorView(pep,PEP_ERROR_BACKWARD,v);}
PETSC_EXTERN PetscErrorCode PEPErrorViewFromOptions(PEP);
PETSC_EXTERN PetscErrorCode PEPReasonView(PEP,PetscViewer);
PETSC_EXTERN PetscErrorCode PEPReasonViewFromOptions(PEP);
PETSC_EXTERN PetscErrorCode PEPValuesView(PEP,PetscViewer);
PETSC_EXTERN PetscErrorCode PEPValuesViewFromOptions(PEP);
PETSC_EXTERN PetscErrorCode PEPVectorsView(PEP,PetscViewer);
PETSC_EXTERN PetscErrorCode PEPVectorsViewFromOptions(PEP);
PETSC_EXTERN PetscErrorCode PEPSetBV(PEP,BV);
PETSC_EXTERN PetscErrorCode PEPGetBV(PEP,BV*);
PETSC_EXTERN PetscErrorCode PEPSetRG(PEP,RG);
PETSC_EXTERN PetscErrorCode PEPGetRG(PEP,RG*);
PETSC_EXTERN PetscErrorCode PEPSetDS(PEP,DS);
PETSC_EXTERN PetscErrorCode PEPGetDS(PEP,DS*);
PETSC_EXTERN PetscErrorCode PEPSetST(PEP,ST);
PETSC_EXTERN PetscErrorCode PEPGetST(PEP,ST*);
PETSC_EXTERN PetscErrorCode PEPRefineGetKSP(PEP,KSP*);

PETSC_EXTERN PetscErrorCode PEPSetTolerances(PEP,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode PEPGetTolerances(PEP,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode PEPSetConvergenceTestFunction(PEP,PetscErrorCode (*)(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode PEPSetConvergenceTest(PEP,PEPConv);
PETSC_EXTERN PetscErrorCode PEPGetConvergenceTest(PEP,PEPConv*);
PETSC_EXTERN PetscErrorCode PEPConvergedAbsolute(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode PEPConvergedRelative(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode PEPConvergedNorm(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode PEPSetStoppingTestFunction(PEP,PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscInt,PetscInt,PEPConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode PEPSetStoppingTest(PEP,PEPStop);
PETSC_EXTERN PetscErrorCode PEPGetStoppingTest(PEP,PEPStop*);
PETSC_EXTERN PetscErrorCode PEPStoppingBasic(PEP,PetscInt,PetscInt,PetscInt,PetscInt,PEPConvergedReason*,void*);
PETSC_EXTERN PetscErrorCode PEPGetConvergedReason(PEP,PEPConvergedReason *);

PETSC_EXTERN PetscErrorCode PEPSetDimensions(PEP,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PEPGetDimensions(PEP,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PEPSetScale(PEP,PEPScale,PetscReal,Vec,Vec,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode PEPGetScale(PEP,PEPScale*,PetscReal*,Vec*,Vec*,PetscInt*,PetscReal*);
PETSC_EXTERN PetscErrorCode PEPSetRefine(PEP,PEPRefine,PetscInt,PetscReal,PetscInt,PEPRefineScheme);
PETSC_EXTERN PetscErrorCode PEPGetRefine(PEP,PEPRefine*,PetscInt*,PetscReal*,PetscInt*,PEPRefineScheme*);
PETSC_EXTERN PetscErrorCode PEPSetExtract(PEP,PEPExtract);
PETSC_EXTERN PetscErrorCode PEPGetExtract(PEP,PEPExtract*);
PETSC_EXTERN PetscErrorCode PEPSetBasis(PEP,PEPBasis);
PETSC_EXTERN PetscErrorCode PEPGetBasis(PEP,PEPBasis*);

PETSC_EXTERN PetscErrorCode PEPGetConverged(PEP,PetscInt*);
PETSC_EXTERN PetscErrorCode PEPGetEigenpair(PEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
PETSC_EXTERN PetscErrorCode PEPComputeError(PEP,PetscInt,PEPErrorType,PetscReal*);
PETSC_DEPRECATED("Use PEPComputeError()") PETSC_STATIC_INLINE PetscErrorCode PEPComputeRelativeError(PEP pep,PetscInt i,PetscReal *r) {return PEPComputeError(pep,i,PEP_ERROR_BACKWARD,r);}
PETSC_DEPRECATED("Use PEPComputeError() with PEP_ERROR_ABSOLUTE") PETSC_STATIC_INLINE PetscErrorCode PEPComputeResidualNorm(PEP pep,PetscInt i,PetscReal *r) {return PEPComputeError(pep,i,PEP_ERROR_ABSOLUTE,r);}
PETSC_EXTERN PetscErrorCode PEPGetErrorEstimate(PEP,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode PEPMonitor(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode PEPMonitorSet(PEP,PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode PEPMonitorSetFromOptions(PEP,const char*,const char*,const char*,PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscBool);
PETSC_EXTERN PetscErrorCode PEPConvMonitorSetFromOptions(PEP,const char*,const char*,const char*,PetscErrorCode (*)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor));
PETSC_EXTERN PetscErrorCode PEPMonitorCancel(PEP);
PETSC_EXTERN PetscErrorCode PEPGetMonitorContext(PEP,void **);
PETSC_EXTERN PetscErrorCode PEPGetIterationNumber(PEP,PetscInt*);

PETSC_EXTERN PetscErrorCode PEPSetInitialSpace(PEP,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode PEPSetWhichEigenpairs(PEP,PEPWhich);
PETSC_EXTERN PetscErrorCode PEPGetWhichEigenpairs(PEP,PEPWhich*);
PETSC_EXTERN PetscErrorCode PEPSetEigenvalueComparison(PEP,PetscErrorCode (*func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);

PETSC_EXTERN PetscErrorCode PEPMonitorAll(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode PEPMonitorFirst(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode PEPMonitorConverged(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor);
PETSC_EXTERN PetscErrorCode PEPMonitorLGCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode PEPMonitorLG(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode PEPMonitorLGAll(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode PEPSetTrackAll(PEP,PetscBool);
PETSC_EXTERN PetscErrorCode PEPGetTrackAll(PEP,PetscBool*);

PETSC_EXTERN PetscErrorCode PEPSetOptionsPrefix(PEP,const char*);
PETSC_EXTERN PetscErrorCode PEPAppendOptionsPrefix(PEP,const char*);
PETSC_EXTERN PetscErrorCode PEPGetOptionsPrefix(PEP,const char*[]);

PETSC_EXTERN PetscFunctionList PEPList;
PETSC_EXTERN PetscErrorCode PEPRegister(const char[],PetscErrorCode(*)(PEP));

PETSC_EXTERN PetscErrorCode PEPSetWorkVecs(PEP,PetscInt);
PETSC_EXTERN PetscErrorCode PEPAllocateSolution(PEP,PetscInt);

/* --------- options specific to particular eigensolvers -------- */

PETSC_EXTERN PetscErrorCode PEPLinearSetCompanionForm(PEP,PetscInt);
PETSC_EXTERN PetscErrorCode PEPLinearGetCompanionForm(PEP,PetscInt*);
PETSC_EXTERN PetscErrorCode PEPLinearSetExplicitMatrix(PEP,PetscBool);
PETSC_EXTERN PetscErrorCode PEPLinearGetExplicitMatrix(PEP,PetscBool*);
PETSC_EXTERN PetscErrorCode PEPLinearSetEPS(PEP,EPS);
PETSC_EXTERN PetscErrorCode PEPLinearGetEPS(PEP,EPS*);

PETSC_EXTERN PetscErrorCode PEPQArnoldiSetRestart(PEP,PetscReal);
PETSC_EXTERN PetscErrorCode PEPQArnoldiGetRestart(PEP,PetscReal*);
PETSC_EXTERN PetscErrorCode PEPQArnoldiSetLocking(PEP,PetscBool);
PETSC_EXTERN PetscErrorCode PEPQArnoldiGetLocking(PEP,PetscBool*);

PETSC_EXTERN PetscErrorCode PEPTOARSetRestart(PEP,PetscReal);
PETSC_EXTERN PetscErrorCode PEPTOARGetRestart(PEP,PetscReal*);
PETSC_EXTERN PetscErrorCode PEPTOARSetLocking(PEP,PetscBool);
PETSC_EXTERN PetscErrorCode PEPTOARGetLocking(PEP,PetscBool*);

PETSC_EXTERN PetscErrorCode PEPSTOARSetLocking(PEP,PetscBool);
PETSC_EXTERN PetscErrorCode PEPSTOARGetLocking(PEP,PetscBool*);

PETSC_EXTERN PetscErrorCode PEPJDSetRestart(PEP,PetscReal);
PETSC_EXTERN PetscErrorCode PEPJDGetRestart(PEP,PetscReal*);

#endif

