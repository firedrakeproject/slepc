/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   User interface for the region object in SLEPc
*/

#if !defined(__SLEPCRG_H)
#define __SLEPCRG_H
#include <slepcsys.h>
#include <slepcrgtypes.h>

PETSC_EXTERN PetscErrorCode RGInitializePackage(void);

/*J
   RGType - String with the name of the region.

   Level: beginner

.seealso: RGSetType(), RG
J*/
typedef const char* RGType;
#define RGINTERVAL  "interval"
#define RGPOLYGON   "polygon"
#define RGELLIPSE   "ellipse"
#define RGRING      "ring"

/* Logging support */
PETSC_EXTERN PetscClassId RG_CLASSID;

PETSC_EXTERN PetscErrorCode RGCreate(MPI_Comm,RG*);
PETSC_EXTERN PetscErrorCode RGSetType(RG,RGType);
PETSC_EXTERN PetscErrorCode RGGetType(RG,RGType*);
PETSC_EXTERN PetscErrorCode RGSetOptionsPrefix(RG,const char *);
PETSC_EXTERN PetscErrorCode RGAppendOptionsPrefix(RG,const char *);
PETSC_EXTERN PetscErrorCode RGGetOptionsPrefix(RG,const char *[]);
PETSC_EXTERN PetscErrorCode RGSetFromOptions(RG);
PETSC_EXTERN PetscErrorCode RGView(RG,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode RGViewFromOptions(RG rg,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)rg,obj,name);}
PETSC_EXTERN PetscErrorCode RGDestroy(RG*);

PETSC_EXTERN PetscErrorCode RGIsTrivial(RG,PetscBool*);
PETSC_EXTERN PetscErrorCode RGSetComplement(RG,PetscBool);
PETSC_EXTERN PetscErrorCode RGGetComplement(RG,PetscBool*);
PETSC_EXTERN PetscErrorCode RGSetScale(RG,PetscReal);
PETSC_EXTERN PetscErrorCode RGGetScale(RG,PetscReal*);
PETSC_EXTERN PetscErrorCode RGPushScale(RG,PetscReal);
PETSC_EXTERN PetscErrorCode RGPopScale(RG);
PETSC_EXTERN PetscErrorCode RGCheckInside(RG,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
PETSC_EXTERN PetscErrorCode RGComputeContour(RG,PetscInt,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode RGComputeBoundingBox(RG,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscFunctionList RGList;
PETSC_EXTERN PetscErrorCode RGRegister(const char[],PetscErrorCode(*)(RG));

/* --------- options specific to particular regions -------- */

PETSC_EXTERN PetscErrorCode RGEllipseSetParameters(RG,PetscScalar,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode RGEllipseGetParameters(RG,PetscScalar*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode RGIntervalSetEndpoints(RG,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode RGIntervalGetEndpoints(RG,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode RGPolygonSetVertices(RG,PetscInt,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode RGPolygonGetVertices(RG,PetscInt*,PetscScalar**,PetscScalar**);

PETSC_EXTERN PetscErrorCode RGRingSetParameters(RG,PetscScalar,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode RGRingGetParameters(RG,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

#endif
