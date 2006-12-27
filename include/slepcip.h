#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H
#include "slepc.h"
PETSC_EXTERN_CXX_BEGIN

typedef enum { IP_MGS_ORTH,  IP_CGS_ORTH } IPOrthogonalizationType;
typedef enum { IP_ORTH_REFINE_NEVER, IP_ORTH_REFINE_IFNEEDED,
               IP_ORTH_REFINE_ALWAYS } IPOrthogonalizationRefinementType;

typedef struct _p_IP* IP;

EXTERN PetscErrorCode IPInitializePackage(char *path);
EXTERN PetscErrorCode IPCreate(MPI_Comm,IP*);
EXTERN PetscErrorCode IPAppendOptionsPrefix(IP,const char *);
EXTERN PetscErrorCode IPSetFromOptions(IP);
EXTERN PetscErrorCode IPSetOrthogonalization(IP,IPOrthogonalizationType,IPOrthogonalizationRefinementType,PetscReal);
EXTERN PetscErrorCode IPGetOrthogonalization(IP,IPOrthogonalizationType*,IPOrthogonalizationRefinementType*,PetscReal*);
EXTERN PetscErrorCode IPView(IP,PetscViewer);
EXTERN PetscErrorCode IPDestroy(IP);

EXTERN PetscErrorCode IPOrthogonalize(IP,int,PetscTruth*,Vec*,Vec,PetscScalar*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode IPOrthogonalizeGS(IP,int,PetscTruth*,Vec*,Vec,PetscScalar*,PetscReal*,PetscReal*,Vec);

typedef enum { IPINNER_HERMITIAN, IPINNER_SYMMETRIC,
               IPINNER_B_HERMITIAN, IPINNER_B_SYMMETRIC } IPBilinearForm;
EXTERN PetscErrorCode IPSetBilinearForm(IP,IPBilinearForm);
EXTERN PetscErrorCode IPGetBilinearForm(IP,IPBilinearForm*);

EXTERN PetscErrorCode IPInnerProduct(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPInnerProductBegin(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPInnerProductEnd(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPMInnerProduct(IP ip,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPMInnerProductBegin(IP ip,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPMInnerProductEnd(IP ip,PetscInt,Vec,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPNorm(IP ip,Vec,PetscReal*);
EXTERN PetscErrorCode IPNormBegin(IP ip,Vec,PetscReal*);
EXTERN PetscErrorCode IPNormEnd(IP ip,Vec,PetscReal*);

PETSC_EXTERN_CXX_END
#endif
