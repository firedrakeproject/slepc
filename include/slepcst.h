
/*
      Spectral transformation module for eigenvalue problems.  
*/
#if !defined(__SLEPCST_H)
#define __SLEPCST_H
#include "petscksp.h"

extern int ST_COOKIE;

typedef struct _p_ST* ST;

#define STSHELL     "shell"
#define STSHIFT     "shift"
#define STSINV      "sinvert"
#define STCAYLEY    "cayley"
typedef char *STType;

extern int STCreate(MPI_Comm,ST*);
extern int STDestroy(ST);
extern int STSetType(ST,STType);
extern int STGetType(ST,STType*);
extern int STSetOperators(ST,Mat,Mat);
extern int STGetOperators(ST,Mat*,Mat*);
extern int STSetUp(ST);
extern int STSetFromOptions(ST);
extern int STView(ST,PetscViewer);

extern int STApply(ST,Vec,Vec);
extern int STApplyB(ST,Vec,Vec);
extern int STApplyNoB(ST,Vec,Vec);

extern PetscFList STList;
extern int STRegisterAll(char*);
extern int STRegisterDestroy(void);
extern int STRegister(char*,char*,char*,int(*)(ST));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,0)
#else
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,d)
#endif

extern int STSetKSP(ST,KSP);
extern int STGetKSP(ST,KSP*);
extern int STSetShift(ST,PetscScalar);
extern int STGetShift(ST,PetscScalar*);

extern int STSetOptionsPrefix(ST,char*);
extern int STAppendOptionsPrefix(ST,char*);
extern int STGetOptionsPrefix(ST,char**);

extern int STBackTransform(ST,PetscScalar*,PetscScalar*);

extern int STCheckNullSpace(ST,int,Vec*);

extern int STGetNumberLinearIterations(ST,int*);
extern int STResetNumberLinearIterations(ST);

typedef enum { STMATMODE_COPY, STMATMODE_INPLACE, 
               STMATMODE_SHELL } STMatMode;
extern int STSetMatMode(ST,STMatMode);
extern int STGetMatMode(ST,STMatMode*);
extern int STSetMatStructure(ST,MatStructure);

typedef enum { STINNER_HERMITIAN, STINNER_SYMMETRIC,
               STINNER_B_HERMITIAN, STINNER_B_SYMMETRIC } STBilinearForm;
extern int STSetBilinearForm(ST,STBilinearForm);

extern int STInnerProduct(ST st,Vec,Vec,PetscScalar*);
extern int STNorm(ST st,Vec,PetscReal*);

/* --------- options specific to particular spectral transformations-------- */

extern int STShellSetApply(ST, int (*)(void*,Vec,Vec), void*);
extern int STShellSetBackTransform(ST, int (*)(void*,PetscScalar*,PetscScalar*));
extern int STShellSetName(ST,char*);
extern int STShellGetName(ST,char**);

extern int STCayleySetAntishift(ST,PetscScalar);

#endif

