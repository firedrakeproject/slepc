
/*
      Spectral transformation module for eigenvalue problems.  
*/
#if !defined(__SLEPCST_H)
#define __SLEPCST_H
#include "petscsles.h"

extern int ST_COOKIE;

typedef struct _p_ST* ST;

#define STNONE      "none"
#define STSHELL     "shell"
#define STSHIFT     "shift"
#define STSINV      "sinvert"
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

extern int STSetSLES(ST,SLES);
extern int STGetSLES(ST,SLES*);
extern int STSetShift(ST,PetscScalar);
extern int STGetShift(ST,PetscScalar*);
extern int STGetNumberOfShifts(ST,int*);

extern int STSetVector(ST,Vec);
extern int STGetVector(ST,Vec*);

extern int STSetOptionsPrefix(ST,char*);
extern int STAppendOptionsPrefix(ST,char*);
extern int STGetOptionsPrefix(ST,char**);

extern int STBackTransform(ST,PetscScalar*,PetscScalar*);
extern int STAssociatedSLESSolve(ST,Vec,Vec);

/* --------- options specific to particular spectral transformations-------- */

extern int STSinvertSetShiftMat(ST);
extern int STSinvertSetMatStructure(ST,MatStructure);

extern int STShellSetApply(ST, int (*)(void*,Vec,Vec), void*);
extern int STShellSetBackTransform(ST, int (*)(void*,PetscScalar*,PetscScalar*));
extern int STShellSetName(ST,char*);
extern int STShellGetName(ST,char**);

#endif

