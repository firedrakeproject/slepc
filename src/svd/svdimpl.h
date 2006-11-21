#ifndef _SVDIMPL
#define _SVDIMPL

#include "slepcsvd.h"

extern PetscFList SVDList;
extern PetscEvent SVD_SetUp, SVD_Solve;

typedef struct _SVDOps *SVDOps;

struct _SVDOps {
  int  (*solve)(SVD);
  int  (*setup)(SVD);
  int  (*setfromoptions)(SVD);
  int  (*publishoptions)(SVD);
  int  (*destroy)(SVD);
  int  (*view)(SVD,PetscViewer);
};

struct _p_SVD {
  PETSCHEADER(struct _SVDOps);
  Mat       A;		 /* problem matrix */
  PetscReal *sigma;      /* singular values */
  int       nconv;       /* number of converged values */
  void      *data;	 /* placeholder for misc stuff associated 
        		    with a particular solver */
  int       setupcalled;
};

EXTERN PetscErrorCode SVDRegisterAll(char *);
EXTERN PetscErrorCode SVDRegister(const char*,const char*,const char*,int(*)(SVD));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,0)
#else
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,d)
#endif

#endif
