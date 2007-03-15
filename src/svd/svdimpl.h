#ifndef _SVDIMPL
#define _SVDIMPL

#include "slepcsvd.h"
#include "slepcip.h"

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

/*
     Maximum number of monitors you can run with a single SVD
*/
#define MAXSVDMONITORS 5 

/*
   Defines the SVD data structure.
*/
struct _p_SVD {
  PETSCHEADER(struct _SVDOps);
  Mat              OP;          /* problem matrix */
  Mat              A;	        /* problem matrix (m>n) */
  Mat              AT;          /* transposed matrix */
  SVDTransposeMode transmode;   /* transpose mode */
  PetscReal        *sigma;	/* singular values */
  Vec              *U,*V;	/* left and right singular vectors */
  Vec              vec_initial; /* initial vector */
  int              n;           /* maximun size of descomposition */
  SVDWhich         which;       /* which singular values are computed */
  int              nconv;	/* number of converged values */
  int              nsv;         /* number of requested values */
  int              ncv;         /* basis size */
  int              its;         /* iteration counter */
  int              max_it;      /* max iterations */
  PetscReal        tol;         /* tolerance */
  PetscReal        *errest;     /* error estimates */
  void             *data;	/* placeholder for misc stuff associated
                   		   with a particular solver */
  int              setupcalled;
  SVDConvergedReason reason;
  IP               ip;
  
  int  (*monitor[MAXSVDMONITORS])(SVD,int,int,PetscReal*,PetscReal*,int,void*);
  int  (*monitordestroy[MAXSVDMONITORS])(void*);
  void *monitorcontext[MAXSVDMONITORS];
  int  numbermonitors;
  
  int matvecs;
  int dots;
};

EXTERN PetscErrorCode SVDRegisterAll(char *);
EXTERN PetscErrorCode SVDRegister(const char*,const char*,const char*,int(*)(SVD));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,0)
#else
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,d)
#endif

#define SVDMonitor(svd,it,nconv,sigma,errest,nest) \
        { int _ierr,_i,_im = svd->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*svd->monitor[_i])(svd,it,nconv,sigma,errest,nest,svd->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

#endif

EXTERN PetscErrorCode SVDMatMult(SVD,PetscTruth,Vec,Vec);
EXTERN PetscErrorCode SVDMatGetVecs(SVD,Vec*,Vec*);
EXTERN PetscErrorCode SVDMatGetSize(SVD,PetscInt*,PetscInt*);
EXTERN PetscErrorCode SVDMatGetLocalSize(SVD,PetscInt*,PetscInt*);
EXTERN PetscErrorCode SVDTwoSideLanczos(SVD,PetscReal*,PetscReal*,Vec*,Vec,Vec*,int,int,PetscScalar*,Vec,Vec);

