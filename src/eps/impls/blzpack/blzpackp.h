/*
   Private data structure used by the BLZPACK interface
*/

#if !defined(__BLZPACKP_H)
#define __BLZPACKP_H

#include "src/eps/epsimpl.h"

typedef struct {
  int         block_size;      /* block size */
  PetscReal   initial,final;   /* computational interval */
  int         slice;           /* use spectrum slicing */
  int         nsteps;          /* maximum number of steps per run */
  int         *istor;
  PetscReal   *rstor;
  PetscScalar *u;
  PetscScalar *v;
  PetscScalar *eig;
} EPS_BLZPACK;

/*
   Definition of routines from the BLZPACK package
*/

#include "slepcblaslapack.h"

/*
    These are real case, current version of BLZPACK only supports real
    matrices
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define BLZpack_ SLEPC_FORTRAN(blzdrs,BLZDRS)
#else 
#define BLZpack_ SLEPC_FORTRAN(blzdrd,BLZDRD)
#endif

#define BLZistorr_ SLEPC_FORTRAN(istorr,ISTORR)
#define BLZrstorr_ SLEPC_FORTRAN(rstorr,RSTORR)

EXTERN_C_BEGIN

extern void      BLZpack_(int*,PetscReal*,PetscScalar*,int*,PetscScalar*,
		          PetscScalar*,int*,int*,PetscScalar*,PetscScalar*);

extern int       BLZistorr_(int*,char*,int);
extern PetscReal BLZrstorr_(PetscReal*,char*,int);

EXTERN_C_END

#endif

