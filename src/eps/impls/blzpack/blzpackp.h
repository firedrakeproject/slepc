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

#if defined(SLEPC_BLZPACK_HAVE_UNDERSCORE)
#define SLEPC_BLZPACK(lcase,ucase) lcase##_
#elif defined(SLEPC_BLZPACK_HAVE_CAPS)
#define SLEPC_BLZPACK(lcase,ucase) ucase
#else
#define SLEPC_BLZPACK(lcase,ucase) lcase
#endif

/*
    These are real case, current version of BLZPACK only supports real
    matrices
*/

#if defined(PETSC_USE_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define BLZpack_ SLEPC_BLZPACK(blzdrs,BLZDRS)
#else 
#define BLZpack_ SLEPC_BLZPACK(blzdrd,BLZDRD)
#endif

#define BLZistorr_ SLEPC_BLZPACK(istorr,ISTORR)
#define BLZrstorr_ SLEPC_BLZPACK(rstorr,RSTORR)

EXTERN_C_BEGIN

EXTERN void	 BLZpack_(int*,PetscReal*,PetscScalar*,int*,PetscScalar*,
        		  PetscScalar*,int*,int*,PetscScalar*,PetscScalar*);

EXTERN int	 BLZistorr_(int*,char*,int);
EXTERN PetscReal BLZrstorr_(PetscReal*,char*,int);

EXTERN_C_END

#endif

