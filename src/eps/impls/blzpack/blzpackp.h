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

#include "petsc.h"

/*
   This include file on the Cray T3D/T3E defines the interface between 
  Fortran and C representations of character strings.
*/
#if defined(PETSC_USES_CPTOFCD)
#include <fortran.h>
#endif

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case, current version of BLZPACK only supports real
    matrices
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define BLZDRD   BLZDRS 
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define BLZpack_   blzdrd_
#define BLZistorr_ istorr_
#define BLZrstorr_ rstorr_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define BLZpack_   BLZDRD
#define BLZistorr_ ISTORR
#define BLZrstorr_ RSTORR
#else
#define BLZpack_   blzdrd
#define BLZistorr_ istorr
#define BLZrstorr_ rstorr
#endif

#endif

EXTERN_C_BEGIN

extern void      BLZpack_(int*,PetscReal*,PetscScalar*,int*,PetscScalar*,
		          PetscScalar*,int*,int*,PetscScalar*,PetscScalar*);

extern int       BLZistorr_(int*,char*,int);
extern PetscReal BLZrstorr_(PetscReal*,char*,int);

EXTERN_C_END

#endif

