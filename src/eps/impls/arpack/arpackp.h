/*
   Private data structure used by the ARPACK interface
*/

#if !defined(__ARPACKP_H)
#define __ARPACKP_H

#include "src/eps/epsimpl.h"

typedef struct {
  PetscTruth  *select;
  PetscScalar *workev;
  PetscScalar *workd;
  PetscScalar *workl;
  int         lworkl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal  *rwork;
#endif
} EPS_ARPACK;


/*
   Definition of routines from the ARPACK package
*/
#include "petsc.h"

/*
   This include file on the Cray T3D/T3E defines the interface between 
  Fortran and C representations of character strings.
*/
#if defined(PETSC_USES_CPTOFCD)
#include <fortran.h>
#endif

#if !defined(_petsc_mpi_uni)

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define PDNAUPD  PSNAUPD
#define PDNEUPD  PSNEUPD
#define PDSAUPD  PSSAUPD
#define PDSEUPD  PSSEUPD
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define ARnaupd_ pdnaupd_
#define ARneupd_ pdneupd_
#define ARsaupd_ pdsaupd_
#define ARseupd_ pdseupd_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define ARnaupd_ PDNAUPD
#define ARneupd_ PDNEUPD
#define ARsaupd_ PDSAUPD
#define ARseupd_ PDSEUPD
#else
#define ARnaupd_ pdnaupd
#define ARneupd_ pdneupd
#define ARsaupd_ pdsaupd
#define ARseupd_ pdseupd
#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USES_FORTRAN_SINGLE) 
#define PZNAUPD  PCNAUPD
#define PZNEUPD  PCNEUPD
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define ARnaupd_ pznaupd_
#define ARneupd_ pzneupd_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define ARnaupd_ PZNAUPD
#define ARneupd_ PZNEUPD
#else
#define ARnaupd_ pznaupd
#define ARneupd_ pzneupd
#endif

#endif

#else
/* _petsc_mpi_uni */

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define DNAUPD  SNAUPD
#define DNEUPD  SNEUPD
#define DSAUPD  SSAUPD
#define DSEUPD  SSEUPD
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) 
#define ARnaupd__ dnaupd
#define ARneupd__ dneupd
#define ARsaupd__ dsaupd
#define ARseupd__ dseupd
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define ARnaupd__ DNAUPD
#define ARneupd__ DNEUPD
#define ARsaupd__ DSAUPD
#define ARseupd__ DSEUPD
#else
#define ARnaupd__ dnaupd
#define ARneupd__ dneupd
#define ARsaupd__ dsaupd
#define ARseupd__ dseupd
#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USES_FORTRAN_SINGLE) 
#define ZNAUPD  CNAUPD
#define ZNEUPD  CNEUPD
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ARnaupd__ znaupd
#define ARneupd__ zneupd
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define ARnaupd__ ZNAUPD
#define ARneupd__ ZNEUPD
#else
#define ARnaupd__ pznaupd
#define ARneupd__ pzneupd
#endif

#endif

#endif

EXTERN_C_BEGIN

#if !defined(_petsc_mpi_uni)

extern void   ARsaupd_(MPI_Comm*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
extern void   ARseupd_(MPI_Comm*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);

#if !defined(PETSC_USE_COMPLEX)
extern void   ARnaupd_(MPI_Comm*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
extern void   ARneupd_(MPI_Comm*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#else
extern void   ARnaupd_(MPI_Comm*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
extern void   ARneupd_(MPI_Comm*,PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#endif

#else
/* _petsc_mpi_uni */

extern void   ARsaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARsaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARsaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
extern void   ARseupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARseupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) ARseupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) 

#if !defined(PETSC_USE_COMPLEX)
extern void   ARnaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
extern void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) 
#else
extern void   ARnaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) 
extern void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) 
#endif

#endif

EXTERN_C_END

#endif

