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

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) ucase
#else
#define SLEPC_ARPACK(lcase,ucase) lcase
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
#define ARnaupd_ SLEPC_ARPACK(psnaupd,PSNAUPD)
#define ARneupd_ SLEPC_ARPACK(psneupd,PSNEUPD)
#define ARsaupd_ SLEPC_ARPACK(pssaupd,PSSAUPD)
#define ARseupd_ SLEPC_ARPACK(psseupd,PSSEUPD)

#else

#define ARnaupd_ SLEPC_ARPACK(pdnaupd,PDNAUPD)
#define ARneupd_ SLEPC_ARPACK(pdneupd,PDNEUPD)
#define ARsaupd_ SLEPC_ARPACK(pdsaupd,PDSAUPD)
#define ARseupd_ SLEPC_ARPACK(pdseupd,PDSEUPD)

#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USE_SINGLE) 

#define ARnaupd_ SLEPC_ARPACK(pcnaupd,PCNAUPD)
#define ARneupd_ SLEPC_ARPACK(pcneupd,PCNEUPD)

#else

#define ARnaupd_ SLEPC_ARPACK(pznaupd,PZNAUPD)
#define ARneupd_ SLEPC_ARPACK(pzneupd,PZNEUPD)

#endif

#endif

#else
/* _petsc_mpi_uni */

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USE_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define ARnaupd__ SLEPC_ARPACK(snaupd,SNAUPD)
#define ARneupd__ SLEPC_ARPACK(sneupd,SNEUPD)
#define ARsaupd__ SLEPC_ARPACK(ssaupd,SSAUPD)
#define ARseupd__ SLEPC_ARPACK(sseupd,SSEUPD)

#else

#define ARnaupd__ SLEPC_ARPACK(dnaupd,DNAUPD)
#define ARneupd__ SLEPC_ARPACK(dneupd,DNEUPD)
#define ARsaupd__ SLEPC_ARPACK(dsaupd,DSAUPD)
#define ARseupd__ SLEPC_ARPACK(dseupd,DSEUPD)

#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USE_SINGLE) 

#define ARnaupd__ SLEPC_ARPACK(cnaupd,CNAUPD)
#define ARneupd__ SLEPC_ARPACK(cneupd,CNEUPD)

#else

#define ARnaupd__ SLEPC_ARPACK(znaupd,ZNAUPD)
#define ARneupd__ SLEPC_ARPACK(zneupd,ZNEUPD)

#endif

#endif

#endif

EXTERN_C_BEGIN

#if !defined(_petsc_mpi_uni)

EXTERN void   ARsaupd_(MPI_Fint*,int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
EXTERN void   ARseupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);

#if !defined(PETSC_USE_COMPLEX)
EXTERN void   ARnaupd_(MPI_Fint*,int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
EXTERN void   ARneupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#else
EXTERN void   ARnaupd_(MPI_Fint*,int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
EXTERN void   ARneupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#endif

#else
/* _petsc_mpi_uni */

EXTERN void   ARsaupd__(int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARsaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARsaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
EXTERN void   ARseupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARseupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) ARseupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) 

#if !defined(PETSC_USE_COMPLEX)
EXTERN void   ARnaupd__(int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
EXTERN void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) 
#else
EXTERN void   ARnaupd__(int*,char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) 
EXTERN void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,const char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) 
#endif

#endif

EXTERN_C_END

#endif

