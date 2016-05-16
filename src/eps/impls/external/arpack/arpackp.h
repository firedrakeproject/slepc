/*
   Private data structure used by the ARPACK interface

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__ARPACKP_H)
#define __ARPACKP_H

typedef struct {
  PetscBool    *select;
  PetscScalar  *workev;
  PetscScalar  *workd;
  PetscScalar  *workl;
  PetscBLASInt lworkl;
  PetscReal    *rwork;
} EPS_ARPACK;

/*
   Definition of routines from the ARPACK package
*/

#if defined(PETSC_HAVE_MPIUNI)

#if defined(PETSC_USE_COMPLEX)

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) c##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) C##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) c##lcase
#endif

#else

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) z##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) Z##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) z##lcase
#endif

#endif

#else

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) s##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) S##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) s##lcase
#endif

#else

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) d##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) D##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) d##lcase
#endif

#endif

#endif

#else  /* not MPIUNI */

#if defined(PETSC_USE_COMPLEX)

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) pc##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) PC##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) pc##lcase
#endif

#else

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) pz##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) PZ##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) pz##lcase
#endif

#endif

#else

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) ps##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) PS##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) ps##lcase
#endif

#else

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) pd##lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) PD##ucase
#else
#define SLEPC_ARPACK(lcase,ucase) pd##lcase
#endif

#endif

#endif

#endif

#if defined(PETSC_HAVE_MPIUNI)

#define COMM_ARG

#if !defined(PETSC_USE_COMPLEX)

#define ARPACKnaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_ARPACK(naupd,NAUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,2)
#define ARPACKneupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y) SLEPC_ARPACK(neupd,NEUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),1,1,2)
#define ARPACKsaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_ARPACK(saupd,SAUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,2)
#define ARPACKseupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,o,p,q,r,s,t,u,v,w) SLEPC_ARPACK(seupd,SEUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,2)

#else

#define ARPACKnaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_ARPACK(naupd,NAUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,2)
#define ARPACKneupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x) SLEPC_ARPACK(neupd,NEUPD) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),1,1,2)

#endif

#else /* not MPIUNI */

#define COMM_ARG MPI_Fint*,

#if !defined(PETSC_USE_COMPLEX)

#define ARPACKnaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_ARPACK(naupd,NAUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,2)
#define ARPACKneupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y) SLEPC_ARPACK(neupd,NEUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),1,1,2)
#define ARPACKsaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) SLEPC_ARPACK(saupd,SAUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),1,2)
#define ARPACKseupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,o,p,q,r,s,t,u,v,w) SLEPC_ARPACK(seupd,SEUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(o),(p),(q),(r),(s),(t),(u),(v),(w),1,1,2)

#else

#define ARPACKnaupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) SLEPC_ARPACK(naupd,NAUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),1,2)
#define ARPACKneupd_(comm,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x) SLEPC_ARPACK(neupd,NEUPD) ((comm),(a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),1,1,2)

#endif

#endif

PETSC_EXTERN void   SLEPC_ARPACK(saupd,SAUPD)(COMM_ARG PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,int,int);
PETSC_EXTERN void   SLEPC_ARPACK(seupd,SEUPD)(COMM_ARG PetscBool*,char*,PetscBool*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,int,int,int);

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void   SLEPC_ARPACK(naupd,NAUPD)(COMM_ARG PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,int,int);
PETSC_EXTERN void   SLEPC_ARPACK(neupd,NEUPD)(COMM_ARG PetscBool*,char*,PetscBool*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,int,int,int);
#else
PETSC_EXTERN void   SLEPC_ARPACK(naupd,NAUPD)(COMM_ARG PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,int,int);
PETSC_EXTERN void   SLEPC_ARPACK(neupd,NEUPD)(COMM_ARG PetscBool*,char*,PetscBool*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,int,int,int);
#endif

#endif

