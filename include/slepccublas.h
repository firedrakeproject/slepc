/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Macro definitions to use CUBLAS functionality
*/

#if !defined(SLEPCCUBLAS_H)
#define SLEPCCUBLAS_H
#include <petscdevice.h>

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasCgemm((a),(b),(c),(d),(e),(f),(const cuComplex*)(g),(const cuComplex*)(h),(i),(const cuComplex*)(j),(k),(const cuComplex*)(l),(cuComplex*)(m),(n))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) cublasCgemv((a),(b),(c),(d),(cuComplex*)(e),(cuComplex*)(f),(g),(cuComplex*)(h),(i),(cuComplex*)(j),(cuComplex*)(k),(l))
#define cublasXscal(a,b,c,d,e) cublasCscal((a),(b),(const cuComplex*)(c),(cuComplex*)(d),(e))
#define cublasXnrm2(a,b,c,d,e) cublasScnrm2((a),(b),(const cuComplex*)(c),(d),(e))
#define cublasXaxpy(a,b,c,d,e,f,g) cublasCaxpy((a),(b),(const cuComplex*)(c),(const cuComplex*)(d),(e),(cuComplex*)(f),(g))
#define cublasXdotc(a,b,c,d,e,f,g) cublasCdotc((a),(b),(const cuComplex *)(c),(d),(const cuComplex *)(e),(f),(cuComplex *)(g))
#define cublasXgetrfBatched(a,b,c,d,e,f,g) cublasCgetrfBatched((a),(b),(cuComplex**)(c),(d),(e),(f),(g))
#define cublasXgetrsBatched(a,b,c,d,e,f,g,h,i,j,k) cublasCgetrsBatched((a),(b),(c),(d),(const cuComplex**)(e),(f),(g),(cuComplex**)(h),(i),(j),(k))
#else /* complex double */
#define cublasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) cublasZgemm((a),(b),(c),(d),(e),(f),(const cuDoubleComplex*)(g),(const cuDoubleComplex*)(h),(i),(const cuDoubleComplex*)(j),(k),(const cuDoubleComplex*)(l),(cuDoubleComplex *)(m),(n))
#define cublasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) cublasZgemv((a),(b),(c),(d),(cuDoubleComplex*)(e),(cuDoubleComplex*)(f),(g),(cuDoubleComplex*)(h),(i),(cuDoubleComplex*)(j),(cuDoubleComplex*)(k),(l))
#define cublasXscal(a,b,c,d,e) cublasZscal((a),(b),(const cuDoubleComplex*)(c),(cuDoubleComplex*)(d),(e))
#define cublasXnrm2(a,b,c,d,e) cublasDznrm2((a),(b),(const cuDoubleComplex*)(c),(d),(e))
#define cublasXaxpy(a,b,c,d,e,f,g) cublasZaxpy((a),(b),(const cuDoubleComplex*)(c),(const cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g))
#define cublasXdotc(a,b,c,d,e,f,g) cublasZdotc((a),(b),(const cuDoubleComplex *)(c),(d),(const cuDoubleComplex *)(e),(f),(cuDoubleComplex *)(g))
#define cublasXgetrfBatched(a,b,c,d,e,f,g) cublasZgetrfBatched((a),(b),(cuDoubleComplex**)(c),(d),(e),(f),(g))
#define cublasXgetrsBatched(a,b,c,d,e,f,g,h,i,j,k) cublasZgetrsBatched((a),(b),(c),(d),(const cuDoubleComplex**)(e),(f),(g),(cuDoubleComplex**)(h),(i),(j),(k))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXgemm cublasSgemm
#define cublasXgemv cublasSgemv
#define cublasXscal cublasSscal
#define cublasXnrm2 cublasSnrm2
#define cublasXaxpy cublasSaxpy
#define cublasXdotc cublasSdot
#define cublasXgetrfBatched cublasSgetrfBatched
#define cublasXgetrsBatched cublasSgetrsBatched
#else /* real double */
#define cublasXgemm cublasDgemm
#define cublasXgemv cublasDgemv
#define cublasXscal cublasDscal
#define cublasXnrm2 cublasDnrm2
#define cublasXaxpy cublasDaxpy
#define cublasXdotc cublasDdot
#define cublasXgetrfBatched cublasDgetrfBatched
#define cublasXgetrsBatched cublasDgetrsBatched
#endif
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#define cublasXCaxpy(a,b,c,d,e,f,g)                cublasCaxpy((a),(b),(const cuComplex *)(c),(const cuComplex *)(d),(e),(cuComplex *)(f),(g))
#define cublasXCgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  cublasCgemm((a),(b),(c),(d),(e),(f),(const cuComplex *)(g),(const cuComplex *)(h),(i),(const cuComplex *)(j),(k),(const cuComplex *)(l),(cuComplex *)(m),(n))
#define cublasXCscal(a,b,c,d,e)                    cublasCscal((a),(b),(const cuComplex *)(c),(cuComplex *)(d),(e))
#else
#define cublasXCaxpy(a,b,c,d,e,f,g)                cublasZaxpy((a),(b),(const cuDoubleComplex *)(c),(const cuDoubleComplex *)(d),(e),(cuDoubleComplex *)(f),(g))
#define cublasXCgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  cublasZgemm((a),(b),(c),(d),(e),(f),(const cuDoubleComplex *)(g),(const cuDoubleComplex *)(h),(i),(const cuDoubleComplex *)(j),(k),(const cuDoubleComplex *)(l),(cuDoubleComplex *)(m),(n))
#define cublasXCscal(a,b,c,d,e)                    cublasZscal((a),(b),(const cuDoubleComplex *)(c),(cuDoubleComplex *)(d),(e))
#endif /* COMPLEX */

#endif
