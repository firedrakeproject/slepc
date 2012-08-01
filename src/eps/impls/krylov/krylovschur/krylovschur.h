/*                       

   Private header for Krylov-Schur.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#if !defined(__KRYLOVSCHUR_H)
#define __KRYLOVSCHUR_H

extern PetscErrorCode EPSSolve_KrylovSchur_Default(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS);

typedef struct {
  PetscReal keep;
} EPS_KRYLOVSCHUR;

#endif
