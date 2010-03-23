/*                       

   Private header for QEPLINEAR.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

typedef struct {
  PetscTruth explicitmatrix;
  PetscInt   cform;           /* companion form */
  Mat        A,B;             /* matrices of generalized eigenproblem */
  EPS        eps;             /* linear eigensolver for Az=lBz */
  Mat        M,C,K;           /* copy of QEP coefficient matrices */
  Vec        x1,x2,y1,y2;     /* work vectors */
} QEP_LINEAR;

EXTERN PetscErrorCode MatMult_QEPLINEAR_N1A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_N1B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N1A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N1B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N1A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N1B(MPI_Comm,QEP_LINEAR*,Mat*);

