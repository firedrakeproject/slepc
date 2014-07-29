%%
%
%  Computes a partial SVD of a matrix with SLEPc
%  User creates directly a PETSc Mat
%

%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%  SLEPc - Scalable Library for Eigenvalue Problem Computations
%  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
%
%  This file is part of SLEPc.
%
%  SLEPc is free software: you can redistribute it and/or modify it under  the
%  terms of version 3 of the GNU Lesser General Public License as published by
%  the Free Software Foundation.
%
%  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
%  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
%  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
%  more details.
%
%  You  should have received a copy of the GNU Lesser General  Public  License
%  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

%%
%  Set the Matlab path and initialize SLEPc
%
path(path,'../../')
if ~exist('PetscInitialize','file')
  PETSC_DIR = getenv('PETSC_DIR');
  if isempty(PETSC_DIR)
    error('Must set environment variable PETSC_DIR or add the appropriate dir to Matlab path')
  end
  path(path,[PETSC_DIR '/share/petsc/matlab/classes'])
end
SlepcInitialize({'-malloc','-malloc_debug','-malloc_dump'});

%%
%  Create the Lauchli matrix
%
n = 100;
mu = 1e-7;
mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(n+1,n,n+1,n);
mat.SetUp();
for i=1:n
  mat.SetValues(1,i,1.0);
end
for i=2:n+1
  mat.SetValues(i,i-1,mu);
end
mat.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
mat.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%%
%  Create the solver, pass the matrix and solve the problem
%
svd = SlepcSVD();
svd.SetOperator(mat);
svd.SetType('trlanczos');
svd.SetFromOptions();
svd.Solve();
nconv = svd.GetConverged();
if nconv>0
  fprintf('         sigma         residual norm\n')
  fprintf('   ----------------- ------------------\n')
  for i=1:nconv
    [sigma,u,v] = svd.GetSingularTriplet(i);
    relerr = svd.ComputeRelativeError(i);
    fprintf('   %12f       %12g\n',sigma,relerr)
  end
end

%%
%   Free objects and shutdown SLEPc
%
mat.Destroy();
svd.Destroy();
SlepcFinalize();
