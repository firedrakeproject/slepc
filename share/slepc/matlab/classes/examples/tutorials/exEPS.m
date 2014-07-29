%%
%
%  Solves a standard eigenvalue problem with SLEPc
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
SlepcInitialize({'-eps_monitor','-malloc','-malloc_debug','-malloc_dump'});

%%
%  Create a tridiagonal matrix (1-D Laplacian)
%
n = 130;
mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(n,n,n,n);
mat.SetUp();
for i=1:n
  mat.SetValues(i,i,2.0);
end
for i=1:n-1
  mat.SetValues(i+1,i,-1.0);
  mat.SetValues(i,i+1,-1.0);
end
mat.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
mat.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%%
%  Create the eigensolver, pass the matrix and solve the problem
%
eps = SlepcEPS();
eps.SetType('krylovschur');
eps.SetOperators(mat);
eps.SetProblemType(SlepcEPS.HEP);
eps.SetWhichEigenpairs(SlepcEPS.SMALLEST_MAGNITUDE);
eps.SetFromOptions();
eps.Solve();
nconv = eps.GetConverged();
if nconv>0
  fprintf('           k          ||Ax-kx||/||kx||\n')
  fprintf('   ----------------- ------------------\n')
  for i=1:nconv
    [lambda,x] = eps.GetEigenpair(i);
    figure,plot(x)
    relerr = eps.ComputeRelativeError(i);
    if isreal(lambda)
      fprintf('    %12f        %12g\n',lambda,relerr)
    else
      fprintf('  %6f%+6fj      %12g\n',real(lambda),imag(lambda),relerr)
    end
  end
end

%%
%   Free objects and shutdown SLEPc
%
mat.Destroy();
eps.Destroy();
SlepcFinalize();
