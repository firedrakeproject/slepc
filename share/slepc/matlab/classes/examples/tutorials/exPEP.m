%%
%
%  Solves a polynomial eigenvalue problem with SLEPc
%  User creates directly the three PETSc Mat
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
%  Create problem matrices
%
n = 10;
N = n*n;

%  K is the 2-D Laplacian
K = PetscMat();
K.SetType('seqaij');
K.SetSizes(N,N,N,N);
K.SetUp();
for II=0:N-1
  i = floor(II/n);
  j = II-i*n;
  if i>0,   K.SetValues(II+1,II+1-n,-1.0); end
  if i<n-1, K.SetValues(II+1,II+1+n,-1.0); end
  if j>0,   K.SetValues(II+1,II+1-1,-1.0); end
  if j<n-1, K.SetValues(II+1,II+1+1,-1.0); end
  K.SetValues(II+1,II+1,4.0);
end
K.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
K.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%  C is the zero matrix
C = PetscMat();
C.SetType('seqaij');
C.SetSizes(N,N,N,N);
C.SetUp();
C.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
C.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%  M is the identity matrix
M = PetscMat();
M.SetType('seqaij');
M.SetSizes(N,N,N,N);
M.SetUp();
for II=1:N
  M.SetValues(II,II,1.0);
end
M.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
M.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%%
%  Create the eigensolver, pass the matrices and solve the problem
%
pep = SlepcPEP();
pep.SetType('toar');
pep.SetOperators({K,C,M});
pep.SetProblemType(SlepcPEP.GENERAL);
pep.SetFromOptions();
pep.Solve();
nconv = pep.GetConverged();
if nconv>0
  fprintf('           k          ||Ax-kx||/||kx||\n')
  fprintf('   ----------------- ------------------\n')
  for i=1:nconv
    [lambda,x] = pep.GetEigenpair(i);
    relerr = pep.ComputeRelativeError(i);
    if isreal(lambda)
      fprintf('    %14.2f        %12g\n',lambda,relerr)
    else
      fprintf('  %9f%+9f      %12g\n',real(lambda),imag(lambda),relerr)
    end
  end
end

%%
%   Free objects and shutdown SLEPc
%
K.Destroy();
C.Destroy();
M.Destroy();
pep.Destroy();
SlepcFinalize();
