function err = SlepcInitialize(args,argfile,arghelp)
%  SlepcInitialize: start to use SLEPc classes in MATLAB.
%  In order to use the SLEPc MATLAB classes, PETSc must have been configured with
%  specific options. See: help PetscInitialize.
%
%  Add ${PETSC_DIR}/share/petsc/matlab/classes to your MATLAB path, as well as
%  ${SLEPC_DIR}/share/slepc/matlab/classes
%
%  In MATLAB use help Slepc to get started using SLEPc from MATLAB

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

if exist('PetscInitialize')~=2
  error('Must add ${PETSC_DIR}/share/petsc/matlab/classes to your MATLAB path')
end

if ~libisloaded('libslepc')
  SLEPC_DIR = getenv('SLEPC_DIR');
  PETSC_ARCH = getenv('PETSC_ARCH');
  if isempty(SLEPC_DIR)
    disp('Must have environmental variable SLEPC_DIR set')
  end
  if isempty(PETSC_ARCH)
    disp('Must have environmental variable PETSC_ARCH set')
  end
  loadlibrary([SLEPC_DIR '/' PETSC_ARCH '/lib/' 'libslepc'], [SLEPC_DIR '/share/slepc/matlab/classes/slepcmatlabheader.h']);
end

if ~libisloaded('libpetsc')
  PETSC_DIR = getenv('PETSC_DIR');
  PETSC_ARCH = getenv('PETSC_ARCH');
  if isempty(PETSC_DIR)
    disp('Must have environmental variable PETSC_DIR set')
  end
  if isempty(PETSC_ARCH)
    disp('Must have environmental variable PETSC_ARCH set')
  end
  loadlibrary([PETSC_DIR '/' PETSC_ARCH '/lib/' 'libpetsc'], [PETSC_DIR '/share/petsc/matlab/classes/matlabheader.h']);
end

if (nargin == 0)
  args = '';
end
if (nargin < 2)
  argfile = '';
end
if (nargin < 3)
  arghelp = '';
end
if (ischar(args))
  args = {args};
end

% append any options in the options variable
global options
if (length(options) > 0)
  args = [args,options];
  disp('Using additional options')
  disp(options)
end

% first argument should be program name, use matlab for this
arg = ['matlab',args];
%
% If the user forgot to SlepcFinalize() we do it for them, before restarting SLEPc
%
init = 0;
err = calllib('libslepc', 'SlepcInitialized',init);
if (init)
  err = calllib('libslepc', 'SlepcFinalize');PetscCHKERRQ(err);
end
err = calllib('libslepc', 'SlepcInitializeNoPointers', length(arg), arg,argfile,arghelp);PetscCHKERRQ(err);


