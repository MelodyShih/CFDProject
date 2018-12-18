include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/petscvariables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

INCLUDE_DIR=./src

%.o: %.c
	-${CLINKER} -c ${PETSC_CCPPFLAGS} ${PETSC_LIB} $< -o $@

timedepstokes: src/timedepstokes.c src/operators.o src/utils.o src/solver.o
	-${CLINKER} -o timedepstokes $^ ${PETSC_CCPPFLAGS} ${PETSC_LIB}

heat: src/heat.c src/operators.o src/utils.o src/solver.o
	-${CLINKER} -o heat $^ ${PETSC_CCPPFLAGS} ${PETSC_LIB}

poission_mg: src/poission_mg.c
	-${CLINKER} -o poission_mg $< ${PETSC_CCPPFLAGS} ${PETSC_LIB}
