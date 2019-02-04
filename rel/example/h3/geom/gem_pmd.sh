#!/bin/bash
for i in `ls -1 *.wfn`; do
cat << EOF > ${i%.wfn}.pmd
${i}
nosymmetry
tes
  epsiscp 0.220
  radialquad 7
  rmapping 2
  lmax 10
  nr 551
  lebedev 5810
  betasphere
  betarad 1 0.200
  radialquadbeta 7
  rmappingbeta 3
  nrb 451
  lmaxbeta 8
  lebedevbeta 3074
  dafh
endtes
EOF
done
