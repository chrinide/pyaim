#!/usr/bin/env python



################
log.info('Go outside betasphere')
EPS = 1e-6
def inbasin(r,j):

    isin = False
    rs1 = 0.0
    irange = nlimsurf[j]
    irange = int(irange)
    for k in range(irange):
        rs2 = rsurf[j,k]
        if (r >= rs1-EPS and r <= rs2+EPS):
            if (((k+1)%2) == 0):
                isin = False
            else:
                isin = True
            return isin
        rs1 = rs2

    return isin

r0 = brad
rfar = rmax
rad = 0.41 #rbrag[]
mapr = 2
log.info('Quadrature mapping : %d ', mapr)
rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
rlmr = 0.0
t0 = time.clock()
for n in range(nrad):
    r = rmesh[n]
    rlm = 0.0
    coords = []
    weigths = []
    for j in range(npang):
        inside = True
        inside = inbasin(r,j)
        if (inside == True):
            cost = coordsang[j,0]
            sintcosp = coordsang[j,1]*coordsang[j,2]
            sintsinp = coordsang[j,1]*coordsang[j,3]
            xcoor[0] = r*sintcosp
            xcoor[1] = r*sintsinp
            xcoor[2] = r*cost    
            p = xnuc + xcoor
            coords.append(p)
            weigths.append(coordsang[j,4])
    coords = numpy.array(coords)
    weigths = numpy.array(weigths)
    den = rho(coords)
    rlm = numpy.einsum('i,i->', den, weigths)
    rlmr += rlm*dvol[n]*rwei[n]
log.info('Electron density outside bsphere %8.5f ', rlmr)    
log.timer('Out Bsphere build', t0)
rhoo = rlmr
log.info('Electron density %8.5f ', (rhob+rhoo))    

