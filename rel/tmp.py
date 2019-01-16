#!/usr/bin/env python
    if (small == True):
        mo_occ = numpy.zeros(nao)
        #mo_occ[] = 0.5/lib.param.LIGHT_SPEED
        pos = mo_occ > OCCDROP
        #cposa = mo_coeff[nao:nao/2,pos]*c1**2
        #cposb = mo_coeff[nao:,pos]*c1**2
        aoa = aoa[:,:,nao/2:nao]
        aob = aob[:,:,nao/2:nao]
    else:
        pos = mo_occ > OCCDROP
        cposa = mo_coeff[0:nao/2,pos]
        cposb = mo_coeff[nao/2:nao,pos]
        aoa = aoa[:,:,0:nao/2]
        aob = aob[:,:,0:nao/2]
        print aoa.shape

    if (xctype == 'LDA'):
        c0a = lib.dot(aoa, cposa)
        rhoaa = numpy.einsum('pi,pi->p', cposa.real, c0a.real)
        rhoaa += numpy.einsum('pi,pi->p', cposa.imag, c0a.imag)
        c0b = lib.dot(aob, cposb)
        rhobb = numpy.einsum('pi,pi->p', cposb.real, c0b.real)
        rhobb += numpy.einsum('pi,pi->p', cposb.imag, c0b.imag)
        rho = (rhoaa + rhobb)
    elif xctype == 'GGA':
        rho = numpy.zeros((4,ngrids))
        print aoa[0].shape, cposa.shape
        c0a = lib.dot(aoa[0], cposa)
        rhoaa = numpy.einsum('pi,pi->p', cposa.real, c0a.real)
        rhoaa += numpy.einsum('pi,pi->p', cposa.imag, c0a.imag)
        c0b = lib.dot(aob[0], cposb)
        rhobb = numpy.einsum('pi,pi->p', cposb.real, c0b.real)
        rhobb += numpy.einsum('pi,pi->p', cposb.imag, c0b.imag)
        print rhobb
        rho[0] = (rhoaa + rhobb)
        for i in range(1, 4):
            c1a = numpy.dot(aoa[i], cpos)
            c1b = numpy.dot(aob[i], cpos)
            rho[i] += numpy.einsum('pi,pi->p', c0a.real, c1a.real)*2 # *2 for +c.c.
            rho[i] += numpy.einsum('pi,pi->p', c0a.imag, c1a.imag)*2 # *2 for +c.c.
            rho[i] += numpy.einsum('pi,pi->p', c0b.real, c1b.real)*2 # *2 for +c.c.
            rho[i] += numpy.einsum('pi,pi->p', c0b.imag, c1b.imag)*2 # *2 for +c.c.

    return rho
