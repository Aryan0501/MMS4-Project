""" Code from John Dorelli's `mmspy` repository """
__author__ = "John C. Dorelli <john.dorelli@nasa.gov>"

import numpy as num

class Constants(object):

    def __init__(self):

        self.kb = 1.38e-16
        self.me = 9.1094e-28
        self.mp = 1.6726e-24
        self.echarge = 4.8032e-10
        self.cspeed = 2.9979e10
        self.temp1eV = 1.1604e4
        self.energy1eV = 1.6022e-12
        self.re = 6234.*1e5

class Moments(object):

    def __init__(self, nt = 0, nen_corrected = 0):

        self.nen_corrected = nen_corrected
        self.energy_corrected = num.zeros(nen_corrected, num.float)
        self.scpot = 0.

        self.density = 0.
        self.vx = 0.
        self.vy = 0.
        self.vz = 0.
        self.pxx = 0.
        self.pyy = 0.
        self.pzz = 0.
        self.pxy = 0.
        self.pxz = 0.
        self.pyz = 0.
        self.qx = 0.
        self.qy = 0.
        self.qz = 0.
        self.entropy = 0.
        self.entropy_flux_x = 0.
        self.entropy_flux_y = 0.
        self.entropy_flux_z = 0.

        self.inttheta0 = num.zeros(nen_corrected, num.float)
        self.inttheta1 = num.zeros(nen_corrected, num.float)
        self.inttheta2 = num.zeros(nen_corrected, num.float)
        self.inttheta3 = num.zeros(nen_corrected, num.float)
        self.inttheta4 = num.zeros(nen_corrected, num.float)
        self.inttheta5 = num.zeros(nen_corrected, num.float)
        self.inttheta6 = num.zeros(nen_corrected, num.float)
        self.inttheta7 = num.zeros(nen_corrected, num.float)
        self.inttheta8 = num.zeros(nen_corrected, num.float)
        self.inttheta9 = num.zeros(nen_corrected, num.float)
        self.inttheta0_entropy = num.zeros(nen_corrected, num.float)
        self.inttheta1_entropy = num.zeros(nen_corrected, num.float)
        self.inttheta2_entropy = num.zeros(nen_corrected, num.float)
        self.inttheta3_entropy = num.zeros(nen_corrected, num.float)     

    def compute_energy_integrals(self, E0 = 100., spec = 'des'):

        const = Constants()

        if spec == 'des':
            mass = const.me
        elif spec == 'dis':
            mass = const.mp

        temp1eV = const.temp1eV
        kb = const.kb
        energy1eV = const.energy1eV

        nen = len(self.energy_corrected)

        E0_cgs = E0*const.energy1eV

        ugrid = num.zeros((nen+2), num.float)

        ugrid[0] = 0.
        ugrid[nen+1] = 1.
        ugrid[1:nen+1] = self.energy_corrected/(E0+self.energy_corrected)

        fac_density = num.sqrt(2.)/(mass)**1.5*(E0_cgs)**1.5*num.sqrt(ugrid[1:nen+1])/(1.-ugrid[1:nen+1])**2.5
        fac_v = 2./mass**2*E0_cgs**2*ugrid[1:nen+1]/(1.-ugrid[1:nen+1])**3
        fac_temp = 2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:nen+1]**(1.5)/(1.-ugrid[1:nen+1])**(3.5)
        fac_heat = 2./mass**2*E0_cgs**3*ugrid[1:nen+1]**2/(1.-ugrid[1:nen+1])**4

        integrand = num.zeros((nen+2), num.float64)

        integrand[0] = 0.
        integrand[nen+1] = 0.

        """ number density """

        integrand[1:nen+1] = fac_density*self.inttheta0
        density = self._trap_integ(integrand, x = ugrid)
        if density == 0.:
            density = -999.

        """ entropy per particle """

        integrand[1:nen+1] = fac_density*self.inttheta0_entropy
        hfunc = self._trap_integ(integrand, x = ugrid)
        if hfunc == 0:
            hfunc = -999.
        entropy = hfunc

        """ bulk velocity x component """

        integrand[1:nen+1] = fac_v/density*self.inttheta1
        vx = -self._trap_integ(integrand, x = ugrid)/1e5

        """ bulk velocity y component """

        integrand[1:nen+1] = fac_v/density*self.inttheta2
        vy = -self._trap_integ(integrand, x = ugrid)/1e5

        """ bulk velocity z component """

        integrand[1:nen+1] = fac_v/density*self.inttheta3
        vz = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux x component """

        integrand[1:nen+1] = fac_v*self.inttheta1_entropy
        hx = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux y component """

        integrand[1:nen+1] = fac_v*self.inttheta2_entropy
        hy = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux z component """

        integrand[1:nen+1] = fac_v*self.inttheta3_entropy
        hz = -self._trap_integ(integrand, x = ugrid)/1e5

        """ temperature tensor xx """

        integrand[1:nen+1] = fac_temp*self.inttheta4
        txx = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx**2/kb/temp1eV

        """ temperature tensor yy """

        integrand[1:nen+1] = fac_temp*self.inttheta5
        tyy = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vy**2/kb/temp1eV

        """ temperature tensor zz """

        integrand[1:nen+1] = fac_temp*self.inttheta6
        tzz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vz**2/kb/temp1eV

        """ temperature tensor xy """

        integrand[1:nen+1] = fac_temp*self.inttheta7
        txy = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx*vy/kb/temp1eV

        """ temperature tensor xz """

        integrand[1:nen+1] = fac_temp*self.inttheta8
        txz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx*vz/kb/temp1eV

        """ temperature tensor yz """

        integrand[1:nen+1] = fac_temp*self.inttheta9
        tyz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vy*vz/kb/temp1eV

        pxx = txx*density*kb*temp1eV
        pyy = tyy*density*kb*temp1eV
        pzz = tzz*density*kb*temp1eV
        pxy = txy*density*kb*temp1eV
        pxz = txz*density*kb*temp1eV
        pyz = tyz*density*kb*temp1eV
        pscalar = 1./3.*(pxx+pyy+pzz)

        vmag = num.sqrt(vx**2+vy**2+vz**2)

        """ heat flux x component """

        integrand[1:nen+1] = fac_heat*self.inttheta1
        qtotx = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_x = (pxx*vx+pxy*vy+pxz*vz)*1e5
        pfx = 3./2.*pscalar*vx*1e5
        kex = 0.5*density*mass*vmag**2*vx*1e15
        qx = qtotx - p_dot_v_x - pfx - kex

        """ heat flux y component """

        integrand[1:nen+1] = fac_heat*self.inttheta2
        qtoty = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_y = (pxy*vx+pyy*vy+pyz*vz)*1e5
        pfy = 3./2.*pscalar*vy*1e5
        key = 0.5*density*mass*vmag**2*vy*1e15
        qy = qtoty - p_dot_v_y - pfy - key

        """ heat flux z component """

        integrand[1:nen+1] = fac_heat*self.inttheta3
        qtotz = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_z = (pxz*vx + pyz*vy + pzz*vz)*1e5
        pfz = 3./2.*pscalar*vz*1e5
        kez = 0.5*density*mass*vmag**2*vz*1e15
        qz = qtotz - p_dot_v_z - pfz - kez

        self.density = density
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.pxx = pxx
        self.pyy = pyy
        self.pzz = pzz
        self.pxy = pxy
        self.pxz = pxz
        self.pyz = pyz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.entropy = entropy
        self.entropy_flux_x = hx
        self.entropy_flux_y = hy
        self.entropy_flux_z = hz        

    def compute_angle_integrals(self, fg, thg, phg):

        phimax = num.max(phg)
        phi = num.concatenate((phg, num.array([phimax+11.25])))
        theta = num.concatenate((num.array([0.]), thg))
        theta = num.concatenate((theta, num.array([180.])))

        theta = theta*num.pi/180.
        phi = phi*num.pi/180.

        nth = len(theta)
        nph = len(phi)

        thetatab, phitab = num.meshgrid(theta, phi, indexing = 'ij')
        
        costheta = num.cos(thetatab)
        sintheta = num.sin(thetatab)
        cosphi = num.cos(phitab)
        sinphi = num.sin(phitab)

        cosphi2 = cosphi**2
        sinphi2 = sinphi**2
        cosphi3 = cosphi**3
        sinphi3 = sinphi**3
        costheta2 = costheta**2
        sintheta2 = sintheta**2
        costheta3 = costheta**3
        sintheta3 = sintheta**3

        fgrid = num.zeros((nth,nph),num.float64)
        fgrid[1:nth-1,0:nph-1] = fg
        fgrid[:,nph-1] = fgrid[:,0]
        fgrid[0,:] = 0.
        fgrid[nth-1,:] = 0.

        intphi0 = self._trap_integ(fgrid, x = phitab, dim = 2)
        intphi1 = self._trap_integ(fgrid*cosphi, x = phitab, dim = 2)
        intphi2 = self._trap_integ(fgrid*sinphi, x = phitab, dim = 2)
        intphi3 = self._trap_integ(fgrid*cosphi2, x = phitab, dim = 2)
        intphi4 = self._trap_integ(fgrid*sinphi2, x = phitab, dim = 2)
        intphi5 = self._trap_integ(fgrid*sinphi*cosphi, x = phitab, dim = 2)

        inttheta0 = self._trap_integ(intphi0*sintheta[:,0], x = thetatab[:,0], dim = 1)
        inttheta1 = self._trap_integ(intphi1*sintheta2[:,0], x = thetatab[:,0], dim = 1)
        inttheta2 = self._trap_integ(intphi2*sintheta2[:,0], x = thetatab[:,0], dim = 1)
        inttheta3 = self._trap_integ(intphi0*sintheta[:,0]*costheta[:,0], x = thetatab[:,0], dim = 1)
        inttheta4 = self._trap_integ(intphi3*sintheta3[:,0], x = thetatab[:,0], dim = 1)
        inttheta5 = self._trap_integ(intphi4*sintheta3[:,0], x = thetatab[:,0], dim = 1)
        inttheta6 = self._trap_integ(intphi0*sintheta[:,0]*costheta2[:,0], x = thetatab[:,0], dim = 1)
        inttheta7 = self._trap_integ(intphi5*sintheta3[:,0], x = thetatab[:,0], dim = 1)
        inttheta8 = self._trap_integ(intphi1*sintheta2[:,0]*costheta[:,0], x = thetatab[:,0], dim = 1)
        inttheta9 = self._trap_integ(intphi2*sintheta2[:,0]*costheta[:,0], x = thetatab[:,0], dim = 1)

        intphi0_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100), x = phitab, dim = 2)
        intphi1_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100)*cosphi, x = phitab, dim = 2)
        intphi2_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100)*sinphi, x = phitab, dim = 2)
        
        inttheta0_entropy = self._trap_integ(intphi0_entropy*sintheta[:,0], x = thetatab[:,0], dim = 1)
        inttheta1_entropy = self._trap_integ(intphi1_entropy*sintheta2[:,0], x = thetatab[:,0], dim = 1)
        inttheta2_entropy = self._trap_integ(intphi2_entropy*sintheta2[:,0], x = thetatab[:,0], dim = 1)
        inttheta3_entropy = self._trap_integ(intphi0_entropy*sintheta[:,0]*costheta[:,0], \
                                             x = thetatab[:,0], dim = 1)

        f00 = num.sqrt(1./4./num.pi)*inttheta0
        f1m1= num.sqrt(3./4./num.pi)*inttheta2
        f10 = num.sqrt(3./4./num.pi)*inttheta3
        f11 = num.sqrt(3./4./num.pi)*inttheta1
        f2m2 = num.sqrt(15./4./num.pi)*inttheta7
        f2m1 = num.sqrt(15./4./num.pi)*inttheta9
        f20 = num.sqrt(5./16./num.pi)*(inttheta6-inttheta0)
        f21 = num.sqrt(15./4./num.pi)*inttheta8
        f22 = num.sqrt(15./4./num.pi)*(inttheta4-inttheta5)

        return [inttheta0, inttheta1, inttheta2, inttheta3, inttheta4, inttheta5,
                inttheta6, inttheta7, inttheta8, inttheta9, inttheta0_entropy,
                inttheta1_entropy, inttheta2_entropy, inttheta3_entropy,
                f00, f1m1, f10, f11, f2m2, f2m1, f20, f21, f22]

    def _moments_integrate(self, psd, energy, theta, phi, E0 = 100., species = 'ele'):

        """ set some constants """

        const = Constants()

        if species == 'ele':
            mass = const.me
        elif species == 'ion':
            mass = const.mp

        temp1eV = const.temp1eV
        kb = const.kb
        energy1eV = const.energy1eV

        phimax = num.max(phi)
        phi = num.concatenate((phi, num.array([phimax+11.25])))
        theta = num.concatenate((num.array([0.]), theta))
        theta = num.concatenate((theta, num.array([180.])))

        theta = theta*num.pi/180.
        phi = phi*num.pi/180.

        nen = len(energy)
        nth = len(theta)
        nph = len(phi)

        energytab, thetatab, phitab = num.meshgrid(num.ones(nen), theta, phi, indexing = 'ij')

        costheta = num.cos(thetatab)
        sintheta = num.sin(thetatab)
        cosphi = num.cos(phitab)
        sinphi = num.sin(phitab)

        cosphi2 = cosphi**2
        sinphi2 = sinphi**2
        cosphi3 = cosphi**3
        sinphi3 = sinphi**3
        costheta2 = costheta**2
        sintheta2 = sintheta**2
        costheta3 = costheta**3
        sintheta3 = sintheta**3

        E0_cgs = E0*const.energy1eV

        ugrid = num.zeros((nen+2), num.float)

        ugrid[0] = 0.
        ugrid[nen+1] = 1.
        ugrid[1:nen+1] = energy/(E0+energy)

        fac_density = num.sqrt(2.)/(mass)**1.5*(E0_cgs)**1.5*num.sqrt(ugrid[1:nen+1])/(1.-ugrid[1:nen+1])**2.5
        fac_v = 2./mass**2*E0_cgs**2*ugrid[1:nen+1]/(1.-ugrid[1:nen+1])**3
        fac_temp = 2.**(1.5)/mass**(2.5)*E0_cgs**(2.5)*ugrid[1:nen+1]**(1.5)/(1.-ugrid[1:nen+1])**(3.5)
        fac_heat = 2./mass**2*E0_cgs**3*ugrid[1:nen+1]**2/(1.-ugrid[1:nen+1])**4

        fgrid = num.zeros((nen,nth,nph),num.float64)
        fgrid[:,1:nth-1,0:nph-1] = psd
        fgrid[:,:,nph-1] = fgrid[:,:,0]
        fgrid[:,0,:] = 0.
        fgrid[:,nth-1,:] = 0.

        intphi0 = self._trap_integ(fgrid, x = phitab, dim = 3)
        intphi1 = self._trap_integ(fgrid*cosphi, x = phitab, dim = 3)
        intphi2 = self._trap_integ(fgrid*sinphi, x = phitab, dim = 3)
        intphi3 = self._trap_integ(fgrid*cosphi2, x = phitab, dim = 3)
        intphi4 = self._trap_integ(fgrid*sinphi2, x = phitab, dim = 3)
        intphi5 = self._trap_integ(fgrid*sinphi*cosphi, x = phitab, dim = 3)

        inttheta0 = self._trap_integ(intphi0*sintheta[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta1 = self._trap_integ(intphi1*sintheta2[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta2 = self._trap_integ(intphi2*sintheta2[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta3 = self._trap_integ(intphi0*sintheta[:,:,0]*costheta[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta4 = self._trap_integ(intphi3*sintheta3[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta5 = self._trap_integ(intphi4*sintheta3[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta6 = self._trap_integ(intphi0*sintheta[:,:,0]*costheta2[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta7 = self._trap_integ(intphi5*sintheta3[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta8 = self._trap_integ(intphi1*sintheta2[:,:,0]*costheta[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta9 = self._trap_integ(intphi2*sintheta2[:,:,0]*costheta[:,:,0], x = thetatab[:,:,0], dim = 2)

        intphi0_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100), x = phitab, dim = 3)
        intphi1_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100)*cosphi, x = phitab, dim = 3)
        intphi2_entropy = self._trap_integ(fgrid*num.log(fgrid+1e-100)*sinphi, x = phitab, dim = 3)
        
        inttheta0_entropy = self._trap_integ(intphi0_entropy*sintheta[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta1_entropy = self._trap_integ(intphi1_entropy*sintheta2[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta2_entropy = self._trap_integ(intphi2_entropy*sintheta2[:,:,0], x = thetatab[:,:,0], dim = 2)
        inttheta3_entropy = self._trap_integ(intphi0_entropy*sintheta[:,:,0]*costheta[:,:,0], \
                                             x = thetatab[:,:,0], dim = 2)

        integrand = num.zeros((nen+2), num.float64)

        integrand[0] = 0.
        integrand[nen+1] = 0.

        """ number density """

        integrand[1:nen+1] = fac_density*inttheta0
        density = self._trap_integ(integrand, x = ugrid)
        if density == 0.:
            density = -999.

        """ entropy per particle """

        integrand[1:nen+1] = fac_density*inttheta0_entropy
        hfunc = self._trap_integ(integrand, x = ugrid)
        if hfunc == 0:
            hfunc = -999.
        entropy = hfunc

        """ bulk velocity x component """

        integrand[1:nen+1] = fac_v/density*inttheta1
        vx = -self._trap_integ(integrand, x = ugrid)/1e5

        """ bulk velocity y component """

        integrand[1:nen+1] = fac_v/density*inttheta2
        vy = -self._trap_integ(integrand, x = ugrid)/1e5

        """ bulk velocity z component """

        integrand[1:nen+1] = fac_v/density*inttheta3
        vz = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux x component """

        integrand[1:nen+1] = fac_v*inttheta1_entropy
        hx = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux y component """

        integrand[1:nen+1] = fac_v*inttheta2_entropy
        hy = -self._trap_integ(integrand, x = ugrid)/1e5

        """ entropy flux z component """

        integrand[1:nen+1] = fac_v*inttheta3_entropy
        hz = -self._trap_integ(integrand, x = ugrid)/1e5

        """ temperature tensor xx """

        integrand[1:nen+1] = fac_temp*inttheta4
        txx = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx**2/kb/temp1eV

        """ temperature tensor yy """

        integrand[1:nen+1] = fac_temp*inttheta5
        tyy = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vy**2/kb/temp1eV

        """ temperature tensor zz """

        integrand[1:nen+1] = fac_temp*inttheta6
        tzz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vz**2/kb/temp1eV

        """ temperature tensor xy """

        integrand[1:nen+1] = fac_temp*inttheta7
        txy = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx*vy/kb/temp1eV

        """ temperature tensor xz """

        integrand[1:nen+1] = fac_temp*inttheta8
        txz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vx*vz/kb/temp1eV

        """ temperature tensor yz """

        integrand[1:nen+1] = fac_temp*inttheta9
        tyz = mass*self._trap_integ(integrand, x = ugrid)/density/kb/temp1eV - \
            1e10*mass*vy*vz/kb/temp1eV

        pxx = txx*density*kb*temp1eV
        pyy = tyy*density*kb*temp1eV
        pzz = tzz*density*kb*temp1eV
        pxy = txy*density*kb*temp1eV
        pxz = txz*density*kb*temp1eV
        pyz = tyz*density*kb*temp1eV
        pscalar = 1./3.*(pxx+pyy+pzz)

        vmag = num.sqrt(vx**2+vy**2+vz**2)

        """ heat flux x component """

        integrand[1:nen+1] = fac_heat*inttheta1
        qtotx = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_x = (pxx*vx+pxy*vy+pxz*vz)*1e5
        pfx = 3./2.*pscalar*vx*1e5
        kex = 0.5*density*mass*vmag**2*vx*1e15
        qx = qtotx - p_dot_v_x - pfx - kex

        """ heat flux y component """

        integrand[1:nen+1] = fac_heat*inttheta2
        qtoty = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_y = (pxy*vx+pyy*vy+pyz*vz)*1e5
        pfy = 3./2.*pscalar*vy*1e5
        key = 0.5*density*mass*vmag**2*vy*1e15
        qy = qtoty - p_dot_v_y - pfy - key

        """ heat flux z component """

        integrand[1:nen+1] = fac_heat*inttheta3
        qtotz = -self._trap_integ(integrand, x = ugrid)
        p_dot_v_z = (pxz*vx + pyz*vy + pzz*vz)*1e5
        pfz = 3./2.*pscalar*vz*1e5
        kez = 0.5*density*mass*vmag**2*vz*1e15
        qz = qtotz - p_dot_v_z - pfz - kez

        self.nen_corrected = len(energy)
        self.energy_corrected = energy

        self.density = density
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.pxx = pxx
        self.pyy = pyy
        self.pzz = pzz
        self.pxy = pxy
        self.pxz = pxz
        self.pyz = pyz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.entropy = entropy
        self.entropy_flux_x = hx
        self.entropy_flux_y = hy
        self.entropy_flux_z = hz

    def _trap_integ(self, integrand, x = num.array([0.]), dim = 1):

        if dim == 3:
            ndat = len(x[0,0,:])
            dx = x[:,:,1:ndat]-x[:,:,0:ndat-1]
            res = 0.5*dx*(integrand[:,:,1:ndat]+integrand[:,:,0:ndat-1])
            return res.sum(axis = 2)
        elif dim == 2:
            ndat = len(x[0,:])
            dx = x[:,1:ndat]-x[:,0:ndat-1]
            res = 0.5*dx*(integrand[:,1:ndat]+integrand[:,0:ndat-1])
            return res.sum(axis = 1)
        elif dim == 1:
            ndat = len(x)
            dx = x[1:ndat]-x[0:ndat-1]
            res = 0.5*dx*(integrand[1:ndat]+integrand[0:ndat-1])
            return res.sum()

def moments_test(n0 = 10., V0 = 100., T0 = 10., nen = 32, nth = 16, nph = 32,
                 emin = 10., emax = 30000., spec = 'des'):

    """ set some physical constants """
    
    const = Constants()

    if spec == 'des':
        mass = const.me
    elif spec == 'dis':
        mass = const.mp

    """ define energy grid (nen logarithmically spaced energies) """
        
    logemin = num.log(emin)
    logemax = num.log(emax)
    loge = logemin+(logemax-logemin)*num.arange(nen)/(nen-1.)
    energy = num.exp(loge)

    """ define theta and phi bin centers; note: Moments assumes nth and nph contiguous 11.25 deg bins """
    
    theta = (11.25/2.+11.25*num.arange(nth))*num.pi/180.
    phi = (11.25/2.+11.25*num.arange(nph))*num.pi/180.

    """ define isotropic Maxwellian on 32x16x32 energy/theta/phi bins """
    
    energy_mesh, theta_mesh, phi_mesh = num.meshgrid(energy, theta, phi, indexing = 'ij')

    vspeed = num.sqrt(2.*energy_mesh*const.energy1eV/mass)
    vx = -vspeed*num.sin(theta_mesh)*num.cos(phi_mesh)
    vy = -vspeed*num.sin(theta_mesh)*num.sin(phi_mesh)
    vz = -vspeed*num.cos(theta_mesh)
    
    vbx = V0*1e5
    w0 = num.sqrt(const.kb*T0*const.temp1eV/mass)
    logf = num.log(n0)-3./2.*num.log(2.*num.pi)-3.*num.log(w0) \
      - ((vx-vbx)/w0)**2/2. - (vy/w0)**2/2. - (vz/w0)**2/2.

    """ compute moments """
    
    mom = Moments(nen_corrected = nen)
    for i in range(nen):
        mom.energy_corrected = energy
        f = num.exp(logf[i,:,:])
        res = mom.compute_angle_integrals(f, theta*180./num.pi, phi*180./num.pi)

        mom.inttheta0[i] = res[0]
        mom.inttheta1[i] = res[1]
        mom.inttheta2[i] = res[2]
        mom.inttheta3[i] = res[3]
        mom.inttheta4[i] = res[4]
        mom.inttheta5[i] = res[5]
        mom.inttheta6[i] = res[6]
        mom.inttheta7[i] = res[7]
        mom.inttheta8[i] = res[8]
        mom.inttheta9[i] = res[9]         
            
    mom.compute_energy_integrals()

    return mom