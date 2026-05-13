from typing import List
from mmspy import moments
import numpy as np


class Skymap:
    def __init__(self, steps, name="Skymaps", nen=32):
        self.skymap = np.empty((steps, 16, 32, nen))
        self.name = name
        self.steps = steps

    def __len__(self):
        return self.steps

    def energySpectrogram(self):
        return self.skymap.sum(axis=(1, 2))

    @property
    def momsTS(self):
        if not hasattr(self, '_momsTS'):
            self._updateMomTS()
        return self._momsTS

    def _updateMomTS(self):

        lowE = 0
        highE = 32

        nth = 16
        nph = 32
        theta = (11.25/2.+11.25*np.arange(nth))*np.pi/180.
        phi = (11.25/2.+11.25*np.arange(nph))*np.pi/180.

        nen = 32
        emin = 10
        emax = 30000
        logemin = np.log(emin)
        logemax = np.log(emax)
        loge = logemin+(logemax-logemin)*np.arange(nen)/(nen-1.)
        energy = np.exp(loge)

        self._momsTS = MomentsTimeSeries(self.steps, self.name)

        integrands = np.zeros((self.steps, 23, 32))

        for t in range(self.steps):
            mom = moments.Moments(nen_corrected=nen)
            for energy_i in range(nen):
                mom.energy_corrected = energy

                integrands[t, :, energy_i] = mom.compute_angle_integrals(
                    self.skymap[t, energy_i], theta*180./np.pi, phi*180./np.pi)

                mom.inttheta0[energy_i] = integrands[t, :, energy_i][0]
                mom.inttheta1[energy_i] = integrands[t, :, energy_i][1]
                mom.inttheta2[energy_i] = integrands[t, :, energy_i][2]
                mom.inttheta3[energy_i] = integrands[t, :, energy_i][3]
                mom.inttheta4[energy_i] = integrands[t, :, energy_i][4]
                mom.inttheta5[energy_i] = integrands[t, :, energy_i][5]
                mom.inttheta6[energy_i] = integrands[t, :, energy_i][6]
                mom.inttheta7[energy_i] = integrands[t, :, energy_i][7]
                mom.inttheta8[energy_i] = integrands[t, :, energy_i][8]
                mom.inttheta9[energy_i] = integrands[t, :, energy_i][9]

                mom.inttheta0_entropy[energy_i] = integrands[t,
                                                             :, energy_i][10]
                mom.inttheta1_entropy[energy_i] = integrands[t,
                                                             :, energy_i][11]
                mom.inttheta2_entropy[energy_i] = integrands[t,
                                                             :, energy_i][12]
                mom.inttheta3_entropy[energy_i] = integrands[t,
                                                             :, energy_i][13]

            mom.compute_energy_integrals()
            self._momsTS.addMom(mom, t)

    def __str__(self):
        return self.name


class MomentsTimeSeries:
    def __init__(self, size, name='Moment'):
        self.name = name

        self.density = np.zeros(size)
        self.vx = np.zeros(size)
        self.vy = np.zeros(size)
        self.vz = np.zeros(size)
        self.pxx = np.zeros(size)
        self.pyy = np.zeros(size)
        self.pzz = np.zeros(size)
        self.pxy = np.zeros(size)
        self.pxz = np.zeros(size)
        self.pyz = np.zeros(size)
        self.qx = np.zeros(size)
        self.qy = np.zeros(size)
        self.qz = np.zeros(size)
        self.entropy = np.zeros(size)
        self.entropy_flux_x = np.zeros(size)
        self.entropy_flux_y = np.zeros(size)
        self.entropy_flux_z = np.zeros(size)

    def __len__(self):
        return len(self.vx)

    def addMom(self, mom: moments.Moments, t: int):
        self.density[t] = mom.density
        self.vx[t] = mom.vx
        self.vy[t] = mom.vy
        self.vz[t] = mom.vz
        self.pxx[t] = mom.pxx
        self.pyy[t] = mom.pyy
        self.pzz[t] = mom.pzz
        self.pxy[t] = mom.pxy
        self.pxz[t] = mom.pxz
        self.pyz[t] = mom.pyz
        self.qx[t] = mom.qx
        self.qy[t] = mom.qy
        self.qz[t] = mom.qz
        self.entropy[t] = mom.entropy
        self.entropy_flux_x[t] = mom.entropy_flux_x
        self.entropy_flux_y[t] = mom.entropy_flux_y
        self.entropy_flux_z[t] = mom.entropy_flux_z


def plotMomentsTimeSeries(momsTS: MomentsTimeSeries):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3)

    # Density
    axs[0, 0].plot(momsTS.density)

    # Entropy
    axs[0, 1].plot(momsTS.entropy)

    # Pressure tensor
    axs[0, 2].plot(momsTS.pxx)
    axs[0, 2].plot(momsTS.pxy)
    axs[0, 2].plot(momsTS.pxz)
    axs[0, 2].plot(momsTS.pyy)
    axs[0, 2].plot(momsTS.pyz)
    axs[0, 2].plot(momsTS.pzz)

    # Bulk velocity
    axs[1, 0].plot(momsTS.vx)
    axs[1, 0].plot(momsTS.vy)
    axs[1, 0].plot(momsTS.vz)

    # Entropy Flux
    axs[1, 1].plot(momsTS.entropy_flux_x)
    axs[1, 1].plot(momsTS.entropy_flux_y)
    axs[1, 1].plot(momsTS.entropy_flux_z)

    # Heat flux
    axs[1, 2].plot(momsTS.qx)
    axs[1, 2].plot(momsTS.qy)
    axs[1, 2].plot(momsTS.qz)

    plt.show()
