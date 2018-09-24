import numpy as np
import pickle
from numpy.linalg import inv


class SetupCylinder(object):

    def __init__(self, system, center, axis, length, radius, nCylinderPhi, nCylinderZ, direction=1, transMatrix=None, invMatrix=None, typeIDoffset=None):

        self.system = system
        self.center = np.array(center, dtype=float)
        self.axis = np.array(axis, dtype=float)
        self.length = length
        self.radius = radius
        self.nCylinderPhi = nCylinderPhi
        self.nCylinderZ = nCylinderZ
        self.direction = direction
        self.splitCutoff = 0.

        if all(self.axis == np.array([0., 0., 1.])):
            self.useTrans = False
        else:
            self.useTrans = True

        if transMatrix is not None and invMatrix is not None:
            # use given matrices
            if not (hasattr(transMatrix, '__iter__') or hasattr(invMatrix, '__iter__')):
                raise TypeError('transformation matrices have to be arrays!')

            self.transMatrix = np.array(transMatrix, dtype=float)
            self.invMatrix = np.array(invMatrix, dtype=float)
            if self.transMatrix.shape != (3, 3) or self.invMatrix.shape != (3, 3):
                raise ValueError('Transfomation matrices have to be float arrays of shape (3, 3)')
        else:
            # calculate transformation matix
            # plane normal is ez
            ez = self.axis / np.sqrt(np.sum(np.square(self.axis)))

            # Find a vector orthogonal to ez, since {1,0,0} and {0,1,0} are independent, ez can not be parallel to both of them. Then we can do Gram-Schmidt
            if (np.dot(np.array([1., 0., 0.]), ez) < 1.):
                ex = np.array([1., 0., 0.]) - np.dot(ez, np.array([1., 0., 0.])) * ez
            else:
                ex = np.array([0., 1., 0.]) - np.dot(ez, np.array([0., 1., 0.])) * ez

            ex = ex / np.sqrt(np.sum(np.square(ex)))

            # since ez and ex are normalized, ey is normalized too
            ey = np.cross(ez, ex)

            self.transMatrix = np.column_stack((ex, ey, ez))

            try:
                self.invMatrix = inv(self.transMatrix)
            except np.linalg.LinAlgError:
                # this error cannot appear since we construct the matric ourselfes
                raise ValueError(
                    "Calculated transformation matrix is not invertible!")

    def _registerTypes(self, icc):
        # register types to ICC!
        self.typeIDoffset = icc.addTypeCylinder(_center=self.center,
                                                _axis=self.axis,
                                                _length=self.length,
                                                _radius=self.radius,
                                                _direction=self.direction,
                                                _cutoff=self.splitCutoff,
                                                _useTrans=self.useTrans,
                                                _transMatrix=self.transMatrix.flatten(),
                                                _invMatrix=self.invMatrix.flatten())


    def initParticles(self, particleTypeID, iccTypeID, initCharge, sigma, epsilon, splitCutoff=0.):
        '''
          initialize all particles for given parts

          Parameters
          ----------
          initCharge: 'float'
            initial particle charge
          sigma: 'float'
            additional surface charge density
          epsilon: 'float'
            epsilon value for this part
          parts: 'string' optional
            parts that are initialized
            splitCutoff: 'float, np.ndarray' optional
                    minimum allowed displacement size of DxDyDz, specificable in order cylinder, torus, wall and interface and direction DxDyDz
                    e.g. i or [i, ...] or [[i, i, i], ...]
        '''

        if not (isinstance(initCharge, float) or isinstance(initCharge, int)):
            raise TypeError('Initial particle charge has to be a float!')

        if not (isinstance(sigma, float) or isinstance(sigma, int)):
            raise TypeError('sigma value has to be a float!')

        if not (isinstance(epsilon, float) or isinstance(epsilon, int)):
            raise TypeError('epsilon value has to be a float!')

        if not hasattr(splitCutoff, '__iter__'):
            if not (isinstance(splitCutoff, float) or isinstance(splitCutoff, int)):
                raise TypeError('splitCutoff value has to be a float!')
            else:
                # splitCutoff needs to be transformed since DxDyDz[2] is always 0 by default!
                self.splitCutoff = np.array([splitCutoff, splitCutoff, 0.])

        else:
            self.splitCutoff = np.array(splitCutoff, dtype=float)
            if self.splitCutoff.size != 3:
                raise ValueError('splitCutoff has to be an array of size 3')

        # this method is only correct for (100) type normal vectors!

        # check for existing particle types to prevent override

        # check if transformation matrix is necessary
#        if all(self.normal == [0., 0., 1.]):
#            self.useTrans = False
#        else:
#            self.useTrans = True
#
#        # register types to ICC!
#        self.typeIDoffset = icc.addTypeWall(_normal=normal,
#                                 _dist=dist,
#                                 _cutoff=cutoff,
#                                 _useTrans=useTrans,
#                                 _transMatrix=transMatrix.flatten(),
#                                 _invMatrix=invMatrix.flatten())

        if self.nCylinderPhi > 0 and self.nCylinderZ > 0:
            DeltaPhi = 2. * np.pi / self.nCylinderPhi
            DeltaZ = self.length / self.nCylinderZ
y
            # cylinder particles
            for Iphi in range(self.nCylinderPhi):
                phi = (Iphi + 0.5) * DeltaPhi
                for IZ in range(-int(self.nCylinderZ / 2), int(self.nCylinderZ / 2) + self.nCylinderZ % 2):
                    z = IZ * DeltaZ

                    pos = self.calcCylPart(phi, z, np.array([0., DeltaPhi / 2., DeltaZ / 2.]))

                    self.system.part.add(pos=np.dot(self.transMatrix, pos[0]),
                                         q=initCharge,
                                         normal=np.dot(self.transMatrix, pos[1]),
                                         area=pos[2],
                                         sigma=sigma,
                                         eps=epsilon,
                                         displace=[0., DeltaPhi / 2., DeltaZ / 2.],
                                         iccTypeID=iccTypeID,
                                         type=particleTypeID,
                                         fix=[1, 1, 1],
                                         f=[0, 0, 0])


    def calcCylPart(self, phi, z, dxdydz):
        '''
            calculate positions for the cylinder part. Area is calculated by 4 * R * dphi * dz
        '''

        pos = np.array([self.radius * np.cos(phi),
                        self.radius * np.sin(phi), z])
        norm = self.direction * pos.copy()
        norm[2] = 0.
        return [pos + self.center, norm / np.sqrt(np.sum(np.square(norm))), 4. * self.radius * dxdydz[1] * dxdydz[2]]
