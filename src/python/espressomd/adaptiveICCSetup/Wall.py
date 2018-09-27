import numpy as np
import pickle
from numpy.linalg import inv


class SetupWall(object):

    def __init__(self, dist, normal, nICC, transMatrix=None, invMatrix=None, typeIDoffset=None):

        if hasattr(normal, '__iter__'):
            self.normal = np.array(normal, dtype=float)

            if self.normal.size != 3:
                raise ValueError('normal argument should be an array of size 3')
        else:
            raise TypeError('normal argument should be an array of size 3')

        if not (isinstance(dist, float) or isinstance(dist, int)):
            raise TypeError('dist argument has to be a float!')

        if not isinstance(nICC, int):
            raise TypeError('number of icc particles has to be an int')

        self.dist = dist
        self.nICC = nICC
        self.pos = self.normal / np.sqrt(np.sum(np.square(self.normal))) * self.dist
        self.splitCutoff = 0.

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
            ez = self.normal / np.sqrt(np.sum(np.square(self.normal)))

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

    def registerTypes(self, icc):

        if all(self.normal == [0., 0., 1.]):
            self.useTrans = False
        else:
            self.useTrans = True

        # register types to ICC!
        self.typeIDoffset = icc.addTypeWall(_cutoff=self.splitCutoff,
                                 _useTrans=self.useTrans,
                                 _transMatrix=self.transMatrix.flatten(),
                                 _invMatrix=self.invMatrix.flatten())


    def initParticles(self, system, particleTypeID, iccTypeID, initCharge, sigma, epsilon, splitCutoff=0.):
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


        dx = system.box_l[0] / self.nICC
        dy = system.box_l[1] / self.nICC

        area = system.box_l[0] * system.box_l[1] / (self.nICC * self.nICC)

        for i in range(self.nICC):
            x = (i + 0.5) * dx
            for j in range(self.nICC):
                y = (j + 0.5) * dy
                system.part.add(pos=np.dot(self.transMatrix, self.pos + [x, y, 0.]),
                                         q=initCharge,
                                         normal=np.array([0., 0., 1.]),
                                         area=area,
                                         sigma=sigma,
                                         eps=epsilon,
                                         displace=[dx / 2., dy / 2., 0.],
                                         iccTypeID=iccTypeID,
                                         type=particleTypeID,
                                         fix=[1, 1, 1])
