import numpy as np
from numpy.linalg import inv


class SetupAdaptiveICCWall(object):

    def __init__(self, system, )
 setupAdaptiveICCWall(system, icc, dist, normal, nWall, cutoff, initCharge, sigma, eps, particleType, transMatrix=None, invMatrix=None):
    normal = np.array(normal, dtype=float) / np.sqrt(np.sum(np.square(normal)))
    pos = normal * dist

    if transMatrix is not None and invMatrix is not None:
        # use given matrices
        if not (hasattr(transMatrix, '__iter__') or hasattr(invMatrix, '__iter__')):
            raise TypeError('transformation matrices have to be arrays!')

        transMatrix = np.array(transMatrix, dtype=float)
        invMatrix = np.array(invMatrix, dtype=float)
        if transMatrix.shape != (3, 3) or invMatrix.shape != (3, 3):
            raise ValueError('Transfomation matrices have to be float arrays of shape (3, 3)')
    else:
        # calculate transformation matix
        # plane axis is ez
        ez = normal / np.sqrt(np.sum(np.square(normal)))

        # Find a vector orthogonal to ez, since {1,0,0} and {0,1,0} are independent, ez can not be parallel to both of them. Then we can do Gram-Schmidt
        if (np.dot(np.array([1., 0., 0.]), ez) < 1.):
            ex = np.array([1., 0., 0.]) - np.dot(ez, np.array([1., 0., 0.])) * ez
        else:
            ex = np.array([0., 1., 0.]) - np.dot(ez, np.array([0., 1., 0.])) * ez

        ex = ex / np.sqrt(np.sum(np.square(ex)))

        # since ez and ex are normalized, ey is normalized too
        ey = np.cross(ez, ex)

        transMatrix = np.column_stack((ex, ey, ez))

        try:
            invMatrix = inv(transMatrix)
        except np.linalg.LinAlgError:
            # this error cannot appear since we construct the matric ourselfes
            raise ValueError(
                "Calculated transformation matrix is not invertible!")

    if all(normal == [0., 0., 1.]):
        useTrans = False
    else:
        useTrans = True
    
    print(transMatrix)
    print(invMatrix)
    # register types to ICC!
    wallID = icc.addTypeWall(_normal=normal,
                             _dist=dist,
                             _cutoff=cutoff,
                             _useTrans=useTrans,
                             _transMatrix=transMatrix.flatten(),
                             _invMatrix=invMatrix.flatten())

    # init particles
    dx = system.box_l[0] / nWall
    dy = system.box_l[1] / nWall
    area = dx * dy
    for i in range(nWall):
        x = (i + 0.5) * dx
        for j in range(nWall):
            y = (j + 0.5) * dy
            system.part.add(pos=np.dot(transMatrix, pos + [x, y, 0.]),
                            q=initCharge,
                            normal=normal,
                            area=area,
                            sigma=sigma,
                            eps=eps,
                            displace=[dx/2., dy/2., 0.],
                            type=particleType,
                            iccTypeID=wallID)
    
