import numpy as np
from numpy.linalg import inv

class SetupSimplePore(object):

  def __init__(self, center, axis, length, radius, smoothing_radius, nCylinderPhi, nCylinderZ, nTorusPhi, nTorusZ, transMatrix=None, invMatrix=None):

    if hasattr(center, '__iter__'):
      self.center = np.array(center)

      if self.center.size != 3:
        raise ValueError('center argument should be an array of size 3')
    else:
      # this is not fool proof!
      raise ValueError('center argument should be an array of size 3')

    if hasattr(axis, '__iter__'):
      self.axis = np.array(axis)

      if self.axis.size != 3:
        raise ValueError('axis argument should be an array of size 3')
    else:
      # this is not fool proof!
      raise ValueError('axis argument should be an array of size 3')

    if not (isinstance(length, float) or isinstance(length, int)):
      raise ValueError('length argument has to be a float!')

    if not (isinstance(radius, float) or isinstance(radius, int)):
      raise ValueError('radius argument has to be a float!')

    if not (isinstance(smoothing_radius, float) or isinstance(smoothing_radius, int)):
      raise ValueError('smoothing_radius argument has to be a float!')

    if not isinstance(nCylinderPhi, int):
      raise ValueError('Number of ICC cylinder angle particles has to be an int!')

    if not isinstance(nCylinderZ, int):
      raise ValueError('Number of ICC cylinder length particles has to be an int!')

    if not isinstance(nTorusPhi, int):
      raise ValueError('Number of ICC torus angle particles has to be an int!')

    if not isinstance(nTorusZ, int):
      raise ValueError('Number of ICC torus length particles has to be an int!')

    self.length = length
    self.radius = radius
    self.smoothing_radius = smoothing_radius
    self.nCylinderPhi = nCylinderPhi
    self.nCylinderZ = nCylinderZ
    self.nTorusPhi = nTorusPhi
    self.nTorusZ = nTorusZ
    self.lengthInner = self.length - 2 * self.smoothing_radius
    self.lengthInnerHalf = 0.5 * self.lengthInner
    self.radiusOuter = self.radius + self.smoothing_radius
    self.radiusOuter2 = self.radiusOuter ** 2.
    self.typeIDoffset = 0

    if transMatrix is not None and invMatrix is not None:
      # use given matrices
      self.transMatrix = np.array(transMatrix)
      self.invMatrix = np.array(invMatrix)
      if self.transMatrix.shape != (3, 3) or self.invMatrix.shape != (3, 3):
        raise ValueError('Transfomation matrices have to be float arrays of shape (3, 3)')
    else:
      # calculate transformation matix

      # cylinder axis is ez
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
          raise ValueError("Calculated transformation matrix is not invertible!")


  def registerTypes(self, icc):

        self.useTrans = True

        if all(self.axis == [0., 0., 1.]):
            self.useTrans = False


        icc.addTypeWall(_cutoff=self.splitCutoff[0],
                        _useTrans=self.useTrans,
                        _transMatrix=self.transMatrix.flatten(),
                        _invMatrix=self.invMatrix.flatten())

        icc.addTypeCylinder(_center=self.center,
                            _axis=self.axis,
                            _length=self.length,
                            _radius=self.radius,
                            _direction=-1.,
                            _cutoff=self.splitCutoff[1],
                            _useTrans=self.useTrans,
                            _transMatrix=self.transMatrix.flatten(),
                            _invMatrix=self.invMatrix.flatten())

        icc.addTypeTorus(_center=self.center,
                         _axis=self.axis,
                         _length=self.length,
                         _radius=self.radius,
                         _smoothingRadius=self.smoothing_radius,
                         _cutoff=self.splitCutoff[2],
                         _useTrans=self.useTrans,
                         _transMatrix=self.transMatrix.flatten(),
                         _invMatrix=self.invMatrix.flatten())

        icc.addTypeInterface(_center=self.center,
                             _radius=self.radius,
                             _smoothingRadius=self.smoothing_radius,
                             _cutoff=[self.splitCutoff[3][0], 0, 0],
                             _useTrans=self.useTrans,
                             _transMatrix=self.transMatrix.flatten(),
                             _invMatrix=self.invMatrix.flatten())

        a = icc.addTypeInterface(_center=self.center,
                             _radius=self.radius,
                             _smoothingRadius=self.smoothing_radius,
                             _cutoff=[0, self.splitCutoff[3][1], 0],
                             _useTrans=self.useTrans,
                             _transMatrix=self.transMatrix.flatten(),
                             _invMatrix=self.invMatrix.flatten())
        print(a)

  def initParticles(self, system, TypeID, initCharge, sigma, epsilon, splitCutoff=0):
    '''
      initialize all particles for given parts. Available parts are "cylinder", "torus", "wall" and "interface"

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
      raise ValueError('Initial particle charge has to be a float!')

    if not (isinstance(sigma, float) or isinstance(sigma, int)):
      raise ValueError('sigma value has to be a float!')

    if not (isinstance(epsilon, float) or isinstance(epsilon, int)):
      raise ValueError('epsilon value has to be a float!')

    self.splitCutoff = splitCutoff


    if self.nCylinderPhi > 0 and self.nCylinderZ > 0:
        DeltaPhi = 2. * np.pi / self.nCylinderPhi
        DeltaZ = self.lengthInner / self.nCylinderZ

        # cylinder particles
        for Iphi in range(self.nCylinderPhi):
            phi = (Iphi + 0.5) * DeltaPhi
            for IZ in range(-int(self.nCylinderZ / 2), int(self.nCylinderZ / 2) + self.nCylinderZ % 2):
                z = IZ * DeltaZ

                pos = self.calcCylPart(phi, z, np.array([0., DeltaPhi / 2., DeltaZ / 2.]))

                system.part.add(pos=np.dot(self.transMatrix, pos[0]),
                                q=initCharge,
                                normal=np.dot(self.transMatrix, pos[1]),
                                area=pos[2],
                                sigma=sigma,
                                eps=epsilon,
                                displace=np.array([0., DeltaPhi / 2., DeltaZ / 2.]),
                                type=TypeID,
                                iccTypeID=1)


    if self.nTorusPhi > 0 and self.nTorusZ > 0:
        DeltaPhi = 2. * np.pi / self.nTorusPhi
        DeltaZ = self.smoothing_radius / self.nTorusZ

        for Iphi in range(self.nTorusPhi):
            phi = (Iphi + 0.5) * DeltaPhi
            for IZ in range(self.nTorusZ):
                z = (IZ + 0.5) * DeltaZ + self.lengthInnerHalf

                pos = self.calcTorusPart(phi, z, np.array([0., DeltaPhi /2., DeltaZ /2.]))

                system.part.add(pos=np.dot(self.transMatrix, pos[0]),
                                q=initCharge,
                                normal=np.dot(self.transMatrix, pos[1]),
                                area=pos[2],
                                sigma=sigma,
                                eps=epsilon,
                                displace=np.array([0., DeltaPhi / 2., DeltaZ / 2.]),
                                type=TypeID,
                                iccTypeID=2)

                pos[0][2] = pos[0][2] - self.lengthInner - 2 * (IZ + 0.5) * DeltaZ
                pos[1][2] = -pos[1][2]

                system.part.add(pos=np.dot(self.transMatrix, pos[0]),
                                q=initCharge,
                                normal=np.dot(self.transMatrix, pos[1]),
                                area=pos[2],
                                sigma=sigma,
                                eps=epsilon,
                                displace=np.array([0., DeltaPhi / 2., DeltaZ / 2.]),
                                type=TypeID,
                                iccTypeID=2)

    # wall particles
    hpx = (system.box_l[0] - 2. * self.radiusOuter) / 2.
    hpy = (system.box_l[1] - 2. * self.radiusOuter) / 2.
    offsetx = self.radiusOuter + hpx / 2.
    offsety = self.radiusOuter + hpy / 2.

    arx = 2. * hpx * self.radiusOuter
    ary = 2. * hpy * self.radiusOuter

    # x-direction, set largest possible rectangle and the place one particle in the leftover area
    div, rem = divmod(hpx, 2. * self.radiusOuter)
    offset = 2. * self.radiusOuter

    for _ in range(int(div)):
        for fact in [-1., 1.]:
            system.part.add(pos=np.dot(self.transMatrix, self.center + [fact * offset, 0, self.length / 2.]),
                            q=initCharge,
                            normal=np.array([0., 0., 1.]),
                            area=arx,
                            sigma=sigma,
                            eps=epsilon,
                            displace=np.array([self.radiusOuter, self.radiusOuter, 0.]),
                            type=TypeID,
                            iccTypeID=0)

            system.part.add(pos=np.dot(self.transMatrix, self.center + [fact * offset, 0, -self.length / 2.]),
                            q=initCharge,
                            normal=np.array([0., 0., -1.]),
                            area=arx,
                            sigma=sigma,
                            eps=epsilon,
                            displace=np.array([self.radiusOuter, self.radiusOuter, 0.]),
                            type=TypeID,
                            iccTypeID=0)

        offset += 2. * self.radiusOuter

    if rem > 0.:
        offset -= self.radiusOuter
        offset += rem / 2.
        for fact in [-1., 1.]:
            system.part.add(pos=np.dot(self.transMatrix, self.center + [fact * offset, 0, self.length / 2.]),
                                     q=initCharge,
                                     normal=np.array([0., 0., 1.]),
                                     area=arx,
                                     sigma=sigma,
                                     eps=epsilon,
                                     displace=np.array([rem / 2., self.radiusOuter, 0.]),
                                     type=TypeID,
                                     iccTypeID=0)

            system.part.add(pos=np.dot(self.transMatrix, self.center + [fact * offset, 0, -self.length / 2.]),
                                     q=initCharge,
                                     normal=np.array([0., 0., -1.]),
                                     area=arx,
                                     sigma=sigma,
                                     eps=epsilon,
                                     displace=np.array([rem / 2., self.radiusOuter, 0.]),
                                     type=TypeID,
                                     iccTypeID=0)


    # y-direction
    div, rem = divmod(hpy, 2. * self.radiusOuter)
    offset = 2. * self.radiusOuter

    for _ in range(int(div)):
        for fact in [-1., 1.]:
            system.part.add(pos=np.dot(self.transMatrix, self.center + [0., fact * offset, self.length / 2.]),
                                     q=initCharge,
                                     normal=np.array([0., 0., 1.]),
                                     area=arx,
                                     sigma=sigma,
                                     eps=epsilon,
                                     displace=np.array([self.radiusOuter, self.radiusOuter, 0.]),
                                     type=TypeID,
                                     iccTypeID=0)

            system.part.add(pos=np.dot(self.transMatrix, self.center + [0., fact * offset, -self.length / 2.]),
                                     q=initCharge,
                                     normal=np.array([0., 0., -1.]),
                                     area=arx,
                                     sigma=sigma,
                                     eps=epsilon,
                                     displace=np.array([self.radiusOuter, self.radiusOuter, 0.]),
                                     type=TypeID,
                                     iccTypeID=0)

        offset += 2. * self.radiusOuter

    if rem > 0.:
        offset -= self.radiusOuter
        offset += rem / 2.
        for fact in [-1., 1.]:
            system.part.add(pos=np.dot(self.transMatrix, self.center + [0., fact * offset, self.length / 2.]),
                            q=initCharge,
                            normal=np.array([0., 0., 1.]),
                            area=arx,
                            sigma=sigma,
                            eps=epsilon,
                            displace=np.array([self.radiusOuter, rem / 2., 0.]),
                            type=TypeID,
                            iccTypeID=0)

            system.part.add(pos=np.dot(self.transMatrix, self.center + [0., fact * offset, -self.length / 2.]),
                                     q=initCharge,
                                     normal=np.array([0., 0., -1.]),
                                     area=arx,
                                     sigma=sigma,
                                     eps=epsilon,
                                     displace=np.array([self.radiusOuter, rem / 2., 0.]),
                                     type=TypeID,
                                     iccTypeID=0)

    # diagonal areas
    ar = hpx * hpy

    for dx, dy in [[offsetx, offsety], [-offsetx, offsety], [offsetx, -offsety], [-offsetx, -offsety]]:
        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., 1.]),
                                 area=ar,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array(
                                     [hpx / 2., hpy / 2., 0.]),
                                 type=TypeID,
                                 iccTypeID=0)

        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, -self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., -1.]),
                                 area=ar,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array(
                                     [hpx / 2., hpy / 2., 0.]),
                                 type=TypeID,
                                 iccTypeID=0)

    # side length of the squares in the corner of the interfaces
    a = (1. - np.sqrt(2) / 2.) * self.radiusOuter
    area = a ** 2.
    offset = self.radiusOuter - a / 2.

    for dx, dy in [[offset, offset], [-offset, offset], [offset, -offset], [-offset, -offset]]:
        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., 1.]),
                                 area=area,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array([a / 2., a / 2., 0.]),
                                 type=TypeID,
                                 iccTypeID=0)

        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, - self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., -1.]),
                                 area=area,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array([a / 2., a / 2., 0.]),
                                 type=TypeID,
                                 iccTypeID=0)

    # interface particles
    ar = 0.5 * self.radiusOuter2 * (np.sqrt(2) - 0.5 - 0.25 * np.pi)
    xs = np.sqrt(2) / 4. * self.radiusOuter
    ys = (1. - 0.5 * (1 - np.sqrt(7. / 8.))) * self.radiusOuter

    for dx, dy in [[xs, ys], [xs, -ys], [-xs, ys], [-xs, -ys]]:
        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, self.length / 2.]),
                        q=initCharge,
                        normal=np.array([0., 0., 1.]),
                        area=ar,
                        sigma=sigma,
                        eps=epsilon,
                        displace=np.array([xs, 0., 0.]),
                        type=TypeID,
                        iccTypeID=3)

        system.part.add(pos=np.dot(self.transMatrix, self.center + [dx, dy, -self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., -1.]),
                                 area=ar,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array([xs, 0., 0.]),
                                 type=TypeID,
                                 iccTypeID=3)

        system.part.add(pos=np.dot(self.transMatrix, self.center + [dy, dx, self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., 1.]),
                                 area=ar,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array([0., xs, 0.]),
                                 type=TypeID,
                                 iccTypeID=4)

        system.part.add(pos=np.dot(self.transMatrix, self.center + [dy, dx, -self.length / 2.]),
                                 q=initCharge,
                                 normal=np.array([0., 0., -1.]),
                                 area=ar,
                                 sigma=sigma,
                                 eps=epsilon,
                                 displace=np.array([0., xs, 0.]),
                                 type=TypeID,
                                 iccTypeID=4)




  ##########
  # Cylinder Functions
  ##########

  def calcCylPart(self, phi, z, dxdydz):
    '''
    calculate positions for the cylinder part. Area is calculated by 4 * R * dphi * dz
    '''

    pos = np.array([self.radius * np.cos(phi), self.radius * np.sin(phi), z])
    norm = - pos.copy()
    norm[2] = 0.
    return [pos + self.center, norm / np.sqrt(np.sum(np.square(norm))), 4. * self.radius * dxdydz[1] * dxdydz[2]]

  def splitExtCylinder(self, ppos, dxdydz):
    '''
    split function for cylinder part. Transfomation to cylinder coordinates
    '''

    ppos = ppos - self.center
    # numerical precision error causes values bigger than 1, which cant be evaluated -> clipping
    phi = np.arctan2(np.clip(ppos[1] / self.radius, -1., 1.), np.clip(ppos[0] / self.radius, -1., 1.))

    return [self.calcCylPart(phi + dxdydz[1], ppos[2] + dxdydz[2], dxdydz),
            self.calcCylPart(phi + dxdydz[1], ppos[2] - dxdydz[2], dxdydz),
            self.calcCylPart(phi - dxdydz[1], ppos[2] + dxdydz[2], dxdydz),
            self.calcCylPart(phi - dxdydz[1], ppos[2] - dxdydz[2], dxdydz)]

  def reduceExtCylinder(self, ppos, dxdydz):
    '''
    merge function for cylinder part. Transfomation to cylinder coordinates
    '''

    ppos = ppos - self.center

    phi = np.arctan2(np.clip(ppos[1] / self.radius, -1., 1.), np.clip(ppos[0] / self.radius, -1., 1.))

    return self.calcCylPart(phi - dxdydz[1], ppos[2] - dxdydz[2], dxdydz)


  ##########
  # Torus Segment Functions
  ##########

  def calcTorusPart(self, phi, z, dxdydz):
    '''
    calculate positions for the torus segment particles. Area is calculated via surface integration
    '''

    temp = np.sqrt(np.clip(self.smoothing_radius ** 2. - (np.abs(z) - self.lengthInnerHalf)**2, 0., np.inf))

    radiusTorus = self.radiusOuter - temp
    pos = np.array([radiusTorus * np.cos(phi), radiusTorus * np.sin(phi), z])
    if temp > 0.:
      # particle is on the torus segment with a finite partial derivative
      norm = - pos.copy()
      norm[2] = np.sign(z) * (np.abs(z) - self.lengthInnerHalf) / temp * radiusTorus
    else:
      # position is on outer position, normal vector is cylinder axis
      norm = np.array([0., 0., 1.])
    return [pos + self.center, norm / np.sqrt(np.sum(np.square(norm))), 2 * (self.TorusArea(np.abs(z) + dxdydz[2]) - self.TorusArea(np.abs(z) - dxdydz[2])) * dxdydz[1]]

  def TorusArea(self, z):
    '''
    Calculate area for the torus segment for a gizen z
    '''
    temp = self.smoothing_radius ** 2. - (np.abs(z) - self.lengthInnerHalf) ** 2.

    # move position relative to the torus center
    zprime = np.abs(z) - self.lengthInnerHalf
    if temp > 0.:
      # suface integration
      temp = np.sqrt(temp)
      return 2 * self.radius * np.abs(z) - 0.5 * (zprime * temp + self.smoothing_radius ** 2. * np.arctan(zprime / temp))
    else:
      # area is approaching the infinite partial derivate part
      return 2 * self.radius * np.abs(z) - 0.25 * self.smoothing_radius ** 2. * np.pi

  def splitExtTorus(self, ppos, dxdydz):
    '''
    splitting function for the torus particles. Cylindrical coordinates are used
    '''

    ppos = ppos - self.center
    # calculate radius for that z position
    r = self.radiusOuter - np.sqrt(np.clip(self.smoothing_radius ** 2. - (np.abs(ppos[2]) - self.lengthInnerHalf) ** 2., 0., np.inf))
    phi = np.arctan2(ppos[1] / r, ppos[0] / r)
    return [self.calcTorusPart(phi + dxdydz[1], ppos[2] + dxdydz[2], dxdydz),
            self.calcTorusPart(phi + dxdydz[1], ppos[2] - dxdydz[2], dxdydz),
            self.calcTorusPart(phi - dxdydz[1], ppos[2] + dxdydz[2], dxdydz),
            self.calcTorusPart(phi - dxdydz[1], ppos[2] - dxdydz[2], dxdydz)]

  def reduceExtTorus(self, ppos, dxdydz):
    '''
    merging function for the torus particles. Cylindrical coordinates are used
    '''

    ppos = ppos - self.center
    r = self.radiusOuter - np.sqrt(np.clip(self.smoothing_radius ** 2. - (np.abs(ppos[2]) - self.lengthInnerHalf) ** 2., 0., np.inf))
    phi = np.arctan2(ppos[1] / r, ppos[0] / r)
    return self.calcTorusPart(phi - dxdydz[1], ppos[2] - dxdydz[2], dxdydz)

  ##########
  # Wall Functions
  ##########

  def splitExtPlane(self, ppos, dxdydz):
    '''
    splitting function for the wall particles
    '''
    if ppos[2] - self.center[2] > 0:
      norm = np.array([0., 0., 1.])
    else:
      norm = np.array([0., 0., -1.])
    return [[ppos + [dxdydz[0], dxdydz[1], 0.], norm, 4 * dxdydz[0] * dxdydz[1]],
            [ppos + [-dxdydz[0], dxdydz[1], 0.], norm, 4 * dxdydz[0] * dxdydz[1]],
            [ppos + [dxdydz[0], -dxdydz[1], 0.], norm, 4 * dxdydz[0] * dxdydz[1]],
            [ppos + [-dxdydz[0], -dxdydz[1], 0.], norm, 4 * dxdydz[0] * dxdydz[1]]]

  def reduceExtPlane(self, ppos, dxdydz):
    '''
    merging funciton for the wall particles
    '''

    if ppos[2] - self.center[2] > 0:
      norm = np.array([0., 0., 1.])
    else:
      norm = np.array([0., 0., -1.])
    return [ppos - [dxdydz[0], dxdydz[1], 0.], norm]


  ##########
  # Inteface Functions
  ##########

  def calcInterfaceArea(self, x0, x1):
    '''
      calculates area for the interface particles via CylROuter * (x1 - x0) - 0.5 * (x1 * temp2 - x0 * temp1 + CylROuter**2 * (np.arctan(x1 / temp2) - np.arctan(x0 / temp1)))
    '''

    if x0 > x1: # make sure x1 is always bigger
      a = x1
      x1 = x0
      x0 = a
    out = self.radiusOuter * (x1 - x0)
    temp1 = self.radiusOuter2 - x0 ** 2.
    temp2 = self.radiusOuter2 - x1 ** 2.

    if temp1 > 0.:
      temp1 = np.sqrt(temp1)
      out += 0.5 * (x0 * temp1 + self.radiusOuter2 * np.arctan(x0 / temp1))
    else:
      out += 0.25 * seld.radiusOuter ** 2. * np.pi

    if temp2 > 0.:
      temp2 = np.sqrt(temp2)
      out -= 0.5 * (x1 * temp2 + self.radiusOuter2 * np.arctan(x1 / temp2))
    else:
      out -= 0.25 * self.radiusOuter2 * np.pi

    return out


  def splitExtInterface(self, ppos, dxdydz):
    '''
    splitting interface function
    '''
    ppos = ppos - self.center

    if ppos[2] > 0:
      norm = np.array([0., 0., 1.])
    else:
      norm = np.array([0., 0., -1.])

    if dxdydz[0] > 0.:
      # horizontal displacement
      newpos = ppos.copy()
      newpos[0] += dxdydz[0]
      newpos[1] = 0.5 * np.sign(ppos[1]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos[0] ** 2., 0, np.inf)))

      newpos1 = ppos.copy()
      newpos1[0] -= dxdydz[0]
      newpos1[1] = 0.5 * np.sign(ppos[1]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos1[0] ** 2., 0, np.inf)))

      return [[newpos + self.center, norm, self.calcInterfaceArea(ppos[0], ppos[0] + 2 * dxdydz[0])],
            [newpos1 + self.center, norm, self.calcInterfaceArea(ppos[0] - 2 * dxdydz[0], ppos[0])]]
    else:
      # vertical displacement
      newpos = ppos.copy()
      newpos[1] += dxdydz[1]
      newpos[0] = 0.5 * np.sign(ppos[0]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos[1] ** 2., 0, np.inf)))

      newpos1 = ppos.copy()
      newpos1[1] -= dxdydz[1]
      newpos1[0] = 0.5 * np.sign(ppos[0]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos1[1] ** 2., 0, np.inf)))
      return [[newpos + self.center, norm, self.calcInterfaceArea(ppos[1], ppos[1] + 2 * dxdydz[1])],
            [newpos1 + self.center, norm, self.calcInterfaceArea(ppos[1] - 2 * dxdydz[1], ppos[1])]]

  def reduceExtInterface(self, ppos, dxdydz):
    '''
    merging function of interface particles
    '''
    ppos = ppos - self.center

    if ppos[2] > 0:
      norm = np.array([0., 0., 1.])
    else:
      norm = np.array([0., 0., -1.])

    if dxdydz[0] > 0.:
      newpos = ppos.copy()
      newpos[0] -= dxdydz[0]
      newpos[1] = 0.5 * np.sign(ppos[1]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos[0] ** 2., 0, np.inf)))

    else:
      newpos = ppos.copy()
      newpos[1] -= dxdydz[1]
      newpos[0] = 0.5 * np.sign(ppos[0]) * (self.radiusOuter + np.sqrt(np.clip(self.radiusOuter2 - newpos[1] ** 2., 0, np.inf)))

    return [newpos + self.center, norm]
