import re
import numpy as np


class Parameter:
    def __init__(self, num, value, trainable):
        self.num = num
        self.value = value
        self.trainable = trainable

    def read_parameter(self, line):
        line = re.split(r'\s+', line)
        self.num = line[0].strip(')')
        self.value = float(line[1])
        self.trainable = bool(int(line[2]))

    def __repr__(self) -> str:
        return f"{self.num}) {self.value} {int(self.trainable)}"

    def __str__(self) -> str:
        return f"{self.num}) {self.value} {int(self.trainable)}"


class DoubleParam(Parameter):
    def __init__(self, num, x, y, trainable_x, trainable_y):
        Parameter.__init__(self, num, (x, y), (trainable_x, trainable_y))

    def read_parameter(self, line):
        line = re.split(r'\s+', line)
        self.num = line[0].strip(')')
        self.value = (float(line[1]), float(line[2]))
        self.trainable = (bool(int(line[3])), bool(int(line[4])))

    def __repr__(self) -> str:
        return f"{self.num}) {self.value[0]} {self.value[1]} {int(self.trainable[0])} {int(self.trainable[1])}"

    def __str__(self) -> str:
        return f"{self.num}) {self.value[0]} {self.value[1]} {int(self.trainable[0])} {int(self.trainable[1])}"


class StrParam(Parameter):
    def __init__(self, num, value):
        Parameter.__init__(self, num, value, False)

    def read_parameter(self, line):
        line = re.split(r'\s+', line, 1)
        self.num = line[0].strip(')')
        self.value = line[1].split('#')[0].rstrip()

    def __repr__(self) -> str:
        return f"{self.num}) {self.value}"

    def __str__(self) -> str:
        return f"{self.num}) {self.value}"


class Component:
    def __init__(self, type):
        self.__type__ = type
        self.__parameters__ = []
        self.__param_index__ = {}
        self.__output_option__ = StrParam('Z', 0)

    @property
    def output_option(self):
        return self.__output_option__.value

    @output_option.setter
    def output_option(self, option: bool):
        self.__output_option__.value = int(option)

    def read(self, file):
        line = file.readline()
        while line:
            line = line.lstrip()
            pos = line.find(')')
            if len(line) > 0 and pos > 0:
                if line[0] == '0':
                    file.seek(file.tell()-len(line))
                    break
                if line[:pos] in self.__param_index__:
                    self.__parameters__[self.__param_index__[
                        line[:pos]]].read_parameter(line)
            line = file.readline()
        return file

    def __repr__(self) -> str:
        s = f"0) {self.__type__}\n"
        for parameter in self.__parameters__:
            s += parameter.__repr__() + '\n'
        return s

    def set_output_option(self, option):
        # option: bool
        # Outputting image options, the options are:
        #   0 = normal, i.e. subtract final model from the data to create the residual image
        #   1 = Leave in the model -- do not subtract from the data
        self.__output_option__.value = option


class Anisotropic(Component):
    def __init__(self, type):
        # super(__Anisotropic, self).__init__(type)
        Component.__init__(self, type)
        self.__position__ = DoubleParam(1, 0, 0, True, True)
        self.__position_angle__ = Parameter(10, 0, True)

    @property
    def position(self):
        return self.__position__.value

    @property
    def position_angle(self):
        return self.__position_angle__.value

    @position.setter
    def position(self, position: tuple):
        self.__position__.value = position

    @position_angle.setter
    def position_angle(self, angle: float):
        self.__position_angle__.value = angle

    def set_position(self, x: float = None, y: float = None, trainable_x=True, trainable_y=True):
        """
        Set initial position
        :param x: float, x position [pixels]
        :param y: float, y position [pixels]
        :param trainable_x: bool, indicates whether the param x are trainable
        :param trainable_y: bool, indicates whether the param y are trainable
        """
        if x is not None and y is not None:
            self.__position__.value = (x, y)
        elif x is not None:
            self.__position__.value = (x, self.__position__.value[1])
        elif y is not None:
            self.__position__.value = (self.__position__.value[0], y)
        self.__position__.trainable = (trainable_x, trainable_y)

    def set_position_angle(self, angle: float, trainable=True):
        """
        Set initial position angle
        :param angle: float, position angle [degrees: Up=0, Left=90]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        self.__position_angle__.value = angle
        self.__position_angle__.trainable = trainable


class Sersic(Anisotropic):
    state_num = 4

    def __init__(self):
        super(Sersic, self).__init__("sersic")
        # __Anisotropic.__init__(self, "sersic")
        self.__magnitude__ = Parameter(3, 0, True)
        self.__effective_radius__ = Parameter(4, 0, True)
        self.__sersic_index__ = Parameter(5, 1, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__magnitude__,
                               self.__effective_radius__, self.__sersic_index__,
                               self.__axis_ratio__, self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '5': 3, '9': 4, '10': 5, 'Z': 6}

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @property
    def effective_radius(self):
        return self.__effective_radius__.value

    @property
    def sersic_index(self):
        return self.__sersic_index__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    @effective_radius.setter
    def effective_radius(self, radius: float):
        self.__effective_radius__.value = radius

    @sersic_index.setter
    def sersic_index(self, index: float):
        self.__sersic_index__.value = index

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    @property
    def state(self):
        if self.__sersic_index__.trainable:
            return 0
        elif self.sersic_index == 1:
            return 1
        elif self.sersic_index == 4:
            return 2
        elif self.sersic_index == 0.5:
            return 3
        else:
            return -1

    @state.setter
    def state(self, state: int):
        if state == 0:
            self.set_sersic_index(None, True)
        elif state == 1:
            self.set_sersic_index(1, False)
        elif state == 2:
            self.set_sersic_index(4, False)
        elif state == 3:
            self.set_sersic_index(0.5, False)

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable

    def set_effective_radius(self, radius: float = None, trainable=True):
        """
        Set initial effective radius
        :param radius: float, effective radius [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if radius is not None:
            self.__effective_radius__.value = radius
        self.__effective_radius__.trainable = trainable

    def set_sersic_index(self, index: float = None, trainable=True):
        """
        Set initial Sersic index
        :param index: float, Sersic index (de Vaucouleurs n=4, expdisk=1)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if index is not None:
            self.__sersic_index__.value = index
        self.__sersic_index__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio
        :param ratio: float, axis ratio (b/a)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class Nuker(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "nuker")
        self.__surface_brightness__ = Parameter(3, 0, True)
        self.__break_radius__ = Parameter(4, 0, True)
        self.__alpha__ = Parameter(5, 0, True)
        self.__beta__ = Parameter(6, 0, True)
        self.__gamma__ = Parameter(7, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__surface_brightness__,
                               self.__break_radius__, self.__alpha__, self.__beta__,
                               self.__gamma__, self.__axis_ratio__, self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1, '4': 2, '5': 3,
                                '6': 4, '7': 5, '9': 6, '10': 7, 'Z': 8}

    @property
    def surface_brightness(self):
        return self.__surface_brightness__.value

    @property
    def break_radius(self):
        return self.__break_radius__.value

    @property
    def alpha(self):
        return self.__alpha__.value

    @property
    def beta(self):
        return self.__beta__.value

    @property
    def gamma(self):
        return self.__gamma__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @surface_brightness.setter
    def surface_brightness(self, mu_rb: float):
        self.__surface_brightness__.value = mu_rb

    @break_radius.setter
    def break_radius(self, rb: float):
        self.__break_radius__.value = rb

    @alpha.setter
    def alpha(self, alpha: float):
        self.__alpha__.value = alpha

    @beta.setter
    def beta(self, beta: float):
        self.__beta__.value = beta

    @gamma.setter
    def gamma(self, gamma: float):
        self.__gamma__.value = gamma

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_surface_brightness(self, mu_rb: float = None, trainable=True):
        """
        Set initial surface brightness at Rb
        :param mu_rb: float, surface brightness at Rb [mag/arcsec^2]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if mu_rb is not None:
            self.__surface_brightness__.value = mu_rb
        self.__surface_brightness__.trainable = trainable

    def set_radius_break(self, rb: float = None, trainable=True):
        """
        Set initial radius at break
        :param rb: float, radius at break [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if rb is not None:
            self.__break_radius__.value = rb
        self.__break_radius__.trainable = trainable

    def set_alpha(self, alpha: float = None, trainable=True):
        """
        Set initial alpha (sharpness of transition)
        :param alpha: float, sharpness of transition
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if alpha is not None:
            self.__alpha__.value = alpha
        self.__alpha__.trainable = trainable

    def set_beta(self, beta: float = None, trainable=True):
        """
        Set initial beta (outer powerlaw slope)
        :param beta: float, outer powerlaw slope
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if beta is not None:
            self.__beta__.value = beta
        self.__beta__.trainable = trainable

    def set_gamma(self, gamma: float = None, trainable=True):
        """
        Set initial gamma (inner powerlaw slope)
        :param gamma: float, inner powerlaw slope
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if gamma is not None:
            self.__gamma__.value = gamma
        self.__gamma__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio
        :param ratio: float, axis ratio (b/a)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class ExpDisk(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "expdisk")
        self.__magnitude__ = Parameter(3, 0, True)
        self.__effective_radius__ = Parameter(4, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__magnitude__,
                               self.__effective_radius__, self.__axis_ratio__,
                               self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '9': 3, '10': 4, 'Z': 5}

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @property
    def effective_radius(self):
        return self.__effective_radius__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    @effective_radius.setter
    def effective_radius(self, radius: float):
        self.__effective_radius__.value = radius

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable

    def set_effective_radius(self, radius: float = None, trainable=True):
        """
        Set initial effective radius
        :param radius: float, effective radius [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if radius is not None:
            self.__effective_radius__.value = radius
        self.__effective_radius__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio
        :param ratio: float, axis ratio (b/a)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class EdgeDisk(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "edgedisk")
        self.__central_surface_brightness__ = Parameter(3, 0, True)
        self.__scale_height__ = Parameter(4, 0, True)
        self.__scale_length__ = Parameter(5, 0, True)
        self.__parameters__ = [self.__position__, self.__central_surface_brightness__,
                               self.__scale_height__, self.__scale_length__,
                               self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '5': 3, '10': 4, 'Z': 5}

    @property
    def central_surface_brightness(self):
        return self.__central_surface_brightness__.value

    @property
    def scale_height(self):
        return self.__scale_height__.value

    @property
    def scale_length(self):
        return self.__scale_length__.value

    @central_surface_brightness.setter
    def central_surface_brightness(self, mu_0: float):
        self.__central_surface_brightness__.value = mu_0

    @scale_height.setter
    def scale_height(self, scale_height: float):
        self.__scale_height__.value = scale_height

    @scale_length.setter
    def scale_length(self, scale_length: float):
        self.__scale_length__.value = scale_length

    def set_central_surface_brightness(self, mu_0: float = None, trainable=True):
        """
        Set initial central surface brightness
        :param mu_0: float, central surface brightness [mag/arcsec^2]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if mu_0 is not None:
            self.__central_surface_brightness__.value = mu_0
        self.__central_surface_brightness__.trainable = trainable

    def set_scale_height(self, scale_height: float = None, trainable=True):
        """
        Set initial disk scale-height
        :param scale_height: float, disk scale-height [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if scale_height is not None:
            self.__scale_height__.value = scale_height
        self.__scale_height__.trainable = trainable

    def set_scale_length(self, scale_length: float = None, trainable=True):
        """
        Set initial disk scale-length
        :param scale_length: float, disk scale-length [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if scale_length is not None:
            self.__scale_length__.value = scale_length
        self.__scale_length__.trainable = trainable


class DeVauc(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "devauc")
        self.__magnitude__ = Parameter(3, 0, True)
        self.__effective_radius__ = Parameter(4, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__magnitude__,
                               self.__effective_radius__, self.__axis_ratio__,
                               self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '9': 3, '10': 4, 'Z': 5}

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @property
    def effective_radius(self):
        return self.__effective_radius__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    @effective_radius.setter
    def effective_radius(self, radius: float):
        self.__effective_radius__.value = radius

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable

    def set_effective_radius(self, radius: float = None, trainable=True):
        """
        Set initial effective radius
        :param radius: float, effective radius [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if radius is not None:
            self.__effective_radius__.value = radius
        self.__effective_radius__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio
        :param ratio: float, axis ratio (b/a)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class King(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "king")
        self.__central_surface_brightness__ = Parameter(3, 0, True)
        self.__core_radius__ = Parameter(4, 0, True)
        self.__tidal_radius__ = Parameter(5, 0, True)
        self.__alpha__ = Parameter(6, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__central_surface_brightness__,
                               self.__core_radius__, self.__tidal_radius__, self.__alpha__,
                               self.__axis_ratio__, self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1, '4': 2, '5': 3,
                                '6': 4, '9': 5, '10': 6, 'Z': 7}

    @property
    def central_surface_brightness(self):
        return self.__central_surface_brightness__.value

    @property
    def core_radius(self):
        return self.__core_radius__.value

    @property
    def tidal_radius(self):
        return self.__tidal_radius__.value

    @property
    def alpha(self):
        return self.__alpha__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @central_surface_brightness.setter
    def central_surface_brightness(self, mu_0: float):
        self.__central_surface_brightness__.value = mu_0

    @core_radius.setter
    def core_radius(self, rc: float):
        self.__core_radius__.value = rc

    @tidal_radius.setter
    def tidal_radius(self, rt: float):
        self.__tidal_radius__.value = rt

    def set_central_surface_brightness(self, mu_0: float = None, trainable=True):
        """
        Set initial central surface brightness
        :param mu_0: float, central surface brightness
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if mu_0 is not None:
            self.__central_surface_brightness__.value = mu_0
        self.__central_surface_brightness__.trainable = trainable

    def set_core_radius(self, rc: float = None, trainable=True):
        """
        Set initial core radius
        :param rc: float, core radius
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if rc is not None:
            self.__core_radius__.value = rc
        self.__core_radius__.trainable = trainable

    def set_tidal_radius(self, rt: float = None, trainable=True):
        """
        Set initial tidal radius
        :param rt: float, tidal radius
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if rt is not None:
            self.__tidal_radius__.value = rt
        self.__tidal_radius__.trainable = trainable

    def set_alpha(self, alpha: float = None, trainable=True):
        """
        Set initial alpha
        :param alpha: float, alpha parameter
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if alpha is not None:
            self.__alpha__.value = alpha
        self.__alpha__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio
        :param ratio: float, axis ratio (b/a)
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class Moffat(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "moffat")
        self.__magnitude__ = Parameter(3, 0, True)
        self.__fwhm__ = Parameter(4, 0, True)
        self.__power_law__ = Parameter(5, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__magnitude__,
                               self.__fwhm__, self.__power_law__, self.__axis_ratio__,
                               self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '5': 3, '9': 4, '10': 5, 'Z': 6}

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @property
    def fwhm(self):
        return self.__fwhm__.value

    @property
    def power_law(self):
        return self.__power_law__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    @fwhm.setter
    def fwhm(self, fwhm: float):
        self.__fwhm__.value = fwhm

    @power_law.setter
    def power_law(self, power_law: float):
        self.__power_law__.value = power_law

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable

    def set_fwhm(self, fwhm: float = None, trainable=True):
        """
        Set initial FWHM (Full Width at Half Maximum)
        :param fwhm: float, FWHM [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if fwhm is not None:
            self.__fwhm__.value = fwhm
        self.__fwhm__.trainable = trainable

    def set_power_law(self, power_law: float = None, trainable=True):
        """
        Set initial power law
        :param power_law: float, power law
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if power_law is not None:
            self.__power_law__.value = power_law
        self.__power_law__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio b/a
        :param ratio: float, axis ratio
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class Gaussian(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "gaussian")
        self.__magnitude__ = Parameter(3, 0, True)
        self.__fwhm__ = Parameter(4, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__magnitude__,
                               self.__fwhm__, self.__axis_ratio__,
                               self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1,
                                '4': 2, '9': 3, '10': 4, 'Z': 5}

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @property
    def fwhm(self):
        return self.__fwhm__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    @fwhm.setter
    def fwhm(self, fwhm: float):
        self.__fwhm__.value = fwhm

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable

    def set_fwhm(self, fwhm: float = None, trainable=True):
        """
        Set initial FWHM (Full Width at Half Maximum)
        :param fwhm: float, FWHM [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if fwhm is not None:
            self.__fwhm__.value = fwhm
        self.__fwhm__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio b/a
        :param ratio: float, axis ratio
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class Ferrer(Anisotropic):
    def __init__(self):
        Anisotropic.__init__(self, "ferrer")
        self.__central_surface_brightness__ = Parameter(3, 0, True)
        self.__outer_truncation_radius__ = Parameter(4, 0, True)
        self.__alpha__ = Parameter(5, 0, True)
        self.__beta__ = Parameter(6, 0, True)
        self.__axis_ratio__ = Parameter(9, 1, True)
        self.__parameters__ = [self.__position__, self.__central_surface_brightness__,
                               self.__outer_truncation_radius__, self.__alpha__, self.__beta__,
                               self.__axis_ratio__, self.__position_angle__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1, '4': 2, '5': 3,
                                '6': 4, '9': 5, '10': 6, 'Z': 7}

    @property
    def central_surface_brightness(self):
        return self.__central_surface_brightness__.value

    @property
    def outer_truncation_radius(self):
        return self.__outer_truncation_radius__.value

    @property
    def alpha(self):
        return self.__alpha__.value

    @property
    def beta(self):
        return self.__beta__.value

    @property
    def axis_ratio(self):
        return self.__axis_ratio__.value

    @central_surface_brightness.setter
    def central_surface_brightness(self, mu_0: float):
        self.__central_surface_brightness__.value = mu_0

    @outer_truncation_radius.setter
    def outer_truncation_radius(self, radius: float):
        self.__outer_truncation_radius__.value = radius

    @alpha.setter
    def alpha(self, alpha: float):
        self.__alpha__.value = alpha

    @beta.setter
    def beta(self, beta: float):
        self.__beta__.value = beta

    @axis_ratio.setter
    def axis_ratio(self, ratio: float):
        self.__axis_ratio__.value = ratio

    def set_central_surface_brightness(self, brightness: float = None, trainable=True):
        """
        Set initial central surface brightness
        :param brightness: float, central surface brightness [mag/arcsec^2]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if brightness is not None:
            self.__central_surface_brightness__.value = brightness
        self.__central_surface_brightness__.trainable = trainable

    def set_outer_truncation_radius(self, radius: float = None, trainable=True):
        """
        Set initial outer truncation radius
        :param radius: float, outer truncation radius [pixels]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if radius is not None:
            self.__outer_truncation_radius__.value = radius
        self.__outer_truncation_radius__.trainable = trainable

    def set_alpha(self, alpha: float = None, trainable=True):
        """
        Set initial alpha (outer truncation sharpness)
        :param alpha: float, outer truncation sharpness
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if alpha is not None:
            self.__alpha__.value = alpha
        self.__alpha__.trainable = trainable

    def set_beta(self, beta: float = None, trainable=True):
        """
        Set initial beta (central slope)
        :param beta: float, central slope
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if beta is not None:
            self.__beta__.value = beta
        self.__beta__.trainable = trainable

    def set_axis_ratio(self, ratio: float = None, trainable=True):
        """
        Set initial axis ratio b/a
        :param ratio: float, axis ratio
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if ratio is not None:
            self.__axis_ratio__.value = ratio
        self.__axis_ratio__.trainable = trainable


class PSF(Component):
    def __init__(self):
        Component.__init__(self, "psf")
        self.__position__ = DoubleParam(1, 0, 0, True, True)
        self.__magnitude__ = Parameter(3, 0, True)
        self.__parameters__ = [self.__position__,
                               self.__magnitude__, self.__output_option__]
        self.__param_index__ = {'1': 0, '3': 1, 'Z': 2}

    @property
    def position(self):
        return self.__position__.value

    @property
    def magnitude(self):
        return self.__magnitude__.value

    @position.setter
    def position(self, position: tuple):
        self.__position__.value = position

    @magnitude.setter
    def magnitude(self, magnitude: float):
        self.__magnitude__.value = magnitude

    def set_position(self, x: float = None, y: float = None, trainable_x=True, trainable_y=True):
        """
        Set initial position
        :param x: float, x position [pixels]
        :param y: float, y position [pixels]
        :param trainable_x: bool, indicates whether the param x are trainable
        :param trainable_y: bool, indicates whether the param y are trainable
        """
        if x is not None and y is not None:
            self.__position__.value = (x, y)
        elif x is not None:
            self.__position__.value = (x, self.position[1])
        elif y is not None:
            self.__position__.value = (self.position[0], y)
        self.__position__.trainable = (trainable_x, trainable_y)

    def set_magnitude(self, magnitude: float = None, trainable=True):
        """
        Set initial total magnitude
        :param magnitude: float, total magnitude
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if magnitude is not None:
            self.__magnitude__.value = magnitude
        self.__magnitude__.trainable = trainable


class Sky(Component):
    state_num = 2

    def __init__(self):
        Component.__init__(self, "sky")
        self.__background__ = Parameter(1, 0, True)
        self.__gradient_x__ = Parameter(2, 0, False)
        self.__gradient_y__ = Parameter(3, 0, False)
        self.__parameters__ = [self.__background__, self.__gradient_x__,
                               self.__gradient_y__, self.__output_option__]
        self.__param_index__ = {'1': 0, '2': 1, '3': 2, 'Z': 3}

    @property
    def background(self):
        return self.__background__.value

    @property
    def gradient_x(self):
        return self.__gradient_x__.value

    @property
    def gradient_y(self):
        return self.__gradient_y__.value

    @background.setter
    def background(self, sky_background: float):
        self.__background__.value = sky_background

    @gradient_x.setter
    def gradient_x(self, sky_gradient_x: float):
        self.__gradient_x__.value = sky_gradient_x

    @gradient_y.setter
    def gradient_y(self, sky_gradient_y: float):
        self.__gradient_y__.value = sky_gradient_y

    @property
    def state(self):
        if self.__background__.trainable:
            return 1
        else:
            return 0

    @state.setter
    def state(self, state: int):
        if state == 0:
            self.__background__.trainable = False
        else:
            self.__background__.trainable = True

    def set_background(self, sky_background: float = None, trainable=True):
        """
        Set initial sky background
        :param sky_background: float, sky background [ADU counts]
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if sky_background is not None:
            self.__background__.value = sky_background
        self.__background__.trainable = trainable

    def set_gradient_x(self, sky_gradient_x: float = None, trainable=True):
        """
        Set initial sky gradient in x direction
        :param sky_gradient_x: float, sky gradient in x direction
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if sky_gradient_x is not None:
            self.__gradient_x__.value = sky_gradient_x
        self.__gradient_x__.trainable = trainable

    def set_gradient_y(self, sky_gradient_y: float = None, trainable=True):
        """
        Set initial sky gradient in y direction
        :param sky_gradient_y: float, sky gradient in y direction
        :param trainable: bool, indicates whether the parameter is trainable
        """
        if sky_gradient_y is not None:
            self.__gradient_y__.value = sky_gradient_y
        self.__gradient_y__.trainable = trainable


component_names = {'sersic': Sersic, 'nuker': Nuker, 'expdisk': ExpDisk, 'edgedisk': EdgeDisk,
                   'devauc': DeVauc, 'king': King, 'moffat': Moffat, 'gaussian': Gaussian, 'ferrer': Ferrer, 'psf': PSF, 'sky': Sky}
