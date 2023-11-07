from enum import Enum


class DatasetColumns(Enum):
    RIGHT_ASCENSION_J2000 = "ra"
    DECLINATION_J2000 = "dec"

    MAGNITUDE_FIT_U = "u"
    MAGNITUDE_FIT_G = "g"
    MAGNITUDE_FIT_R = "r"
    MAGNITUDE_FIT_I = "i"
    MAGNITUDE_FIT_Z = "z"

    PETROSIAN_RADIUS_U = "petroRad_u"
    PETROSIAN_RADIUS_G = "petroRad_g"
    PETROSIAN_RADIUS_I = "petroRad_i"
    PETROSIAN_RADIUS_R = "petroRad_r"
    PETROSIAN_RADIUS_Z = "petroRad_z"

    PETROSIAN_FLUX_U = "petroFlux_u"
    PETROSIAN_FLUX_G = "petroFlux_g"
    PETROSIAN_FLUX_I = "petroFlux_i"
    PETROSIAN_FLUX_R = "petroFlux_r"
    PETROSIAN_FLUX_Z = "petroFlux_z"

    PETROSIAN_HALF_LIGHT_RADIUS_U = "petroR50_u"
    PETROSIAN_HALF_LIGHT_RADIUS_G = "petroR50_g"
    PETROSIAN_HALF_LIGHT_RADIUS_I = "petroR50_i"
    PETROSIAN_HALF_LIGHT_RADIUS_R = "petroR50_r"
    PETROSIAN_HALF_LIGHT_RADIUS_Z = "petroR50_z"

    PSF_MAGNITUDE_U = "psfMag_u"
    PSF_MAGNITUDE_R = "psfMag_r"
    PSF_MAGNITUDE_G = "psfMag_g"
    PSF_MAGNITUDE_I = "psfMag_i"
    PSF_MAGNITUDE_Z = "psfMag_z"

    EXPONENTIAL_FIT_U = "expAB_u"
    EXPONENTIAL_FIT_G = "expAB_g"
    EXPONENTIAL_FIT_R = "expAB_r"
    EXPONENTIAL_FIT_I = "expAB_i"
    EXPONENTIAL_FIT_Z = "expAB_z"

    REDSHIFT = "redshift"
    CLASS = "class"
