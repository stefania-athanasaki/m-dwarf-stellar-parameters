"""Stellar-parameter relations used in the 2026 M-dwarf project."""
import numpy as np

MASS_COEFF = (0.5858, 0.3872, -0.1217, 0.0106, -2.7262e-4)
RADIUS_COEFF = (1.9515, -0.3520, 0.01680)
TEFF_COEFF = (9649.1817, -7175.80969, 3642.30312, -1020.37499, 146.20008, -8.30455)
LOGG_SUN = 4.438068

def distance_pc(parallax_mas):
    parallax_mas=np.asarray(parallax_mas,dtype=float)
    return 1000.0/parallax_mas

def distance_error_pc(parallax_mas, parallax_error_mas):
    p=np.asarray(parallax_mas,dtype=float); s=np.asarray(parallax_error_mas,dtype=float)
    return 1000.0*s/p**2

def absolute_k(kmag, parallax_mas):
    p_arcsec=np.asarray(parallax_mas,dtype=float)/1000.0
    return np.asarray(kmag,dtype=float)+5*np.log10(p_arcsec)+5

def absolute_k_error(kmag_error, parallax_mas, parallax_error_mas):
    sk=np.asarray(kmag_error,dtype=float); p=np.asarray(parallax_mas,dtype=float); sp=np.asarray(parallax_error_mas,dtype=float)
    return np.sqrt(sk**2 + ((5/np.log(10))*(sp/p))**2)

def mass_from_mk(mk):
    a,b,c,d,e=MASS_COEFF; x=np.asarray(mk,dtype=float)
    return a+b*x+c*x**2+d*x**3+e*x**4

def mass_error(mk, mk_error):
    _,b,c,d,e=MASS_COEFF; x=np.asarray(mk,dtype=float)
    deriv=b+2*c*x+3*d*x**2+4*e*x**3
    return np.abs(deriv)*np.asarray(mk_error,dtype=float)

def radius_from_mk(mk):
    a,b,c=RADIUS_COEFF; x=np.asarray(mk,dtype=float)
    return a+b*x+c*x**2

def radius_error(mk, mk_error):
    _,b,c=RADIUS_COEFF; x=np.asarray(mk,dtype=float)
    return np.abs(b+2*c*x)*np.asarray(mk_error,dtype=float)

def teff_from_bp_rp(bp_rp):
    c0,c1,c2,c3,c4,c5=TEFF_COEFF; x=np.asarray(bp_rp,dtype=float)
    return c0+c1*x+c2*x**2+c3*x**3+c4*x**4+c5*x**5

def magnitude_error_from_flux(flux, flux_error):
    f=np.asarray(flux,dtype=float); sf=np.asarray(flux_error,dtype=float)
    return (2.5/np.log(10))*(sf/f)

def bp_rp_error(bp_flux,bp_flux_error,rp_flux,rp_flux_error):
    sbp=magnitude_error_from_flux(bp_flux,bp_flux_error); srp=magnitude_error_from_flux(rp_flux,rp_flux_error)
    return np.sqrt(sbp**2+srp**2)

def teff_error(bp_rp, bp_rp_sigma):
    _,c1,c2,c3,c4,c5=TEFF_COEFF; x=np.asarray(bp_rp,dtype=float)
    deriv=c1+2*c2*x+3*c3*x**2+4*c4*x**3+5*c5*x**4
    return np.abs(deriv)*np.asarray(bp_rp_sigma,dtype=float)

def logg(mass, radius):
    m=np.asarray(mass,dtype=float); r=np.asarray(radius,dtype=float)
    return LOGG_SUN+np.log10(m)-2*np.log10(r)

def logg_error(mass,mass_sigma,radius,radius_sigma):
    m=np.asarray(mass,dtype=float); sm=np.asarray(mass_sigma,dtype=float); r=np.asarray(radius,dtype=float); sr=np.asarray(radius_sigma,dtype=float)
    return np.sqrt((sm/(m*np.log(10)))**2 + (2*sr/(r*np.log(10)))**2)
