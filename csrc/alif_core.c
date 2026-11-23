/* Optional compiled helpers for Alif Ephemeris. Everything here has a plain Python twin in swe_compat.py that gives the
   same numbers; this file only makes the hot paths quicker. Written from the published formulas, not from any other
   ephemeris program's code. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define TAU 6.283185307179586476925287
#define PI 3.141592653589793238462643

static double wrap(double x) { double r = fmod(x, TAU); return r < 0.0 ? r + TAU : r; }

/* placidus(ramc_deg, eps_deg, lat_deg) -> (cusps[12], asc, mc, vertex, equasc) in degrees, or None when the method
   fails (inside the polar circles). */
static PyObject *placidus(PyObject *self, PyObject *args) {
    double ramc, eps, lat;
    if (!PyArg_ParseTuple(args, "ddd", &ramc, &eps, &lat)) return NULL;
    double th = ramc * PI / 180.0, e = eps * PI / 180.0, ph = lat * PI / 180.0;
    double ste = sin(e), cte = cos(e), sth = sin(th), cth = cos(th);
    double tan_phi = tan(ph);
    double mc = wrap(atan2(sth, cth * cte));
    double asc = wrap(atan2(cth, -(sth * cte + tan_phi * ste)));
    if (wrap(asc - mc) >= PI) asc = wrap(asc + PI);
    double th2 = th + PI;
    double cot_phi = tan_phi != 0.0 ? 1.0 / tan_phi : 1e16;
    double vtx = wrap(atan2(cos(th2), -(sin(th2) * cte + cot_phi * ste)));
    double mc2 = wrap(atan2(sin(th2), cos(th2) * cte));
    if (wrap(vtx - mc2) >= PI) vtx = wrap(vtx + PI);
    double equasc = wrap(atan2(cth, -sth * cte));
    if (wrap(equasc - mc) >= PI) equasc = wrap(equasc + PI);
    double dm = wrap(asc - mc), da = PI - dm, third = 1.0 / 3.0;
    int hs[4] = {11, 12, 2, 3};
    double fr[4] = {third, 2.0 * third, 2.0 * third, third};
    int noon[4] = {1, 1, 0, 0};
    double x0[4] = {mc + dm * third, mc + 2.0 * dm * third, asc + da * third, asc + 2.0 * da * third};
    double cus[13];
    for (int q = 0; q < 4; q++) {
        double x = x0[q], frac = fr[q];
        double base = noon[q] ? th : th + PI;
        double off = noon[q] ? 0.0 : PI;
        for (int it = 0; it < 40; it++) {
            double sd = ste * sin(x);
            double t = -tan_phi * sd / sqrt(1.0 - sd * sd);
            if (t >= 1.0 || t <= -1.0) Py_RETURN_NONE;
            double ra = base + frac * (acos(t) - off);
            double x1 = wrap(atan2(sin(ra), cos(ra) * cte));
            double d1 = fmod(x1 - x + PI + 4.0 * TAU, TAU) - PI;
            if (d1 > -1e-9 && d1 < 1e-9) { x = x1; break; }
            sd = ste * sin(x1);
            t = -tan_phi * sd / sqrt(1.0 - sd * sd);
            if (t >= 1.0 || t <= -1.0) Py_RETURN_NONE;
            ra = base + frac * (acos(t) - off);
            double x2 = wrap(atan2(sin(ra), cos(ra) * cte));
            double d2 = fmod(x2 - x1 + PI + 4.0 * TAU, TAU) - PI;
            double den = d2 - d1;
            x = fabs(den) > 1e-15 ? wrap(x - d1 * d1 / den) : x2;
        }
        cus[hs[q]] = x;
    }
    double c[12];
    c[0] = asc; c[9] = mc; c[3] = wrap(mc + PI); c[6] = wrap(asc + PI);
    int opp[4] = {5, 6, 8, 9};
    for (int q = 0; q < 4; q++) { c[hs[q] - 1] = cus[hs[q]]; c[opp[q] - 1] = wrap(cus[hs[q]] + PI); }
    PyObject *list = PyList_New(12);
    if (!list) return NULL;
    for (int i = 0; i < 12; i++) PyList_SET_ITEM(list, i, PyFloat_FromDouble(c[i] * 180.0 / PI));
    PyObject *out = Py_BuildValue("(Odddd)", list, asc * 180.0 / PI, mc * 180.0 / PI, vtx * 180.0 / PI, equasc * 180.0 / PI);
    Py_DECREF(list);
    return out;
}

static PyMethodDef methods[] = {
    {"placidus", placidus, METH_VARARGS, "Placidus cusps and angles for one place and time."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_core", "Compiled helpers for Alif Ephemeris", -1, methods};

PyMODINIT_FUNC PyInit__core(void) { return PyModule_Create(&module); }
