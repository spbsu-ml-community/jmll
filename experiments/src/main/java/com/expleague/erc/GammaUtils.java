package com.expleague.erc;

import static java.lang.Math.*;

public class GammaUtils {
    private static final double LANCZOS_G = 607.0 / 128.0;
    private static final double HALF_LOG_2_PI = 0.5 * log(2.0 * PI);
    private static final double[] LANCZOS = {
            0.99999999999999709182,
            57.156235665862923517,
            -59.597960355475491248,
            14.136097974741747174,
            -0.49191381609762019978,
            .33994649984811888699e-4,
            .46523628927048575665e-4,
            -.98374475304879564677e-4,
            .15808870322491248884e-3,
            -.21026444172410488319e-3,
            .21743961811521264320e-3,
            -.16431810653676389022e-3,
            .84418223983852743293e-4,
            -.26190838401581408670e-4,
            .36899182659531622704e-5,
    };
    private static final double INV_GAMMA1P_M1_A0 = .611609510448141581788E-08;
    private static final double INV_GAMMA1P_M1_A1 = .624730830116465516210E-08;
    private static final double INV_GAMMA1P_M1_B1 = .203610414066806987300E+00;
    private static final double INV_GAMMA1P_M1_B2 = .266205348428949217746E-01;
    private static final double INV_GAMMA1P_M1_B3 = .493944979382446875238E-03;
    private static final double INV_GAMMA1P_M1_B4 = -.851419432440314906588E-05;
    private static final double INV_GAMMA1P_M1_B5 = -.643045481779353022248E-05;
    private static final double INV_GAMMA1P_M1_B6 = .992641840672773722196E-06;
    private static final double INV_GAMMA1P_M1_B7 = -.607761895722825260739E-07;
    private static final double INV_GAMMA1P_M1_B8 = .195755836614639731882E-09;
    private static final double INV_GAMMA1P_M1_P0 = .6116095104481415817861E-08;
    private static final double INV_GAMMA1P_M1_P1 = .6871674113067198736152E-08;
    private static final double INV_GAMMA1P_M1_P2 = .6820161668496170657918E-09;
    private static final double INV_GAMMA1P_M1_P3 = .4686843322948848031080E-10;
    private static final double INV_GAMMA1P_M1_P4 = .1572833027710446286995E-11;
    private static final double INV_GAMMA1P_M1_P5 = -.1249441572276366213222E-12;
    private static final double INV_GAMMA1P_M1_P6 = .4343529937408594255178E-14;
    private static final double INV_GAMMA1P_M1_Q1 = .3056961078365221025009E+00;
    private static final double INV_GAMMA1P_M1_Q2 = .5464213086042296536016E-01;
    private static final double INV_GAMMA1P_M1_Q3 = .4956830093825887312020E-02;
    private static final double INV_GAMMA1P_M1_Q4 = .2692369466186361192876E-03;
    private static final double INV_GAMMA1P_M1_C = -.422784335098467139393487909917598E+00;
    private static final double INV_GAMMA1P_M1_C0 = .577215664901532860606512090082402E+00;
    private static final double INV_GAMMA1P_M1_C1 = -.655878071520253881077019515145390E+00;
    private static final double INV_GAMMA1P_M1_C2 = -.420026350340952355290039348754298E-01;
    private static final double INV_GAMMA1P_M1_C3 = .166538611382291489501700795102105E+00;
    private static final double INV_GAMMA1P_M1_C4 = -.421977345555443367482083012891874E-01;
    private static final double INV_GAMMA1P_M1_C5 = -.962197152787697356211492167234820E-02;
    private static final double INV_GAMMA1P_M1_C6 = .721894324666309954239501034044657E-02;
    private static final double INV_GAMMA1P_M1_C7 = -.116516759185906511211397108401839E-02;
    private static final double INV_GAMMA1P_M1_C8 = -.215241674114950972815729963053648E-03;
    private static final double INV_GAMMA1P_M1_C9 = .128050282388116186153198626328164E-03;
    private static final double INV_GAMMA1P_M1_C10 = -.201348547807882386556893914210218E-04;
    private static final double INV_GAMMA1P_M1_C11 = -.125049348214267065734535947383309E-05;
    private static final double INV_GAMMA1P_M1_C12 = .113302723198169588237412962033074E-05;
    private static final double INV_GAMMA1P_M1_C13 = -.205633841697760710345015413002057E-06;
    private static final double EPS = 1e-14;

    public static double lowerGamma(int k, double x) {
        final double reg = regularizedGammaP(k, x);
        long fact = 1;
        for (long i = 2; i < k; ++i) {
            fact *= i;
        }
        return reg * fact;
    }

    public static double regularizedGammaP(double a, double x) {
        return regularizedGammaP(a, x, EPS, Integer.MAX_VALUE);
    }

    private static double regularizedGammaP(double a, double x, double epsilon, int maxIterations) {
        double ret;
        if (Double.isNaN(a) || Double.isNaN(x) || (a <= 0.0) || (x < 0.0)) {
            ret = Double.NaN;
        } else if (x == 0.0) {
            ret = 0.0;
        } else {
            // calculate series
            double n = 0.0; // current element index
            double an = 1.0 / a; // n-th element in the series
            double sum = an; // partial sum
            while (abs(an/sum) > epsilon &&
                    n < maxIterations &&
                    sum < Double.POSITIVE_INFINITY) {
                // compute next element in the series
                n += 1.0;
                an *= x / (a + n);

                // update partial sum
                sum += an;
            }
            if (n >= maxIterations) {
                throw new Error("Too many iterations");
            } else if (Double.isInfinite(sum)) {
                ret = 1.0;
            } else {
                ret = exp(-x + (a * log(x)) - logGamma(a)) * sum;
            }
        }
        return ret;
    }

    private static double logGamma(double x) {
        double ret;
        if (Double.isNaN(x) || (x <= 0.0)) {
            ret = Double.NaN;
        } else if (x < 0.5) {
            return logGamma1p(x) - log(x);
        } else if (x <= 2.5) {
            return logGamma1p((x - 0.5) - 0.5);
        } else if (x <= 8.0) {
            final int n = (int) floor(x - 1.5);
            double prod = 1.0;
            for (int i = 1; i <= n; i++) {
                prod *= x - i;
            }
            return logGamma1p(x - (n + 1)) + log(prod);
        } else {
            double sum = lanczos(x);
            double tmp = x + LANCZOS_G + .5;
            ret = ((x + .5) * log(tmp)) - tmp +
                    HALF_LOG_2_PI + log(sum / x);
        }
        return ret;
    }

    private static double logGamma1p(final double x) {
        if (x < -0.5 || x > 1.5) {
            throw new Error(String.format("Number %f is out of [-0.5, 1.5] range", x));
        }
        return -log1p(invGamma1pm1(x));
    }

    private static double lanczos(final double x) {
        double sum = 0.0;
        for (int i = LANCZOS.length - 1; i > 0; --i) {
            sum += LANCZOS[i] / (x + i);
        }
        return sum + LANCZOS[0];
    }

    private static double invGamma1pm1(final double x) {
        if (x < -0.5 || x > 1.5) {
            throw new Error(String.format("Number %f is out of [-0.5, 1.5] range", x));
        }
        final double ret;
        final double t = x <= 0.5 ? x : (x - 0.5) - 0.5;
        if (t < 0.0) {
            final double a = INV_GAMMA1P_M1_A0 + t * INV_GAMMA1P_M1_A1;
            double b = INV_GAMMA1P_M1_B8;
            b = seqTransform(new double[]{
                    INV_GAMMA1P_M1_B7, INV_GAMMA1P_M1_B6, INV_GAMMA1P_M1_B5, INV_GAMMA1P_M1_B4,
                    INV_GAMMA1P_M1_B3, INV_GAMMA1P_M1_B2, INV_GAMMA1P_M1_B1}, b, t);
            b = 1.0 + t * b;

            double c = INV_GAMMA1P_M1_C13 + t * (a / b);
            c = seqTransform(new double[]{
                    INV_GAMMA1P_M1_C12, INV_GAMMA1P_M1_C11, INV_GAMMA1P_M1_C10, INV_GAMMA1P_M1_C9,
                    INV_GAMMA1P_M1_C8, INV_GAMMA1P_M1_C7, INV_GAMMA1P_M1_C6, INV_GAMMA1P_M1_C5,
                    INV_GAMMA1P_M1_C4, INV_GAMMA1P_M1_C3, INV_GAMMA1P_M1_C2, INV_GAMMA1P_M1_C1, INV_GAMMA1P_M1_C}, c, t);
            if (x > 0.5) {
                ret = t * c / x;
            } else {
                ret = x * ((c + 0.5) + 0.5);
            }
        } else {
            double p = INV_GAMMA1P_M1_P6;
            p = seqTransform(new double[]{
                    INV_GAMMA1P_M1_P5, INV_GAMMA1P_M1_P4, INV_GAMMA1P_M1_P3, INV_GAMMA1P_M1_P2,
                    INV_GAMMA1P_M1_P1, INV_GAMMA1P_M1_P0}, p, t);

            double q = INV_GAMMA1P_M1_Q4;
            q = seqTransform(new double[]{INV_GAMMA1P_M1_Q3, INV_GAMMA1P_M1_Q2, INV_GAMMA1P_M1_Q1}, q, t);
            q = 1.0 + t * q;

            double c = INV_GAMMA1P_M1_C13 + (p / q) * t;
            c = seqTransform(new double[]{INV_GAMMA1P_M1_C12, INV_GAMMA1P_M1_C11, INV_GAMMA1P_M1_C10,
                    INV_GAMMA1P_M1_C9, INV_GAMMA1P_M1_C8, INV_GAMMA1P_M1_C7, INV_GAMMA1P_M1_C6, INV_GAMMA1P_M1_C5,
                    INV_GAMMA1P_M1_C4, INV_GAMMA1P_M1_C3, INV_GAMMA1P_M1_C2, INV_GAMMA1P_M1_C1, INV_GAMMA1P_M1_C0}, c, t);

            if (x > 0.5) {
                ret = (t / x) * ((c - 0.5) - 0.5);
            } else {
                ret = x * c;
            }
        }
        return ret;
    }

    private static double seqTransform(double[] coeffs, double init, double t) {
        for (double coeff : coeffs) {
            init = coeff + t * init;
        }
        return init;
    }
}
