package com.spbsu.ml;

/**
 * Created by noxoomo on 15/07/14.
 */

import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * User: Vasily
 * Date: 25.04.14
 * Time: 23:03
 */


public class
        Utils {
    static Random random = new Random();


    static public double[] rank(double[] values) {
        int[] order = argsort(values);
        double rk = 1.0;
        double[] result = new double[values.length];
        double prev = values[order[0]];
        for (int ind : order) {
            if (values[ind] != prev) {
                ++rk;
                prev = values[ind];
            }
            result[ind] = rk;
        }
        return result;
    }


    public static String mkString(int[] arr) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < arr.length - 1; ++i) {
            builder.append(arr[i]);
            builder.append(" ");
        }
        builder.append(arr[arr.length - 1]);
        return builder.toString();
    }

    public static <T> String mkString(ArrayList<T> list) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < list.size() - 1; ++i) {
            builder.append(list.get(i));
            builder.append(" ");
        }
        builder.append(list.get(list.size() - 1));
        return builder.toString();
    }

    public static String countsString(int length) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length - 1; ++i) {
            builder.append(i);
            builder.append(" ");
        }
        builder.append(length - 1);
        return builder.toString();
    }


    public static int[] argsort(final double[] a) {
        Integer[] index = new Integer[a.length];
        for (int i = 0; i < index.length; i++) {
            index[i] = i;
        }
        Arrays.sort(index, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return Double.compare(a[i1], a[i2]);
            }
        });
        int[] order = new int[index.length];
        for (int i = 0; i < order.length; ++i)
            order[i] = index[i];
        return order;
    }


    //return argsort of i'th column
//    public static int[] argsort(double[] values) {
//        int[] order = new int[values.length];
//        for (int i = 0; i < values.length; ++i)
//            order[i] = i;
//        qsort(values, order, 0, order.length - 1);
//        return order;
//    }


    private static void qsort(double[] values, int[] order, int left, int right) {
        int l = left;
        int r = right;
        if (r - l <= 0) {
            return;
        }
        if (r - l < 8) {
            for (int j = l; j <= r; ++j) {
                for (int k = j + 1; k <= r; ++k) {
                    if (values[order[j]] > values[order[k]])
                        swap(j, k, order);
                }
            }
            return;
        }

        int mid = l + random.nextInt(r - l);
        double pivot = values[order[mid]];

        while (l < r) {
            while (values[order[l]] < pivot) {
                l++;
            }
            while (values[order[r]] > pivot) {
                r--;
            }
            if (l <= r) {
                swap(l, r, order);
                l++;
                r--;
            }
        }
        if (r > left) {
            qsort(values, order, left, r);
        }
        if (l < right) {
            qsort(values, order, l, right);
        }
    }

    private static void swap(int l, int r, int[] order) {
        int tmp = order[l];
        order[l] = order[r];
        order[r] = tmp;
    }


    public static double kappa(int[] real, int[] predict) {
        double[][] counts = new double[3][3];
        for (int i = 0; i < real.length; ++i) {
            counts[real[i]][predict[i]]++;
        }
        double total = 0;
        for (int i = 0; i < counts.length; ++i)
            for (int j = 0; j < counts[i].length; ++j)
                total += counts[i][j];

        double pra = counts[0][0] + counts[1][1] + counts[2][2];
        pra /= total;
        double[] rowSum = new double[3];
        double[] colSum = new double[3];
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i].length; ++j) {
                colSum[i] += counts[i][j];
                rowSum[j] += counts[i][j];
            }
        }

        double pre = 0;
        for (int i = 0; i < colSum.length; ++i) {
            pre += colSum[i] * rowSum[i] / (total * total);
        }

        return (pra - pre) / (1 - pre);
    }

    public static int[] sample(int n) {
        int[] result = new int[n];
        for (int i = 0; i < result.length; ++i) {
            result[i] = i;
        }
        shuffle(result);
        return result;
    }

    public static void shuffle(int[] index) {
        for (int i = index.length - 1; i > 0; --i) {
            int j = random.nextInt(i + 1);
            swap(i, j, index);
        }
    }

    public static void swap(int i, int j, double[] target) {
        double tmp = target[i];
        target[i] = target[j];
        target[j] = tmp;
    }


    public static double mean(TDoubleArrayList sample) {
        double mean = 0;
        for (int i = 0; i < sample.size(); ++i)
            mean += sample.get(i);
        return mean /= sample.size();
    }

    public static double var(TDoubleArrayList sample) {
        double secondMoment = 0;
        for (int i = 0; i < sample.size(); ++i) {
            double d = sample.get(i);
            secondMoment += d * d;
        }
        secondMoment /= sample.size();
        double m = mean(sample);
        return secondMoment - m * m;
    }

    public static double[] stats(TDoubleArrayList sample) {
        if (sample.size() == 0) {
            return new double[4];
        }
        double min;
        double max;
        min = max = sample.get(0);
        double mean = 0;
        double result[] = new double[4];
        double secondMoment = 0;
        for (int i = 0; i < sample.size(); ++i) {
            double d = sample.get(i);
            secondMoment += d * d;
            mean += d;
            if (d > max) max = d;
            if (d < min) min = d;
        }
        mean /= sample.size();
        secondMoment /= sample.size();
        result[0] = mean;
        result[1] = secondMoment - mean * mean;
        result[1] = Math.sqrt(result[1]);
        result[2] = min;
        result[3] = max;
        return result;
    }
}
