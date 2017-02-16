package com.spbsu.ml.methods.cart;

/**
 * Created by n_buga on 17.10.16.
 */
public class Leaf {
    private static double Sigma2;

    private static int idCounter = 0;

    private int idLeaf = idCounter++;
    private int leafNumber;
    private ListFeatures listFeatures = new ListFeatures(idLeaf);
    private double mean;
    private double error;

    private double sqrSum = 0;
    private double sum = 0;
    private int count = 0;

    public Leaf(int number) {
        leafNumber = number;
        error = -1;
    }

    public Leaf(double error, int number) {
        leafNumber = number;
        this.error = error;
    }

    public Leaf(Leaf l, int number) {
        leafNumber = number;
        listFeatures.addAllFeatures(l.listFeatures);
        mean = l.mean;
        error = l.error;
    }

    public static void setSigma2(double sigma2) {
        Sigma2 = sigma2;
    }

    public double getError() {
        return error;
    }

    public void setError(double err) {
        error = err;
    }

    public int getID() {
        return  idLeaf;
    }

    public ListFeatures getListFeatures() {
        return listFeatures;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public double getMean() {
        return mean;
    }

    public void addNewItem(double d) {
        sqrSum += d*d;
        sum += d;
        count++;
    }

    public double getSqrSum() {
        return sqrSum;
    }

    public double getSum() {
        return sum;
    }

    public int getCount() {
        return count;
    }

    public void clearStatistic() {
        sum = 0;
        sqrSum = 0;
        count = 0;
    }

    public void calcError() {
        if (count == 1) {
            error = 0;
        } else {
            error = (sqrSum - sum * sum / count)/(1);
        }
    }

    public void calcMean() {
        mean = sum/(count);
    }

    public double getValue() {
        return sum/(count + 1);
/*        if (count <= 2 || sqrSum < 1e-6)
            return sum/(count + 1);
        return (1 - (count - 2)*Sigma2/sqrSum)*mean; */
    }

    public int getLeafNumber() {
        return leafNumber;
    }
}
