package ml.data;

import ml.data.impl.Bootstrap;
import ml.data.impl.ChangedTarget;
import ml.data.impl.DataSetImpl;
import ml.data.impl.NormalizedDataSet;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.*;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:05
 */
public class DataTools {
    public static DataSet loadFromFeaturesTxt(String file) throws IOException {
        final LineNumberReader reader = new LineNumberReader(new FileReader(file));
        List<double[]> set = new LinkedList<double[]>();
        List<Double> targets = new LinkedList<Double>();
        int maxFeatures = 0;
        String line;
        final List<Double> featuresA = new ArrayList<Double>();
        while ((line = reader.readLine()) != null) {
            featuresA.clear();
            StringTokenizer tok = new StringTokenizer(line, "\t");
            tok.nextToken();
            targets.add(Double.parseDouble(tok.nextToken()));
            tok.nextToken();
            tok.nextToken();
            while (tok.hasMoreTokens()) {
                featuresA.add(Double.parseDouble(tok.nextToken()));
            }
            maxFeatures = Math.max(maxFeatures, featuresA.size());
            double[] features = new double[maxFeatures];
            for (int i = 0; i < featuresA.size(); i++) {
                features[i] = featuresA.get(i);
            }
            set.add(features);
        }
        double[] data = new double[maxFeatures * set.size()];
        double[] target = new double[set.size()];
        Iterator<double[]> iterF = set.iterator();
        Iterator<Double> iterT = targets.iterator();
        int featuresCount = maxFeatures;
        int index = 0;
        while (iterF.hasNext()) {
            final double[] features = iterF.next();
            System.arraycopy(features, 0, data, index * featuresCount, features.length);
            target[index] = iterT.next();
            index++;
        }
        return new DataSetImpl(data, target);
    }

    public static DataSet changeTarget(DataSet base, double[] newTarget) {
        return new ChangedTarget((DataSetImpl)base, newTarget);
    }

    public static DataSet normalize(DataSet learn) {
        return new NormalizedDataSet((DataSetImpl)learn);
    }

    public static DataSet bootstrap(DataSet base) {
        return new Bootstrap((DataSetImpl)base);
    }
}
