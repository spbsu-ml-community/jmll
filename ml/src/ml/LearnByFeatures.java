package ml;

import ml.data.DataSet;
import ml.data.DataTools;
import ml.loss.L2Loss;
import ml.loss.LossFunction;
import ml.methods.LARSMethod;
import ml.methods.MLMethod;


/**
 * User: solar
 * Date: 21.12.2010
 * Time: 21:23:08
 */
public class LearnByFeatures {
    public static void main(String[] args) throws Exception {
        final DataSet learn = DataTools.loadFromFeaturesTxt("/Users/solar/experiments-local/matrixnet/gulin-reference/features.txt");
        final DataSet test = DataTools.loadFromFeaturesTxt("/Users/solar/experiments-local/matrixnet/gulin-reference/featuresTest.txt");

        final LossFunction loss = new L2Loss();
        MLMethod method = new LARSMethod();
//        MLMethod method = new LASSOMethod(1000, 0.001);
//        Boosting method = new Boosting(new BestAtomicSplitMethod(), 20000, 0.002);
//        method.setProgressHandler(new ProgressHandler() {
//            int index = 0;
//            public void progress(Model partial) {
//                if (index % 1000 == 0)
//                    System.out.println(index++
//                                      +" learn: " + loss.value(partial, learn)
//                                      +" test: " + loss.value(partial, test));
//                index++;
//            }
//        });
        Model model = method.fit(learn, loss);
        System.out.println("learn: " + loss.value(model, learn) + " test: " + loss.value(model, test));
        System.out.println("model:\n" + model.toString());
    }
}
