package com.spbsu.ml;

import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.LARSMethod;
import com.spbsu.ml.methods.MLMethod;


/**
 * User: solar
 * Date: 21.12.2010
 * Time: 21:23:08
 */
public class LearnByFeatures {
    public static void main(String[] args) throws Exception {
        final DataSet learn = DataTools.loadFromFeaturesTxt("/Users/solar/experiments-local/matrixnet/gulin-reference/features.txt");
        final DataSet test = DataTools.loadFromFeaturesTxt("/Users/solar/experiments-local/matrixnet/gulin-reference/featuresTest.txt");

        final Oracle1 loss = new L2(learn.target());
        final Oracle1 testLoss = new L2(test.target());
        MLMethod method = new LARSMethod();
//        MLMethod method = new LASSOMethod(1000, 0.001);
//        Boosting method = new Boosting(new BestAtomicSplitMethod(), 20000, 0.002);
//        method.addProgressHandler(new ProgressHandler() {
//            int index = 0;
//            public void invoke(Model partial) {
//                if (index % 1000 == 0)
//                    System.out.println(index++
//                                      +" learn: " + loss.value(partial, learn)
//                                      +" test: " + loss.value(partial, test));
//                index++;
//            }
//        });
        Model model = method.fit(learn, loss);
        System.out.println("learn: " + loss.value(model.value(learn)) + " test: " + loss.value(model.value(test)));
        System.out.println("model:\n" + model.toString());
    }
}
