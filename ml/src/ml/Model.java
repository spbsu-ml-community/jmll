package ml;

import ml.data.DataEntry;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Model {
    double value(DataEntry point);
    double learnScore();
}
