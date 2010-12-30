package ml.data;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:22
 */
public interface DataSet {
    int power();
    int featureCount();

    DSIterator iterator();
    DSIterator orderBy(int featureIndex);

    double statistic(Class<? extends StatisticCalculator> type);
}
