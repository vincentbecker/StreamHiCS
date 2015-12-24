package statisticaltests;

import streamdatastructures.DataBundle;

/**
 * Implements a {@link StatisticalTest} using the Welch-t-test.
 * 
 * @author Vincent
 *
 */
public class WelchT extends StatisticalTest {

	/**
	 * The test instance. 
	 */
	private TTest tTest;

	/**
	 * Creates an instance of this class. 
	 */
	public WelchT() {
		tTest = new TTest();
	}

	@Override
	public double calculateDeviation(double[] sample1, double[] sample2) {
		double p = tTest.tTest(sample1, sample2);
		return 1 - p;
	}

	@Override
	public double calculateWeightedDeviation(DataBundle dataBundle1, DataBundle dataBundle2) {
		StatisticsBundle statistics1 = weightedStatistics(dataBundle1.getData(), dataBundle1.getWeights());
		StatisticsBundle statistics2 = weightedStatistics(dataBundle2.getData(), dataBundle2.getWeights());

		double m1 = statistics1.getMean();
		double v1 = statistics1.getVariance();
		double n1 = statistics1.getWeight();
		double m2 = statistics2.getMean();
		double v2 = statistics2.getVariance();
		double n2 = statistics2.getWeight();

		double p = tTest.tTest(m1, m2, v1, v2, n1, n2);
		return 1 - p;
	}

	/**
	 * Calculates mean, variance, and total weight of a sample and corresponding weights. 
	 * 
	 * @param sample The sample
	 * @param weights The weights
	 * @return A {@link StatisticsBundle} containing the mean, variance, and total weight. 
	 */
	private StatisticsBundle weightedStatistics(double[] sample, double[] weights) {
		if (sample.length != weights.length) {
			throw new IllegalArgumentException("Sample and weights have different length.");
		}
		double weightedSum = 0;
		double sumWeights = 0;
		double sumWeightedSquares = 0;
		for (int i = 0; i < sample.length; i++) {
			weightedSum += weights[i] * sample[i];
			sumWeights += weights[i];
			sumWeightedSquares += weights[i] * sample[i] * sample[i];
		}
		double weightedMean = weightedSum / sumWeights;
		double weightedVariance = sumWeightedSquares / sumWeights - weightedMean * weightedMean;

		return new StatisticsBundle(weightedMean, weightedVariance, sumWeights);
	}
}
