package statisticaltests;

import streamdatastructures.DataBundle;

/**
 * This class represents a statistical test, which compares two samples and
 * returns a measure how improbable it is that they originated form the same
 * distribution (a deviation).
 * 
 * @author Vincent
 *
 */
public abstract class StatisticalTest {

	/**
	 * Calculates a deviation value for the two samples.
	 * 
	 * @param sample1
	 *            The first sample
	 * @param sample2
	 *            The second sample
	 * @return The deviation.
	 */
	public abstract double calculateDeviation(double[] sample1, double[] sample2);

	/**
	 * Calculates a deviation value for two samples with weights.
	 * 
	 * @param dataBundle1
	 *            Contains the first sample with the appropriate weights
	 * @param dataBundle2
	 *            Contains the second sample with the appropriate weights
	 * @return The deviation.
	 */
	public abstract double calculateWeightedDeviation(DataBundle dataBundle1, DataBundle dataBundle2);
}
