package centroids;

import org.apache.commons.math3.util.MathArrays;

import misc.Callback;
import misc.Contrast;
import statisticalTests.StatisticalTest;
import statisticalTests.StatisticsBundle;
import streamDataStructures.Subspace;
import weka.core.Instance;

public class CentroidContrast implements Contrast, Callback {

	private CentroidsContainer centroids;
	private int m;
	private double alpha;
	private StatisticalTest statisticalTest;
	private Callback callack;

	public CentroidContrast(Callback callback, int m, double alpha) {
		this.callack = callback;
		centroids = new FixedCentroids(this, 10, null, null, 10000, 500, 1000);
		this.m = m;
		this.alpha = alpha;
	}

	@Override
	public void add(Instance instance) {
		centroids.add(instance);
	}

	@Override
	public void clear() {
		centroids.clear();
	}

	/**
	 * Calculate the contrast for a given {@link Subspace} on
	 * a @link{SlidingWindow}. See the HiCS paper for a description of the
	 * algorithm.
	 * 
	 * @param subspace
	 *            The {@link Subspace} for which the contrast should be
	 *            calculated.
	 * @return The contrast of the given {@link Subspace}.
	 */
	@Override
	public double evaluateSubspaceContrast(Subspace subspace) {
		// Variable for collecting the intermediate results of the iterations
		double sum = 0;
		// A deviation could be NaN, so we wont count that calculation
		int numberOfCorrectTests = 0;
		double selectionAlpha = Math.pow(alpha, 1.0 / (subspace.size() - 1));
		int[] shuffledDimensions;
		double deviation;
		// Do Monte Carlo iterations
		for (int i = 0; i < m; i++) {
			shuffledDimensions = subspace.getDimensions();
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// Calculate the number of instances selected per dimension

			// Get the projected data statistics
			StatisticsBundle projectedStatistics = centroids
					.getProjectedDataStatistics(shuffledDimensions[shuffledDimensions.length - 1]);
			// Get the randomly sliced data statistics
			StatisticsBundle slicedStatistics = centroids.getSlicedDataStatistics(shuffledDimensions, selectionAlpha);
			// Calculate the deviation and add it to the overall sum
			deviation = statisticalTest.calculateDeviation(projectedStatistics, slicedStatistics);
			if (!Double.isNaN(deviation)) {
				sum += deviation;
				numberOfCorrectTests++;
			}
		}

		// Return the mean of the intermediate results. If all results were NaN,
		// 0 is returned.
		double mean = 0;
		if (numberOfCorrectTests > 0) {
			mean = sum / numberOfCorrectTests;
		}
		return mean;
	}

	@Override
	public void onAlarm() {
		callack.onAlarm();
	}

}
