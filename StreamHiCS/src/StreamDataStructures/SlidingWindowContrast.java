package streamDataStructures;

import org.apache.commons.math3.util.MathArrays;
import misc.Callback;
import misc.Contrast;
import statisticalTests.KolmogorovSmirnov;
import statisticalTests.StatisticalTest;
import weka.core.Instance;

public class SlidingWindowContrast implements Contrast {
	/**
	 * The @link{Callback} to notify on changes.
	 */
	private Callback callback;
	/**
	 * The number of {@link Instance} that are observed before the
	 * {@link Subspace} contrasts are checked again.
	 */
	private int updateInterval;
	/**
	 * To count the number of {@link Instance}s observed since the last
	 * {@link Subspace} evaluation.
	 */
	private int currentCount = 0;
	/**
	 * Number of Monte Carlo iterations in the contrast evaluation. m must be
	 * positive.
	 */
	private int m;
	/**
	 * The relative size of the conditional (sliced) sample in relation to the
	 * whole data set. The number must be positive.
	 */
	private double alpha;
	/**
	 * Data structure holding the {@link Instance}s.
	 */
	private DataStreamContainer dataStreamContainer;
	/**
	 * The {@link StatisticalTest} used to calculate the deviation of the
	 * marginal sample and the conditional (sliced) sample.
	 */
	private StatisticalTest statisticalTest;
	
	/**
	 * Creates a {@link SlidingWindowContrast} object with the specified update interval.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions of the full space.
	 * @param updateInterval
	 *            The number how many {@link Instance}s are observed between
	 *            evaluations of the correlated {@link Subspace}s.
	 * @param m
	 *            The number of Monte Carlo iterations for the estimation of the
	 *            conditional density.
	 * @param alpha
	 *            The fraction of data that should be selected in the estimation
	 *            of the conditional density.
	 */
	public SlidingWindowContrast(Callback callback, int numberOfDimensions, int updateInterval, int m, double alpha){
		this.callback = callback;
		this.updateInterval = updateInterval;
		this.m = m;
		this.alpha = alpha;
		// dataStreamContainer = new SelfOrganizingMap(numberOfDimensions, 100);
		dataStreamContainer = new SlidingWindow(numberOfDimensions, 10000);
		// Try out other tests
		statisticalTest = new KolmogorovSmirnov();
	}
	
	@Override
	public void add(Instance instance) {
		dataStreamContainer.add(instance);
		currentCount++;
		if (currentCount >= updateInterval) {
			callback.onAlarm();
			currentCount = 0;
		}
	}
	
	/**
	 * Clears all stored @link{Instance}s.
	 */
	public void clear() {
		dataStreamContainer.clear();
		currentCount = 0;
	}
	
	/**
	 * Returns the number of {@link Instance}s currently contained in this
	 * object.
	 * 
	 * @return The number of {@link Instance}s currently contained in this
	 *         object.
	 */
	public int getNumberOfInstances() {
		return dataStreamContainer.getNumberOfInstances();
	}
	
	/**
	 * Calculate the contrast for a given {@link Subspace} on a @link{SlidingWindow}. See the HiCS paper
	 * for a description of the algorithm.
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
		int[] shuffledDimensions;
		double selectionAlpha = Math.pow(alpha, 1.0 / (subspace.size() - 1));
		double deviation;
		// Do Monte Carlo iterations
		for (int i = 0; i < m; i++) {
			shuffledDimensions = subspace.getDimensions();
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// Calculate the number of instances selected per dimension
			
			// Get the projected data
			double[] dimProjectedData = dataStreamContainer
					.getProjectedData(shuffledDimensions[shuffledDimensions.length - 1]);
			// Get the randomly sliced data
			double[] slicedData = dataStreamContainer.getSlicedData(shuffledDimensions, selectionAlpha);
			// Calculate the deviation and add it to the overall sum
			deviation = statisticalTest.calculateDeviation(dimProjectedData, slicedData);
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
}
